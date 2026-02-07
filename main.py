import os
import shutil
import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from utils.config import args
from utils.EMA import EMAHelper
from utils.Traj_UNet import Guide_UNet
from utils.logger import Logger, log_info

# Env flags (avoid noisy logs / ensure CUDA device selection)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def gather(consts: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    # Extract diffusion coefficients at time indices t and reshape for broadcasting
    return consts.gather(-1, t).reshape(-1, 1, 1)


def main(config, logger, exp_dir: Path, timestamp: str) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # q(x_t | x_0): forward diffusion (adds noise according to alpha_bar[t])
    def q_xt_x0(x0: torch.Tensor, t: torch.Tensor):
        mean = gather(alpha_bar, t).sqrt() * x0
        var = 1.0 - gather(alpha_bar, t)
        eps = torch.randn_like(x0, device=x0.device)
        return mean + var.sqrt() * eps, eps

    traj_path = config.data.traj_path1
    head_path = config.data.head_path2
    road_path = config.data.road_path3

    if not (os.path.exists(traj_path) and os.path.exists(head_path) and os.path.exists(road_path)):
        raise FileNotFoundError("Missing input files. Check config paths.")

    traj = np.load(traj_path, allow_pickle=True).astype(np.float32)
    head = np.load(head_path, allow_pickle=True).astype(np.float32)
    road_seq = np.load(road_path, allow_pickle=True)

    # Expect traj as (N, 2, T); transpose if stored as (N, T, 2)
    if traj.ndim != 3:
        raise ValueError("traj must be a 3D array.")
    if traj.shape[1] != 2:
        traj = np.swapaxes(traj, 1, 2)

    traj = torch.from_numpy(traj).float()
    head = torch.from_numpy(head).float()
    road_seq = torch.from_numpy(road_seq).long()

    loader = DataLoader(
        TensorDataset(traj, head, road_seq),
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    # Road IDs are assumed to be contiguous integers in [0, max_id]
    num_road_segments = int(road_seq.max().item()) + 1
    logger.info(f"Data: traj={tuple(traj.shape)} head={tuple(head.shape)} road={tuple(road_seq.shape)}")
    logger.info(f"Init: road_segments={num_road_segments} device={device.type}")

    unet = Guide_UNet(config, num_road_segments=num_road_segments).to(device)

    # Diffusion schedule
    n_steps = int(config.diffusion.num_diffusion_timesteps)
    beta = torch.linspace(config.diffusion.beta_start, config.diffusion.beta_end, n_steps, device=device)
    alpha = 1.0 - beta
    alpha_bar = torch.cumprod(alpha, dim=0)

    optimizer = torch.optim.AdamW(unet.parameters(), lr=1e-4)

    # EMA improves sampling stability; keep disabled if you want exact raw weights
    ema_helper = None
    if config.model.ema:
        ema_helper = EMAHelper(mu=config.model.ema_rate)
        ema_helper.register(unet)

    model_dir = exp_dir / "models" / f"{timestamp}"
    results_dir = exp_dir / "results"
    model_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    losses = []
    epoch_losses = []

    logger.info("Training start")
    for epoch in range(1, int(config.training.n_epochs) + 1):
        unet.train()
        epoch_sum, epoch_n = 0.0, 0

        for x0, h, r in loader:
            x0 = x0.to(device, non_blocking=True)
            h = h.to(device, non_blocking=True)
            r = r.to(device, non_blocking=True)

            # Antithetic sampling for timesteps (variance reduction)
            t = torch.randint(0, n_steps, size=(len(x0) // 2 + 1,), device=device)
            t = torch.cat([t, n_steps - t - 1], dim=0)[: len(x0)]

            xt, noise = q_xt_x0(x0, t)
            pred = unet(xt.float(), t, h, r)

            loss = F.mse_loss(noise.float(), pred)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            if ema_helper is not None:
                ema_helper.update(unet)

            lv = float(loss.item())
            losses.append(lv)
            epoch_sum += lv
            epoch_n += 1

        avg = epoch_sum / max(epoch_n, 1)
        epoch_losses.append(avg)
        logger.info(f"Epoch {epoch}: loss={avg:.6f}")

        # Persist a single loss curve for quick sanity check
        plt.figure(figsize=(6, 4))
        plt.plot(epoch_losses, linestyle="-", linewidth=1.0)
        plt.xlabel("Epoch")
        plt.ylabel("Avg MSE")
        plt.grid(True, alpha=0.3)
        plt.savefig(results_dir / "loss_curve_epoch.png", dpi=600, bbox_inches="tight")
        plt.close()

        if epoch % 10 == 0:
            torch.save(unet.state_dict(), model_dir / f"UNet_{epoch}.pt")
            np.save(results_dir / f"loss_iter_until_epoch{epoch}.npy", np.asarray(losses, dtype=np.float32))


if __name__ == "__main__":
    cfg = {k: SimpleNamespace(**v) for k, v in args.items()}
    config = SimpleNamespace(**cfg)

    root_dir = Path(__file__).resolve().parents[0]
    result_name = "{}_steps={}_len={}_{}_bs={}".format(
        config.data.dataset,
        config.diffusion.num_diffusion_timesteps,
        config.data.traj_length,
        config.diffusion.beta_end,
        config.training.batch_size,
    )
    exp_dir = root_dir / "ST-DiffControlTraj" / result_name
    for d in ["results", "models", "logs", "Files"]:
        (exp_dir / d).mkdir(parents=True, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%m-%d-%H-%M-%S")

    # Snapshot key source files for reproducibility
    files_dir = exp_dir / "Files" / f"{timestamp}"
    files_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy("./utils/config.py", files_dir)
    shutil.copy("./utils/Traj_UNet.py", files_dir)

    logger = Logger(
        __name__,
        log_path=exp_dir / "logs" / f"{timestamp}.log",
        colorize=True,
    )
    log_info(config, logger)
    logger.info(f"Output dir: {exp_dir}")

    main(config, logger, exp_dir, timestamp)
