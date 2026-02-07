import os
import sys
import torch
import numpy as np
from tqdm import tqdm
from types import SimpleNamespace
from torch.utils.data import DataLoader, TensorDataset

# --- Configuration ---
PROJECT_ROOT = "./trajectory/ST-DiffControlTraj-main"
sys.path.append(PROJECT_ROOT)

from utils.Traj_UNet import Guide_UNet
from utils.config import args
from utils.utils import p_xt

# Data & Model Paths
DATA_ROOT_STRUCT = "./trajectory/ST-DiffControlTraj-main/data"
MODEL_PATH = "./trajectory/ST-DiffControlTraj-main/ST-DiffControlTraj/UNet_200.pt"
DATA_ROOT_STATS_LARGE = "./trajectory/ST-DiffControlTraj-main/data"
OUTPUT_DIR = "./trajectory/ST-DiffControlTraj-main/output"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Generation Settings
GEN_NUM = 3000
BATCH_SIZE = 64

# Spatial Filter (City Center)
FILTER_LAT_MIN, FILTER_LAT_MAX = 41.1400, 41.1800
FILTER_LON_MIN, FILTER_LON_MAX = -8.6500, -8.5800


def get_timestamps(heads, length=200, interval=15.0):
    """Calculate timestamps based on time ID (5-min slots) and fixed intervals."""
    num_samples = heads.shape[0]
    timestamps = np.zeros((num_samples, length), dtype=np.float32)
    for i in range(num_samples):
        start_sec = heads[i, 0] * 300.0  # Time ID * 300s
        offsets = np.arange(length) * interval
        timestamps[i, :] = start_sec + offsets
    return timestamps


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load configuration
    temp = {}
    for k, v in args.items():
        temp[k] = SimpleNamespace(**v)
    config = SimpleNamespace(**temp)
    config.model.attr_dim = 10

    # 1. Load Data
    head_path = os.path.join(DATA_ROOT_STRUCT, "heads_porto.npy")
    road_path = os.path.join(DATA_ROOT_STRUCT, "roadseq_porto.npy")
    stats_large = np.load(os.path.join(DATA_ROOT_STATS_LARGE, "porto_stats.npz"))

    head_np = np.load(head_path, allow_pickle=True)
    road_np = np.load(road_path, allow_pickle=True)

    # Recover real coordinates for filtering
    l_mean = np.array([stats_large["lon_mean"], stats_large["lat_mean"]])
    l_std = np.array([stats_large["lon_std"], stats_large["lat_std"]])

    # Heuristic to detect lat/lon columns
    raw_c1, raw_c2 = head_np[:, 1], head_np[:, 2]
    try_lon = raw_c1 * l_std[0] + l_mean[0]
    try_lat = raw_c2 * l_std[1] + l_mean[1]

    if try_lat.mean() > 40:
        real_lons, real_lats = try_lon, try_lat
    else:
        real_lats = raw_c1 * l_std[1] + l_mean[1]
        real_lons = raw_c2 * l_std[0] + l_mean[0]

    # 2. Filter by Location
    mask = (real_lats >= FILTER_LAT_MIN) & (real_lats <= FILTER_LAT_MAX) & \
           (real_lons >= FILTER_LON_MIN) & (real_lons <= FILTER_LON_MAX)
    valid_indices = np.where(mask)[0]

    print(f"Valid samples found: {len(valid_indices)}")
    if len(valid_indices) == 0: return

    # Sampling
    replace = len(valid_indices) < GEN_NUM
    sample_indices = np.random.choice(valid_indices, size=GEN_NUM, replace=replace)

    subset_head = head_np[sample_indices].copy()
    subset_road = road_np[sample_indices].copy()

    # 3. Time Control: Force uniform distribution (00:00 - 24:00)
    # Overwrite the Start Time ID (Attribute 0)
    uniform_time_ids = np.tile(np.arange(288), int(np.ceil(GEN_NUM / 288)))[:GEN_NUM]
    np.random.shuffle(uniform_time_ids)
    subset_head[:, 0] = uniform_time_ids

    # Prepare DataLoader
    dataset = TensorDataset(torch.tensor(subset_head).float(), torch.tensor(subset_road).long())
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 4. Initialize Model
    num_road_segments = int(road_np.max()) + 1
    unet = Guide_UNet(config, num_road_segments=num_road_segments).to(device)

    # Load Weights
    state_dict = torch.load(MODEL_PATH, map_location=device)
    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    try:
        unet.load_state_dict(new_state_dict, strict=True)
    except:
        unet.load_state_dict(new_state_dict, strict=False)

    unet.eval()

    # Diffusion Parameters
    n_steps = config.diffusion.num_diffusion_timesteps
    beta = torch.linspace(config.diffusion.beta_start, config.diffusion.beta_end, n_steps).to(device)

    # Fast Sampling (Skip steps)
    skip = max(n_steps // 100, 1)
    seq = list(range(0, n_steps, skip))
    seq_next = [-1] + list(seq[:-1])

    # 5. Generation Loop
    gen_trajs = []
    print(f"Generating {GEN_NUM} trajectories...")

    with torch.no_grad():
        for head_b, road_b in tqdm(dataloader):
            head_b, road_b = head_b.to(device), road_b.to(device)
            bs = head_b.shape[0]

            # Start from Gaussian noise
            x = torch.randn(bs, 2, config.data.traj_length, device=device)

            for i, j in zip(reversed(seq), reversed(seq_next)):
                t = torch.full((bs,), i, device=device, dtype=torch.long)
                next_t = torch.full((bs,), j, device=device, dtype=torch.long)

                pred_noise = unet(x, t, head_b, road_b)
                x = p_xt(x, pred_noise, t, next_t, beta, 0.0)

            gen_trajs.append(x.cpu().numpy())

    gen_trajs = np.concatenate(gen_trajs, axis=0)

    # 6. Inverse Normalization
    stats_small = np.load(os.path.join(DATA_ROOT_STATS_SMALL, "porto_stats.npz"))
    if "lon_mean" in stats_small:
        mean = np.array([stats_small["lon_mean"], stats_small["lat_mean"]]).reshape(1, 2, 1)
        std = np.array([stats_small["lon_std"], stats_small["lat_std"]]).reshape(1, 2, 1)
    else:
        mean = np.array(stats_small["mean"]).reshape(1, 2, 1)
        std = np.array(stats_small["std"]).reshape(1, 2, 1)

    gen_trajs_real = gen_trajs * std + mean
    gen_trajs_real = np.transpose(gen_trajs_real, (0, 2, 1))  # (N, T, 2)

    # 7. Add Timestamps
    timestamps = get_timestamps(subset_head, length=config.data.traj_length)

    # Final Output: [Lon, Lat, Time]
    output_3d = np.concatenate([gen_trajs_real, timestamps[:, :, np.newaxis]], axis=2)

    # Save Results
    np.save(os.path.join(OUTPUT_DIR, "gen_trajs_3d.npy"), output_3d)
    np.save(os.path.join(OUTPUT_DIR, "gen_trajs.npy"), gen_trajs_real)
    np.save(os.path.join(OUTPUT_DIR, "gen_timestamps.npy"), timestamps)

    print("Generation complete.")


if __name__ == "__main__":

    main()
