import os
import sys
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm

sys.path.append("./trajectory/ST-DiffControlTraj-main")
from utils.model_roadmae import RoadMAE

DATA_PATH = "./trajectory/ST-DiffControlTraj-main/data/porto_roadmae_data.pt"
RESULT_PATH = "./trajectory/ST-DiffControlTraj-main/data/road_embeddings.pt"
MODEL_SAVE_PATH = "./trajectory/ST-DiffControlTraj-main/data/roadmae_model.pth"

BATCH_SIZE = 49152
EPOCHS = 300
LR = 1e-3
MASK_RATIO = 0.6
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    if not os.path.exists(DATA_PATH):
        print(f"missing: {DATA_PATH}")
        return

    data = torch.load(DATA_PATH)
    x = data["x"]
    n = int(x.shape[0])
    print(f"roads={n} device={DEVICE.type}")

    m = RoadMAE(in_dim=6, embed_dim=32, depth=4, num_heads=4).to(DEVICE)
    opt = optim.AdamW(m.parameters(), lr=LR, weight_decay=1e-4)

    m.train()
    for ep in range(1, EPOCHS + 1):
        perm = torch.randperm(n)
        s, k = 0.0, 0

        for i in range(0, n, BATCH_SIZE):
            idx = perm[i : i + BATCH_SIZE]
            bx = x[idx].to(DEVICE)

            bs = bx.size(0)
            mc = int(bs * MASK_RATIO)
            midx = torch.randperm(bs, device=DEVICE)[:mc]

            opt.zero_grad(set_to_none=True)
            recon, _ = m(bx, midx)
            loss = F.mse_loss(recon[midx], bx[midx])
            loss.backward()
            opt.step()

            s += float(loss.item())
            k += 1

        if ep % 2 == 0:
            print(f"ep={ep:03d} loss={s / max(k, 1):.6f}")

    torch.save(m.state_dict(), MODEL_SAVE_PATH)
    print(f"saved: {MODEL_SAVE_PATH}")

    # Cache embeddings for the main model (batched to avoid OOM)
    m.eval()
    out = []
    with torch.no_grad():
        for i in tqdm(range(0, n, BATCH_SIZE), desc="embed"):
            bx = x[i : i + BATCH_SIZE].to(DEVICE)
            _, e = m(bx, None)
            out.append(e.cpu())

    emb = torch.cat(out, dim=0)
    torch.save(emb, RESULT_PATH)
    print(f"saved: {RESULT_PATH} shape={tuple(emb.shape)}")


if __name__ == "__main__":
    main()
