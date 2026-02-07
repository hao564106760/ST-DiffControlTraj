import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_timestep_embedding(t, dim):
    # 1D timesteps -> sinusoidal embedding
    assert t.dim() == 1
    h = dim // 2
    s = math.log(10000.0) / (h - 1)
    freqs = torch.exp(torch.arange(h, device=t.device, dtype=torch.float32) * (-s))
    x = t.float()[:, None] * freqs[None, :]
    emb = torch.cat([torch.sin(x), torch.cos(x)], dim=1)
    if dim & 1:
        emb = F.pad(emb, (0, 1, 0, 0))
    return emb


def nonlinearity(x):
    return x * torch.sigmoid(x)


def Normalize(c):
    return nn.GroupNorm(num_groups=32, num_channels=c, eps=1e-6, affine=True)


class AttnBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.norm = Normalize(c)
        self.q = nn.Conv2d(c, c, 1, 1, 0)
        self.k = nn.Conv2d(c, c, 1, 1, 0)
        self.v = nn.Conv2d(c, c, 1, 1, 0)
        self.proj_out = nn.Conv2d(c, c, 1, 1, 0)

    def forward(self, x):
        h = self.norm(x)
        q, k, v = self.q(h), self.k(h), self.v(h)

        b, c, hh, ww = q.shape
        q = q.view(b, c, hh * ww).permute(0, 2, 1)     # (b, hw, c)
        k = k.view(b, c, hh * ww)                      # (b, c, hw)

        w = torch.bmm(q, k) * (c ** -0.5)
        w = F.softmax(w, dim=2)

        v = v.view(b, c, hh * ww)
        h = torch.bmm(v, w.permute(0, 2, 1)).view(b, c, hh, ww)

        return x + self.proj_out(h)
