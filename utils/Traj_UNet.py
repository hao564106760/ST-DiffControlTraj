import math
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_timestep_embedding(t, dim):
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


class RoadTopologyEncoder(nn.Module):
    def __init__(self, n_seg, d, T):
        super().__init__()
        p = "./trajectory/ST-DiffControlTraj-main/data/road_embeddings.pt"
        if os.path.exists(p):
            print(f"[RoadMAE] {p}")
            w = torch.load(p)  # (N, d0)
            if w.size(0) < n_seg:
                w = torch.cat([w, torch.zeros(n_seg - w.size(0), w.size(1))], dim=0)
            w = w[:n_seg]
            self.road_embedding = nn.Embedding.from_pretrained(w, freeze=True)
        else:
            print("[RoadMAE] init")
            self.road_embedding = nn.Embedding(n_seg, d)
        self.pos = nn.Parameter(torch.randn(1, d, T))

    def forward(self, rid):
        x = self.road_embedding(rid).transpose(1, 2)
        return x + self.pos


class ST_TransformerBlock(nn.Module):
    def __init__(self, c, heads=4, drop=0.1):
        super().__init__()
        self.n = nn.GroupNorm(32, c)
        self.a = nn.MultiheadAttention(c, heads, dropout=drop, batch_first=True)
        self.f = nn.Sequential(
            nn.Conv1d(c, c * 2, 1), nn.GELU(), nn.Dropout(drop), nn.Conv1d(c * 2, c, 1)
        )

    def forward(self, x):
        b, c, l = x.shape
        h = self.n(x).permute(0, 2, 1)
        h, _ = self.a(h, h, h)
        x = x + h.permute(0, 2, 1)
        return x + self.f(x)


class Upsample(nn.Module):
    def __init__(self, c, with_conv=True):
        super().__init__()
        self.wc = with_conv
        if self.wc:
            self.conv = nn.Conv1d(c, c, 3, 1, 1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x) if self.wc else x


class Downsample(nn.Module):
    def __init__(self, c, with_conv=True):
        super().__init__()
        self.wc = with_conv
        if self.wc:
            self.conv = nn.Conv1d(c, c, 3, 2, 0)

    def forward(self, x):
        if self.wc:
            x = F.pad(x, (1, 1), mode="constant", value=0)
            return self.conv(x)
        return F.avg_pool2d(x, kernel_size=2, stride=2)


class ResnetBlock(nn.Module):
    def __init__(self, in_c, out_c=None, conv_shortcut=False, dropout=0.1, temb_c=512):
        super().__init__()
        out_c = in_c if out_c is None else out_c
        self.in_c, self.out_c = in_c, out_c
        self.use_cs = conv_shortcut

        self.n1 = Normalize(in_c)
        self.c1 = nn.Conv1d(in_c, out_c, 3, 1, 1)
        self.tp = nn.Linear(temb_c, out_c)
        self.n2 = Normalize(out_c)
        self.dp = nn.Dropout(dropout)
        self.c2 = nn.Conv1d(out_c, out_c, 3, 1, 1)

        if in_c != out_c:
            if self.use_cs:
                self.cs = nn.Conv1d(in_c, out_c, 3, 1, 1)
            else:
                self.ns = nn.Conv1d(in_c, out_c, 1, 1, 0)

    def forward(self, x, temb):
        h = self.c1(nonlinearity(self.n1(x)))
        h = h + self.tp(nonlinearity(temb))[:, :, None]
        h = self.c2(self.dp(nonlinearity(self.n2(h))))
        if self.in_c != self.out_c:
            x = self.cs(x) if self.use_cs else self.ns(x)
        return x + h


class WideAndDeep(nn.Module):
    def __init__(self, emb_dim=128, hid=256):
        super().__init__()
        self.w = nn.Linear(7, emb_dim)
        self.e0 = nn.Embedding(288, hid)
        self.e1 = nn.Embedding(257, hid)
        self.e2 = nn.Embedding(257, hid)
        self.d1 = nn.Linear(hid * 3, emb_dim)
        self.d2 = nn.Linear(emb_dim, emb_dim)

    def forward(self, a):
        x = torch.cat([a[:, 1:6], a[:, 8:10]], dim=1)
        w = self.w(x)

        dep = torch.clamp(a[:, 0].long(), 0, 287)
        sid = torch.clamp(a[:, 6].long(), 0, 256)
        eid = torch.clamp(a[:, 7].long(), 0, 256)

        d = torch.cat([self.e0(dep), self.e1(sid), self.e2(eid)], dim=1)
        d = self.d2(F.relu(self.d1(d)))
        return w + d


class Model(nn.Module):
    def __init__(self, config, num_road_segments=20000):
        super().__init__()
        m, d = config.model, config.data
        ch, out_ch = m.ch, m.out_ch
        ch_mult = tuple(m.ch_mult)
        nrb = m.num_res_blocks
        attn_res = m.attn_resolutions
        drop = m.dropout
        in_c = m.in_channels
        T = d.traj_length
        rconv = m.resamp_with_conv

        self.config = config
        self.ch = ch
        self.temb_ch = ch * 4
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = nrb
        self.resolution = T
        self.in_channels = in_c

        self.road_dim = 32
        self.road_encoder = RoadTopologyEncoder(num_road_segments, self.road_dim, T)

        self.temb = nn.Module()
        self.temb.dense = nn.ModuleList([nn.Linear(ch, self.temb_ch), nn.Linear(self.temb_ch, self.temb_ch)])

        self.conv_in = nn.Conv1d(in_c + self.road_dim, ch, 3, 1, 1)

        curr_res = T
        in_ch_mult = (1,) + ch_mult
        self.down = nn.ModuleList()
        block_in = None
        for i in range(self.num_resolutions):
            blk, att = nn.ModuleList(), nn.ModuleList()
            block_in = ch * in_ch_mult[i]
            block_out = ch * ch_mult[i]
            for _ in range(nrb):
                blk.append(ResnetBlock(block_in, block_out, temb_c=self.temb_ch, dropout=drop))
                block_in = block_out
                if curr_res in attn_res:
                    att.append(ST_TransformerBlock(block_in))
            dn = nn.Module()
            dn.block, dn.attn = blk, att
            if i != self.num_resolutions - 1:
                dn.downsample = Downsample(block_in, rconv)
                curr_res //= 2
            self.down.append(dn)

        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(block_in, block_in, temb_c=self.temb_ch, dropout=drop)
        self.mid.attn_1 = ST_TransformerBlock(block_in)
        self.mid.block_2 = ResnetBlock(block_in, block_in, temb_c=self.temb_ch, dropout=drop)

        self.up = nn.ModuleList()
        for i in reversed(range(self.num_resolutions)):
            blk, att = nn.ModuleList(), nn.ModuleList()
            block_out = ch * ch_mult[i]
            skip_in = ch * ch_mult[i]
            for j in range(nrb + 1):
                if j == nrb:
                    skip_in = ch * in_ch_mult[i]
                blk.append(ResnetBlock(block_in + skip_in, block_out, temb_c=self.temb_ch, dropout=drop))
                block_in = block_out
                if curr_res in attn_res:
                    att.append(ST_TransformerBlock(block_in))
            up = nn.Module()
            up.block, up.attn = blk, att
            if i != 0:
                up.upsample = Upsample(block_in, rconv)
                curr_res *= 2
            self.up.insert(0, up)

        self.norm_out = Normalize(block_in)
        self.conv_out = nn.Conv1d(block_in, out_ch, 3, 1, 1)

    def forward(self, x, t, road_seq, extra=None):
        assert x.size(-1) == self.resolution

        temb = get_timestep_embedding(t, self.ch)
        temb = self.temb.dense[0](temb)
        temb = nonlinearity(temb)
        temb = self.temb.dense[1](temb)
        if extra is not None:
            temb = temb + extra

        rf = self.road_encoder(road_seq)
        x = torch.cat([x, rf], dim=1)

        hs = [self.conv_in(x)]
        for i in range(self.num_resolutions):
            for j in range(self.num_res_blocks):
                h = self.down[i].block[j](hs[-1], temb)
                if len(self.down[i].attn) > 0:
                    h = self.down[i].attn[j](h)
                hs.append(h)
            if i != self.num_resolutions - 1:
                hs.append(self.down[i].downsample(hs[-1]))

        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        for i in reversed(range(self.num_resolutions)):
            for j in range(self.num_res_blocks + 1):
                ht = hs.pop()
                if ht.size(-1) != h.size(-1):
                    h = F.pad(h, (0, ht.size(-1) - h.size(-1)))
                h = self.up[i].block[j](torch.cat([h, ht], dim=1), temb)
                if len(self.up[i].attn) > 0:
                    h = self.up[i].attn[j](h)
            if i != 0:
                h = self.up[i].upsample(h)

        h = self.conv_out(nonlinearity(self.norm_out(h)))
        return h


class Guide_UNet(nn.Module):
    def __init__(self, config, num_road_segments=20000):
        super().__init__()
        self.config = config
        self.ch = config.model.ch * 4
        self.attr_dim = config.model.attr_dim
        self.guidance_scale = config.model.guidance_scale

        self.unet = Model(config, num_road_segments=num_road_segments)
        self.guide_emb = WideAndDeep(self.ch)
        self.place_emb = WideAndDeep(self.ch)

    def forward(self, x, t, attr, road_seq):
        g = self.guide_emb(attr)
        z = torch.zeros_like(attr)
        p = self.place_emb(z)

        c = self.unet(x, t, road_seq, g)
        u = self.unet(x, t, road_seq, p)

        return c + self.guidance_scale * (c - u)
