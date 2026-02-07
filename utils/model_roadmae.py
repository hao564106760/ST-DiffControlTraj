import torch
import torch.nn as nn


class RoadMAE(nn.Module):
    def __init__(
        self,
        in_dim=6,
        embed_dim=32,
        depth=4,
        num_heads=4,
        mlp_ratio=4.0,
        drop_rate=0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        self.patchify = nn.Linear(in_dim, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))

        enc = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=drop_rate,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
        self.decoder = nn.Linear(embed_dim, in_dim)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.mask_token, std=0.02)
        self.apply(self._init_module)

    def _init_module(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, mask_idx=None):
        # x: (N, in_dim). Note: attention is O(N^2); feed reasonable N per batch.
        h = self.patchify(x).unsqueeze(0) + self.pos_embed  # (1, N, C)

        if mask_idx is not None:
            # mask_idx: 1D indices on the token axis
            h = h.clone()
            m = self.mask_token.expand(1, len(mask_idx), -1)
            h[0, mask_idx, :] = m

        z = self.norm(self.transformer(h))
        recon = self.decoder(z)
        return recon.squeeze(0), z.squeeze(0)
