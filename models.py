

import torch
import torch.nn as nn
import torch.nn.functional as F


class TextQueryDINOLayer(nn.Module):
    """TQDKV layer: text tokens query DINO spatial patches."""

    def __init__(self, text_dim=512, dino_dim=384, num_heads=8,
                 ffn_mult=4, dropout=0.1):
        super().__init__()
        self.xa_norm = nn.LayerNorm(text_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=text_dim, num_heads=num_heads,
            kdim=dino_dim, vdim=dino_dim,
            dropout=dropout, batch_first=True)
        self.xa_gate = nn.Parameter(torch.tensor(0.65))

        self.ff_norm = nn.LayerNorm(text_dim)
        self.ffn = nn.Sequential(
            nn.Linear(text_dim, text_dim * ffn_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(text_dim * ffn_mult, text_dim),
            nn.Dropout(dropout))

    def forward(self, text_tokens, dino_patches, return_attn=False):
        h = self.xa_norm(text_tokens)
        if return_attn:
            xa_out, attn_w = self.cross_attn(
                h, dino_patches, dino_patches,
                need_weights=True, average_attn_weights=True)
        else:
            xa_out = self.cross_attn(
                h, dino_patches, dino_patches,
                need_weights=False)[0]
            attn_w = None

        text_tokens = text_tokens + torch.tanh(self.xa_gate) * xa_out
        text_tokens = text_tokens + self.ffn(self.ff_norm(text_tokens))

        if return_attn:
            return text_tokens, attn_w
        return text_tokens


class TextQueryDINOCombiner(nn.Module):


    def __init__(self, clip_dim=512, dino_dim=384, num_heads=8,
                 num_layers=2, ffn_mult=4, dropout=0.1, grid_size=16):
        super().__init__()
        fused_dim = clip_dim * 3

        self.layers = nn.ModuleList([
            TextQueryDINOLayer(clip_dim, dino_dim, num_heads, ffn_mult, dropout)
            for _ in range(num_layers)
        ])

        # Attention pooler
        self.pool_query = nn.Parameter(torch.randn(1, 1, clip_dim) * 0.02)
        self.pool_attn = nn.MultiheadAttention(
            clip_dim, num_heads, dropout=dropout, batch_first=True)
        self.pool_norm = nn.LayerNorm(clip_dim)

        self.fusion = nn.Sequential(
            nn.Linear(fused_dim, clip_dim * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(clip_dim * 2, clip_dim))

        self.gate = nn.Sequential(
            nn.Linear(fused_dim, clip_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(clip_dim, 1))
        nn.init.constant_(self.gate[-1].bias, -1.0)


        self.pooled_aux_proj = nn.Linear(clip_dim, clip_dim)

    def forward(self, text_tokens, dino_patches, ref_clip, text_cls,
                return_attn=False):
        attn_maps = []
        for layer in self.layers:
            if return_attn:
                text_tokens, aw = layer(text_tokens, dino_patches,
                                        return_attn=True)
                attn_maps.append(aw)
            else:
                text_tokens = layer(text_tokens, dino_patches)

        B = text_tokens.shape[0]
        pq = self.pool_query.expand(B, -1, -1)
        pooled, _ = self.pool_attn(pq, text_tokens, text_tokens)
        pooled = self.pool_norm(pooled).squeeze(1)       # [B, 512]


        cat = torch.cat([ref_clip, text_cls, pooled], dim=-1)
        output = self.fusion(cat)
        lam = torch.sigmoid(self.gate(cat))


        text_cls = F.normalize(text_cls, dim=-1)
        output = F.normalize(output, dim=-1)
        composed = lam * text_cls + (1 - lam) * output
        composed = F.normalize(composed, dim=-1)

      
        pooled_out = F.normalize(self.pooled_aux_proj(pooled), dim=-1)

        if return_attn:
            return composed, pooled_out, attn_maps
        return composed, pooled_out