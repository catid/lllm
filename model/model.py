import torch
from torch import nn

from model.attention import CausalMHA
from model.srmsnorm import FastSimpleRMSNorm
from model.util import SGLU

from dataclasses import dataclass

@dataclass
class LatentLanguageConfig:
    vocab_size: int = 65529
    dim: int = 512
    layers: int = 12

    heads: int = 8
    dim_head: int = 64

    ffn_mult: int = 4
    ffn_bias: bool = False

    dropout: float = 0.1

class LatentLanguage(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.embedding = nn.Embedding(cfg.vocab_size, cfg.dim)

        self.layers = nn.ModuleList()

        # Use FastSimpleRMSNorm for all layers
        self.norm = FastSimpleRMSNorm(cfg.dim)

        # Use Simple GLU for all layers
        self.outer_attn = CausalMHA(d_in=cfg.dim, d_out=cfg.dim, heads=cfg.heads, dim_head=cfg.dim_head, dropout=cfg.dropout)
        self.outer_ffn = SGLU(d_in=cfg.dim, mult=cfg.ffn_mult, d_out=cfg.dim, bias=cfg.ffn_bias)

        for _ in range(cfg.layers - 1):
            self.layers.append(nn.ModuleList([
                CausalMHA(d_in=cfg.dim, d_out=cfg.dim, heads=cfg.heads, dim_head=cfg.dim_head, dropout=cfg.dropout),
                SGLU(d_in=cfg.dim, mult=cfg.ffn_mult, d_out=cfg.dim, bias=cfg.ffn_bias),
            ]))

        self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size)

        # Tie vocab weights
        self.lm_head.weight = self.embedding.weight

    def forward(self, x):
        x = self.embedding(x)

        # TODO: Try Mamba at first layer
        # TODO: Try Windowed attention

        # One extra attention layer at the start
        x = self.norm(x)
        x = x + self.outer_attn(x)

        for attn, ffn in self.layers:
            # Repeat weights for consecutive layers
            for _ in range(2):
                x = self.norm(x)
                x = x + attn(x)

                x = self.norm(x)
                x = x + ffn(x)

        # One extra FFN layer at the end
        x = self.norm(x)
        x = x + self.outer_ffn(x)

        # TODO: Try Mamba near last layer

        return self.lm_head(x)
