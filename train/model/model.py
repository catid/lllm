import torch
from torch import nn

from mamba_ssm import Mamba2

from model.srmsnorm import FastSimpleRMSNorm

from dataclasses import dataclass

# Simple Gated Linear Unit
# From "TransNormerLLM" (Qin et al, 2024) https://arxiv.org/pdf/2307.14995.pdf
class SGLULayer(nn.Module):
    def __init__(self, d_in=64, mult=4, d_out=64, dropout=0.1, bias=False):
        super().__init__()
        d_inner = d_in * mult

        self.u_proj = nn.Linear(d_in, d_inner, bias=bias)
        self.v_proj = nn.Linear(d_in, d_inner, bias=bias)

        self.o_proj = nn.Sequential(
            nn.Linear(d_inner, d_out, bias=bias),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.o_proj(self.u_proj(x) * self.v_proj(x))

@dataclass
class LatentLanguageConfig:
    n_vocab: int = 0 # Set to tokenizer n_vocab
    dim: int = 1024
    layers: int = 16

    d_state: int = 64
    d_conv: int = 4
    expand: int = 2

    ffn_mult: int = 4 # Standardish number
    ffn_bias: bool = False # Seems fine to leave these off for perf

    dropout: float = 0.1

class LatentLanguage(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.embedding = nn.Embedding(cfg.n_vocab, cfg.dim)

        self.layers = nn.ModuleList()

        # Use FastSimpleRMSNorm for all layers
        self.norm = FastSimpleRMSNorm(cfg.dim)

        for _ in range(cfg.layers):
            self.layers.append(nn.ModuleList([
                Mamba2(d_model=cfg.dim, d_state=cfg.d_state, d_conv=cfg.d_conv, expand=cfg.expand),
                SGLULayer(d_in=cfg.dim, mult=cfg.ffn_mult, d_out=cfg.dim, dropout=cfg.dropout, bias=cfg.ffn_bias),
            ]))

        self.lm_head = nn.Linear(cfg.dim, cfg.n_vocab)

        # Tie vocab weights
        self.lm_head.weight = self.embedding.weight

    def forward(self, x):
        x = self.embedding(x)

        for attn, ffn in self.layers:
            # Repeat weights for consecutive layers
            for _ in range(2):
                x = self.norm(x)
                x = x + attn(x)

                x = self.norm(x)
                x = x + ffn(x)

        return self.lm_head(x)
