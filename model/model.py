import torch
from torch import nn

import einops
import math

from flash_attn.flash_attn_interface import flash_attn_kvpacked_func

from model.srmsnorm import FastSimpleRMSNorm

from dataclasses import dataclass

@dataclass
class LatentLanguageConfig:
    vocab_size: int = 65529 # Match RWKV
    dim: int = 512 # Narrower
    layers: int = 12 # Small

    kv_heads: int = 8 # Half Q heads
    q_heads: int = 16 # Standardish number
    dim_head: int = 64 # Decent size

    ffn_mult: int = 4 # Standardish number
    ffn_bias: bool = False # Seems fine to leave these off for perf

    dropout: float = 0.1

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

# From https://github.com/ofirpress/attention_with_linear_biases/blob/4b92f28a005ead2567abe2359f633e73e08f3833/fairseq/models/transformer.py#L742
def get_alibi_slopes(nheads):
    def get_slopes_power_of_2(nheads):
        start = 2 ** (-(2 ** -(math.log2(nheads) - 3)))
        ratio = start
        return [start * ratio**i for i in range(nheads)]

    if math.log2(nheads).is_integer():
        return get_slopes_power_of_2(nheads)
    else:
        closest_power_of_2 = 2 ** math.floor(math.log2(nheads))
        return (
            get_slopes_power_of_2(closest_power_of_2)
            + get_alibi_slopes(2 * closest_power_of_2)[0::2][: nheads - closest_power_of_2]
        )

# Group Query Attention (GQA) as used in LLaMA-3 models
class GQALayer(nn.Module):
    def __init__(self, dim, kv_heads = 8, q_heads = 16, dim_head = 64, dropout = 0.): 
        super().__init__()

        assert dim % kv_heads == 0, "dim must be divisible by kv_heads"
        assert dim % q_heads == 0, "dim must be divisible by q_heads"
        assert dim % dim_head == 0, "dim must be divisible by dim_head"

        self.dropout = dropout
        self.heads_q = q_heads
        self.heads_kv = kv_heads

        inner_dim_q = dim_head * self.heads_q
        inner_dim_kv = dim_head * self.heads_kv

        self.kv_proj = nn.Sequential(
            nn.Linear(dim, 2 * inner_dim_kv, bias=False), 
            nn.Dropout(dropout)
        )
        self.q_proj = nn.Sequential(
            nn.Linear(dim, inner_dim_q, bias=False),
            nn.Dropout(dropout)
        )
        self.o_proj = nn.Sequential(
            nn.Linear(inner_dim_q, dim),
            nn.Dropout(dropout)
        )

        alibi_slopes = torch.tensor(get_alibi_slopes(kv_heads))
        self.register_buffer("alibi_slopes", alibi_slopes, persistent=False)

        self.softmax_scale = 1.0 / math.sqrt(dim_head)

    def forward(self, x):
        kv = self.kv_proj(x)
        q = self.q_proj(x)

        kv = einops.rearrange(kv, 'b n (r h d) -> b n r h d', r=2, h=self.heads_q)   
        q = einops.rearrange(q, 'b n (h d) -> b n h d', h=self.heads_q) 

        out = flash_attn_kvpacked_func(
            q,
            kv,
            dropout_p=self.dropout,
            softmax_scale=self.softmax_scale,
            alibi_slopes=self.alibi_slopes,
            causal=True,
            window_size=(-1, -1),
            deterministic=False)

        out = einops.rearrange(out, 'b n h d -> b n (h d)', h=self.heads_q)
        return self.o_proj(out)

class LatentLanguage(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.embedding = nn.Embedding(cfg.vocab_size, cfg.dim)

        self.layers = nn.ModuleList()

        # Use FastSimpleRMSNorm for all layers
        self.norm = FastSimpleRMSNorm(cfg.dim)

        # Use Simple GLU for all layers
        self.outer_attn = GQALayer(dim=cfg.dim, kv_heads=cfg.kv_heads, q_heads=cfg.q_heads, dim_head=cfg.dim_head, dropout=cfg.dropout)
        self.outer_ffn = SGLULayer(d_in=cfg.dim, mult=cfg.ffn_mult, d_out=cfg.dim, bias=cfg.ffn_bias)

        for _ in range(cfg.layers - 1):
            self.layers.append(nn.ModuleList([
                GQALayer(dim=cfg.dim, kv_heads=cfg.kv_heads, q_heads=cfg.q_heads, dim_head=cfg.dim_head, dropout=cfg.dropout),
                SGLULayer(d_in=cfg.dim, mult=cfg.ffn_mult, d_out=cfg.dim, bias=cfg.ffn_bias),
            ]))

        self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size)

        # Tie vocab weights
        self.lm_head.weight = self.embedding.weight

    def forward(self, x):
        x = self.embedding(x)

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

        return self.lm_head(x)
