import torch
import torch.nn as nn
import torch.nn.functional as F

# Simple RMS Normalization
# From "TransNormerLLM" (Qin et al, 2024) https://arxiv.org/pdf/2307.14995.pdf
# Based on "Root Mean Square Layer Normalization" (Zhang and Sennrich, 2019)
class SimpleRMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5

    def forward(self, x):
        return F.normalize(x, dim = -1) * self.scale

# Simple Gated Linear Unit
# From "TransNormerLLM" (Qin et al, 2024) https://arxiv.org/pdf/2307.14995.pdf
class SGLU(nn.Module):
    def __init__(self, d_in=64, mult=4, d_out=64, bias=False):
        super().__init__()
        d_inner = d_in * mult

        self.in_u = nn.Linear(d_in, d_inner, bias=bias)
        self.in_v = nn.Linear(d_in, d_inner, bias=bias)
        self.out_proj = nn.Linear(d_inner, d_out, bias=bias)

    def forward(self, x):
        return self.out_proj(self.in_u(x) * self.in_v(x))

# Linearized Relative Positional Encoding (Qin et al, 2023)
# https://openreview.net/forum?id=xoLyps2qWc
class LinRelPosEncoding(nn.Module):
    def __init__(self, heads=8, dim=64):
        super().__init__()
        d = heads * dim

        self.index = torch.empty(0)
        self.theta = nn.Parameter(10000**(-2 / d * torch.arange(d)).reshape(heads, 1, -1))

    def forward(self, x, offset=0):
        # x: b, h, n, d
        # offset: for k, v cache
        n = x.shape[-2]
        if self.index.shape[0] < n:
            self.index = torch.arange(n).reshape(1, -1, 1).to(x)
        index = self.index[:, :n] + offset
        theta = self.theta * index
        return torch.concat([x * torch.cos(theta), x * torch.sin(theta)], dim=-1)
