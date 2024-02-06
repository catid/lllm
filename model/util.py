import torch
import torch.nn as nn
import torch.nn.functional as F


# Simple RMS Normalization
# From "TransNormerLLM" (Qin et al, 2024) https://arxiv.org/pdf/2307.14995.pdf
# Based on "Root Mean Square Layer Normalization" (Zhang and Sennrich, 2019)
# Faster version: from linear_attention.srmsnorm import FastSimpleRMSNorm
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


# Rank reduction layer: Reduce number of parameters by a given factor e.g. 0.5 = 50% smaller model.
class RankReductionLayer(nn.Module):
    def __init__(self, d_in, d_out, r=0.5):
        super(RankReductionLayer, self).__init__()
        rank = int(r * d_in * d_out / (d_in + d_out))

        self.down = nn.Linear(d_in, rank)
        self.up = nn.Linear(rank, d_out)
    
    def forward(self, x):
        return self.up(self.down(x))


# Stacked Kronecker-product Layers https://openreview.net/pdf?id=ZjGr1tMVbjw
# Uses 2r*sqrt(nm) parameters instead of nm.
# For for n=512 x m=2048, r must be 256 or less to make it worthwhile.
class SKLinear(nn.Module):
    def __init__(self, n, m, scale=0.5):
        super().__init__()
        self.n = n
        self.m = m

        import math

        def round_up_sqrt(m):
            sm = int(m ** 0.5 + 0.5)
            while sm*sm > m:
                sm -= 1
            while sm*sm < m:
                sm += 1
            return sm

        self.sn = round_up_sqrt(n)
        self.np = self.sn * self.sn # n rounded up to next square
        self.sm = round_up_sqrt(m)
        self.mp = self.sm * self.sm # m rounded up to next square
        k = self.sn * self.sm

        r = int(scale * (n * m) / 2.0 / k + 0.5)

        #print(f"Using SKLinear: n={n} m={m} r={r} round_sqrt(n)={self.sn} round_sqrt(m)={self.sm} n'={self.np} m'={self.mp} k={k} reduction={(2 * r * k) * 100.0 / (n * m)}%")

        # Initialize A and B using Kaiming initialization
        self.A = nn.Parameter(torch.empty(k, r))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5)) # a is the parameter for the ReLU
        self.B = nn.Parameter(torch.empty(r, k))
        nn.init.kaiming_uniform_(self.B, a=math.sqrt(5))

    def forward(self, x):
        # Validate that the inputs are of the expected sizes
        if x.size(-1) != self.n:
            raise ValueError("Input vector must have size n")

        S = torch.matmul(self.A, self.B).reshape(self.sn, self.sm, self.sn, self.sm).transpose(1, 2).reshape(self.np, self.mp)
        S_truncated = S[:self.n, :self.m]

        return torch.matmul(x, S_truncated)
