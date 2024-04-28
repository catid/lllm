import torch
import torch.nn as nn
import torch.nn.functional as F


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

class TokenMerger(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.down = nn.Linear(dim * 2, dim)

    def forward(self, x):
        batch_size, num_tokens, num_features = x.shape
        assert num_tokens % 2 == 0, "num_tokens must be even"

        x = x.view(batch_size, num_tokens // 2, num_features * 2)
        x = self.down(x)

        return x

class TokenExpander(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.up = nn.Linear(dim, dim * 2)

    def forward(self, x):
        batch_size, num_tokens, num_features = x.shape

        x = self.up(x)
        x = x.view(batch_size, num_tokens * 2, num_features // 2)  

        return x
