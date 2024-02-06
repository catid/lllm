import torch.nn as nn
import torch.nn.functional as F

import math

from util import SGLU
from linear_attention.srmsnorm import FastSimpleRMSNorm
from linear_attention.lightning_attn_interface import lightning_attn_func

class CausalSelfAttention(nn.Module):
    def __init__(self, dim=64, heads=8, dropout=0.1):
        super().__init__()

        assert dim % heads == 0

        self.c_attn = nn.Linear(dim, 3 * dim, bias=False)
        self.c_proj = nn.Linear(dim, dim, bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.heads = heads
        self.dim = dim

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.dim, dim=2)
        k = k.view(B, T, self.heads, C // self.heads).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.heads, C // self.heads).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.heads, C // self.heads).transpose(1, 2) # (B, nh, T, hs)

        y = lightning_attn_func(q, k, v)

        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y



class Block(nn.Module):
    def __init__(self, dim=64, heads=8, dropout=0.1):
        super().__init__()

        self.ln1 = FastSimpleRMSNorm(dim)
        self.attn = CausalSelfAttention(dim, heads, dropout)
        self.ln2 = FastSimpleRMSNorm(dim)
        self.sglu = SGLU(d_in=dim, mult=4, d_out=dim, bias=False)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.sglu(self.ln2(x))
        return x
