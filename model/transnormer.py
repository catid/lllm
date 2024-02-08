import torch.nn as nn

from util import SGLU, LinRelPosEncoding
from linear_attention.srmsnorm import FastSimpleRMSNorm
from linear_attention.lightning_attn_interface import lightning_attn_func

class CausalSelfAttention(nn.Module):
    def __init__(self, dim=64, heads=8, dropout=0.1):
        super().__init__()

        self.heads = heads
        self.dim = dim
        assert dim % heads == 0

        self.lrpe = LinRelPosEncoding(num_heads=heads, embed_dim=dim//heads)
        self.offset = 0

        self.c_attn = nn.Linear(dim, 3 * dim, bias=False)
        self.c_proj = nn.Linear(dim, dim, bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.dim, dim=2)
        k = k.view(B, T, self.heads, C // self.heads).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.heads, C // self.heads).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.heads, C // self.heads).transpose(1, 2) # (B, nh, T, hs)

        q = self.lrpe(q, offset=self.offset)
        k = self.lrpe(k, offset=self.offset)

        y = lightning_attn_func(q, k, v)

        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class Transnormer(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.ln1 = FastSimpleRMSNorm(args.dim)
        self.attn = CausalSelfAttention(args.dim, args.heads, args.dropout)
        self.ln2 = FastSimpleRMSNorm(args.dim)
        self.sglu = SGLU(d_in=args.dim, mult=args.ff_mult, d_out=args.dim, bias=args.ff_bias)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.sglu(self.ln2(x))
        return x
