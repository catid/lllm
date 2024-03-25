import torch
from torch import nn
from einops import rearrange

import xformers.ops as xops

def select_top_tokens_consecutive(input_tensor, scores):
    B, N, _ = input_tensor.shape

    # Adjust for odd N by only considering up to the last pair
    effective_N = N - (N % 2)
    scores = scores[:, :effective_N]
    input_tensor = input_tensor[:, :effective_N, :]

    # Reshape scores to (B, N//2, 2) to consider consecutive pairs
    scores_reshaped = scores.view(B, -1, 2)

    # Find indices of maximum scores in each consecutive pair
    _, max_indices = torch.max(scores_reshaped, dim=-1)

    # Calculate global indices in the flattened version of the input tensor
    row_indices = torch.arange(B, device=scores.device)[:, None]
    global_indices = max_indices + torch.arange(0, effective_N, 2, device=scores.device)[None, :]

    # Select tokens based on calculated indices
    selected_tokens = input_tensor[row_indices, global_indices]

    return selected_tokens

class DownsampleMHA(nn.Module):
    def __init__(self, d_in, d_out, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        self.dropout = dropout
        self.heads = heads
        inner_dim = dim_head * heads

        self.to_q = nn.Linear(d_in, inner_dim, bias = False)
        self.to_k = nn.Linear(d_in, inner_dim, bias = False)
        self.to_v = nn.Linear(d_in, inner_dim, bias = False)

        self.score = nn.Linear(inner_dim, 1)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, d_out),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        q = rearrange(q, 'b n (h d) -> b n h d', h=self.heads)
        k = rearrange(k, 'b n (h d) -> b n h d', h=self.heads)
        v = rearrange(v, 'b n (h d) -> b n h d', h=self.heads)

        out = xops.memory_efficient_attention(
            q, k, v,
            attn_bias=xops.LowerTriangularMask(),
            p=self.dropout,
            scale=None)

        out = rearrange(out, 'b n h d -> b n (h d)', h=self.heads)

        # Calculate a score for each (b, n) token
        score = self.score(out).squeeze(-1)

        out = self.to_out(out)

        out = out + x # Residual connection

        selected = select_top_tokens_consecutive(out, score)

        return out, selected

class CausalMHA(nn.Module):
    def __init__(self, d_in, d_out, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        self.dropout = dropout
        self.heads = heads
        inner_dim = dim_head * heads

        self.to_q = nn.Linear(d_in, inner_dim, bias = False)
        self.to_k = nn.Linear(d_in, inner_dim, bias = False)
        self.to_v = nn.Linear(d_in, inner_dim, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, d_out),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        q = rearrange(q, 'b n (h d) -> b n h d', h=self.heads)
        k = rearrange(k, 'b n (h d) -> b n h d', h=self.heads)
        v = rearrange(v, 'b n (h d) -> b n h d', h=self.heads)

        out = xops.memory_efficient_attention(
            q, k, v,
            attn_bias=xops.LowerTriangularMask(),
            p=self.dropout,
            scale=None)

        out = rearrange(out, 'b n h d -> b n (h d)', h=self.heads)
        return self.to_out(out)

def _materialize_causal_mask(q, kv) -> torch.Tensor:
    dtype = q.dtype
    B, QN, H, _ = q.shape
    _, KVN, _, _ = kv.shape
    device = q.device

    create_as = dtype if dtype is not torch.bfloat16 else torch.float32
    tensor = torch.full(  # type: ignore
        torch.Size([B, H, QN, KVN]),
        dtype=create_as,
        fill_value=1,
        device=device,
    )

    mask = torch.triu(tensor, diagonal=-2).to(dtype)  # type: ignore
    mask = torch.log(mask)

    return mask.to(dtype)

class UpsampleMHA(nn.Module):
    def __init__(self, d_in_full, d_in_downsampled, d_out, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        self.dropout = dropout
        self.heads = heads
        inner_dim = dim_head * heads

        self.to_q = nn.Linear(d_in_full, inner_dim, bias = False)
        self.to_k = nn.Linear(d_in_downsampled, inner_dim, bias = False)
        self.to_v = nn.Linear(d_in_downsampled, inner_dim, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, d_out),
            nn.Dropout(dropout)
        )

    def forward(self, unet_full, unet_downsampled):
        q = self.to_q(unet_full)
        k = self.to_k(unet_downsampled)
        v = self.to_v(unet_downsampled)

        # Repeat 2x downsampled tokens in pairs to line up with old token sequence
        k = k.repeat(1, 2, 1)
        v = v.repeat(1, 2, 1)

        q = rearrange(q, 'b n (h d) -> b n h d', h=self.heads)
        k = rearrange(k, 'b n (h d) -> b n h d', h=self.heads)
        v = rearrange(v, 'b n (h d) -> b n h d', h=self.heads)

        attn_bias = _materialize_causal_mask(q, k)

        out = xops.memory_efficient_attention(
            q, k, v,
            attn_bias=attn_bias,
            p=self.dropout,
            scale=None)

        out = rearrange(out, 'b n h d -> b n (h d)', h=self.heads)
        return self.to_out(out)
