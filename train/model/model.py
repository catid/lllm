import torch
from torch import nn
import torch.nn.functional as F

from flash_attn import flash_attn_func
from .srmsnorm import FastSimpleRMSNorm
from mamba_ssm import Mamba2

import math
from dataclasses import dataclass

@dataclass
class LatentLanguageConfig:
    n_vocab: int = 0 # Set to tokenizer n_vocab
    padding_token: int = 0

    n_embd: int = 512
    bias: bool = False
    dropout: float = 0.2
    n_head: int = 8
    n_layer: int = 6
    window_size: int = (2048, 0)

    ffn_mult: int = 4

    d_state: int = 128
    d_conv: int = 4
    expand: int = 2
    headdim: int = 64

# activation functions

class ReLUSquared(nn.Module):
    """ Introduced by Noam's Primer paper: https://arxiv.org/pdf/2109.08668v2 """
    def forward(self, x):
        return F.relu(x) ** 2

# FFN layer

class RWKV_CMix_x060(nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        with torch.no_grad():  # fancy init of time_mix
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, args.n_embd)
            for i in range(args.n_embd):
                ddd[0, 0, i] = i / args.n_embd
            self.time_maa_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_r = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))

        self.key = nn.Linear(args.n_embd, args.n_embd * args.ffn_mult, bias=False)
        self.receptance = nn.Linear(args.n_embd, args.n_embd, bias=False)
        self.value = nn.Linear(args.n_embd * args.ffn_mult, args.n_embd, bias=False)

        self.act = ReLUSquared()

    def forward(self, x):
        xx = self.time_shift(x) - x
        xk = x + xx * self.time_maa_k
        xr = x + xx * self.time_maa_r

        k = self.key(xk)
        k = self.act(k)
        kv = self.value(k)
        # TBD: Gate per head here instead
        # TBD: Gate next layer per head via mask
        return torch.sigmoid(self.receptance(xr)) * kv

# Adapted from https://nn.labml.ai/transformers/primer_ez/index.html
class SpatialDepthWiseConvolution(nn.Module):
    def __init__(self, head_dim: int, kernel_size: int = 3):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(
            in_channels=head_dim,
            out_channels=head_dim,
            kernel_size=(kernel_size,),
            padding=(kernel_size - 1,),
            groups=head_dim)

    def forward(self, x: torch.Tensor):
        # x has shape [B, heads, seq_len, head_dim]
        B, heads, seq_len, head_dim = x.shape
        # Permute to [B, heads, head_dim, seq_len]
        x = x.permute(0, 1, 3, 2)
        # Change the shape to [B * heads, head_dim, seq_len]
        x = x.view(B * heads, head_dim, seq_len)
        x = self.conv(x)
        # Crop the right most kernel_size - 1 results since we padded both sides
        x = x[:, :, :-(self.kernel_size - 1)]
        # Reshape to [B, heads, head_dim, seq_len]
        x = x.view(B, heads, head_dim, seq_len)
        # Permute to [B, heads, seq_len, head_dim]
        x = x.permute(0, 1, 3, 2)
        return x

# Multi-query attention: https://arxiv.org/pdf/1911.02150
# Primer-EZ DConv: https://arxiv.org/pdf/2109.08668v2
class MultiQueryAttentionDConv(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.head_dim = config.n_embd // config.n_head  # head dimension
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.window_size = config.window_size if hasattr(config, 'window_size') else (-1, -1)
        # query projections for all heads, key and value projections for one head
        self.c_attn = nn.Linear(config.n_embd, config.n_embd + 2 * self.head_dim, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.resid_dropout = nn.Dropout(config.dropout)
        # causal depthwise convolutions for Q, K, and V
        self.q_conv = SpatialDepthWiseConvolution(config.n_embd)
        self.k_conv = SpatialDepthWiseConvolution(self.head_dim)
        self.v_conv = SpatialDepthWiseConvolution(self.head_dim)

    def forward(self, x, mask=None):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query for all heads, and key & value for a single head
        qkv = self.c_attn(x)
        q = qkv[..., :self.n_embd]
        k = qkv[..., self.n_embd:self.n_embd + C // self.n_head]
        v = qkv[..., -C // self.n_head:]

        # split query into multiple heads
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        # k and v remain single-headed
        k = k.unsqueeze(1) # (B, 1, T, hs)
        v = v.unsqueeze(1) # (B, 1, T, hs)

        q = self.q_conv(q)
        k = self.k_conv(k)
        v = self.v_conv(v)

        # efficient attention using Flash Attention CUDA kernels
        dropout_p = self.dropout if self.training else 0
        y = flash_attn_func(q, k, v, dropout_p=dropout_p, causal=True, window_size=self.window_size)

        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class Block(nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        self.ln = FastSimpleRMSNorm(args.n_embd)
        self.mamba = Mamba2(
                    d_model=args.n_embd,
                    d_state=args.d_state,
                    d_conv=args.d_conv,
                    expand=args.expand,
                    headdim=args.headdim)
        self.attn = MultiQueryAttentionDConv(args)
        self.mlp = RWKV_CMix_x060(args, layer_id)

    def forward(self, x):
        x = x + self.mamba(self.ln(x))
        x = x + self.attn(self.ln(x))
        x = x + self.mlp(self.ln(x))
        return x

# Predicts next two tokens
class LatentLanguage(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args

        self.embedding = nn.Embedding(args.n_vocab, args.n_embd)

        self.layers = nn.ModuleList()

        for layer_id in range(args.n_layer):
            self.layers.append(Block(args, layer_id))

        self.ln = FastSimpleRMSNorm(args.n_embd)
        self.pred1 = RWKV_CMix_x060(args, args.n_layer)
        self.pred2 = RWKV_CMix_x060(args, args.n_layer)

        self.lm_head = nn.Linear(args.n_embd, args.n_vocab)

        # Tie vocab weights
        self.lm_head.weight = self.embedding.weight

        self.drop = nn.Dropout(args.dropout)

    def forward(self, x, targets_1, targets_2):
        B, N = x.size()

        # True: should take part in attention
        #attn_mask = (x != -1).unsqueeze(1).unsqueeze(2)

        x = x.masked_fill(x == -1, self.args.padding_token)

        x = self.embedding(x)

        x = self.drop(x)

        for block in self.layers:
            x = block(x)

        # Apply output prediction heads
        x1 = self.pred1(self.ln(x))
        x2 = self.pred2(self.ln(x))

        # Use the same lm_head for both predictions
        logits_1 = self.lm_head(x1)
        logits_2 = self.lm_head(x2)

        # Calculate losses using provided targets
        loss_1 = F.cross_entropy(
            logits_1.view(-1, logits_1.size(-1)),
            targets_1.view(-1),
            ignore_index=-1,
            reduction='mean')

        loss_2 = F.cross_entropy(
            logits_2.view(-1, logits_2.size(-1)),
            targets_2.view(-1),
            ignore_index=-1,
            reduction='mean')

        loss = (loss_1 + loss_2) * 0.5

        return logits_1, loss
