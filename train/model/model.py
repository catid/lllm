import torch
from torch import nn
import torch.nn.functional as F

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

    ffn_mult: int = 4

    d_state: int = 128
    d_conv: int = 4
    expand: int = 2
    headdim: int = 64

#
#Model TODO:
#* SWA + Primer Spatial D-Conv 3x1: https://arxiv.org/pdf/2109.08668v2 (Figure 4)
#* Produce 2 tokens at once using 2x MLP heads

# activation functions

class ReLUSquared(nn.Module):
    """ Introduced by Noam's Primer paper: https://arxiv.org/pdf/2109.08668v2 """
    def forward(self, x):
        return F.relu(x) ** 2

class LaplacianActFn(nn.Module):
    """ https://arxiv.org/abs/2209.10655 claims this is more stable than Relu squared """

    def forward(self, x):
        mu = math.sqrt(0.5)
        std = math.sqrt((4 * math.pi) ** -1)
        return (1 + torch.special.erf((x - mu) / (std * math.sqrt(2)))) * 0.5

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

        self.act = LaplacianActFn()

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

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

    def forward(self, x, mask=None):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # efficient attention using Flash Attention CUDA kernels
        dropout = self.dropout if self.training else 0
        y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=dropout, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class MultiQueryAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # query projections for all heads, key and value projections for one head
        self.c_attn = nn.Linear(config.n_embd, config.n_embd + 2 * (config.n_embd // config.n_head), bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

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

        # efficient attention using Flash Attention CUDA kernels
        dropout = self.dropout if self.training else 0
        y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=dropout, is_causal=True)

        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class MultiQueryAttentionPEZ(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.head_dim = config.n_embd // config.n_head
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

        # query projections for all heads, key and value projections for one head
        self.c_attn = nn.Linear(config.n_embd, config.n_embd + 2 * self.head_dim, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.resid_dropout = nn.Dropout(config.dropout)

        # Causal depthwise convolutions for Q, K, and V
        self.q_conv = nn.Conv1d(self.n_embd, self.n_embd, kernel_size=3, padding=2, groups=self.n_embd)
        self.k_conv = nn.Conv1d(self.head_dim, self.head_dim, kernel_size=3, padding=2, groups=self.head_dim)
        self.v_conv = nn.Conv1d(self.head_dim, self.head_dim, kernel_size=3, padding=2, groups=self.head_dim)

    def causal_conv1d(self, x, conv):
        # Ensure causal convolution by masking future timesteps
        padding = conv.padding[0]
        x = F.pad(x, (padding, 0))  # Pad left side only
        x = conv(x)
        return x[:, :, :x.size(2) - padding]  # Remove extra padding on right side

    def forward(self, x, mask=None):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query for all heads, and key & value for a single head
        qkv = self.c_attn(x)
        q = qkv[..., :self.n_embd]
        k = qkv[..., self.n_embd:self.n_embd + self.head_dim]
        v = qkv[..., -self.head_dim:]

        # Apply causal depthwise convolutions
        q = self.causal_conv1d(q.transpose(1, 2), self.q_conv).transpose(1, 2)
        k = self.causal_conv1d(k.transpose(1, 2), self.k_conv).transpose(1, 2)
        v = self.causal_conv1d(v.transpose(1, 2), self.v_conv).transpose(1, 2)

        # split query into multiple heads
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2) # (B, nh, T, hs)
        # k and v remain single-headed
        k = k.unsqueeze(1) # (B, 1, T, hs)
        v = v.unsqueeze(1) # (B, 1, T, hs)

        # efficient attention using Flash Attention CUDA kernels
        dropout = self.dropout if self.training else 0
        y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=dropout, is_causal=True)

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
        self.attn = MultiQueryAttention(args)
        self.mlp = RWKV_CMix_x060(args, layer_id)

    def forward(self, x):
        x = x + self.mamba(self.ln(x))
        x = x + self.attn(self.ln(x))
        x = x + self.mlp(self.ln(x))
        return x

class LatentLanguage(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args

        self.embedding = nn.Embedding(args.n_vocab, args.n_embd)

        self.layers = nn.ModuleList()

        for layer_id in range(args.n_layer):
            self.layers.append(Block(args, layer_id))

        self.lm_head = nn.Linear(args.n_embd, args.n_vocab)

        # Tie vocab weights
        self.lm_head.weight = self.embedding.weight

        self.drop = nn.Dropout(args.dropout)

    def forward(self, x, targets):
        B, N = x.size()

        # True: should take part in attention
        #attn_mask = (x != -1).unsqueeze(1).unsqueeze(2)

        x = x.masked_fill(x == -1, self.args.padding_token)

        x = self.embedding(x)

        x = self.drop(x)

        for block in self.layers:
            x = block(x)

        logits = self.lm_head(x)

        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=-1,
            reduction='mean')

        return logits, loss
