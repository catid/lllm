import torch
from torch import nn
import torch.nn.functional as F

from torch.utils.checkpoint import checkpoint

import math

from dataclasses import dataclass

#
#Model TODO:
#* Mamba2 interleaved with SWA layers
#* SWA + Primer Spatial D-Conv 3x1: https://arxiv.org/pdf/2109.08668v2 (Figure 4)
#* Produce 2 tokens at once using 2x MLP heads
#* RWKV_CMix_x060
#* SRMSNorm

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

        self.key = nn.Linear(args.n_embd, args.dim_ffn, bias=False)
        self.receptance = nn.Linear(args.n_embd, args.n_embd, bias=False)
        self.value = nn.Linear(args.dim_ffn, args.n_embd, bias=False)

    def forward(self, x):
        xx = self.time_shift(x) - x
        xk = x + xx * self.time_maa_k
        xr = x + xx * self.time_maa_r

        k = self.key(xk)
        k = torch.relu(k) ** 2
        kv = self.value(k)
        # TBD: Gate per head here instead
        # TBD: Gate next layer per head via mask
        return torch.sigmoid(self.receptance(xr)) * kv

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

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
        y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=self.dropout if self.training else 0, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x, mask):
        x = x + self.attn(self.ln_1(x), mask=mask)
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class LatentLanguageConfig:
    n_vocab: int = 0 # Set to tokenizer n_vocab

    n_embd: int = 768
    bias: bool = False
    dropout: float = 0.2
    n_head: int = 12
    layers: int = 12
    block_size: int = 256

    enable_checkpointing: bool = False

class LatentLanguage(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

        self.embedding = nn.Embedding(cfg.n_vocab, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)

        self.layers = nn.ModuleList()

        for _ in range(cfg.layers):
            self.layers.append(Block(config=cfg))

        self.lm_head = nn.Linear(cfg.n_embd, cfg.n_vocab)

        # Tie vocab weights
        self.lm_head.weight = self.embedding.weight

        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x, targets):
        B, N = x.size()

        # True: should take part in attention
        attn_mask = (x != -1).unsqueeze(1).unsqueeze(2)

        x = x.masked_fill(x == -1, 0) # replace padding with some other value

        def custom_forward(x):
            x = self.embedding(x)

            pos = torch.arange(0, N, dtype=torch.long, device=x.device)
            pos_emb = self.pos_emb(pos)

            x = self.drop(x + pos_emb)

            for block in self.layers:
                x = block(x, mask=attn_mask)

            logits = self.lm_head(x)

            return logits

        if self.cfg.enable_checkpointing:
            logits = checkpoint(custom_forward, x, use_reentrant=False, preserve_rng_state=True)
        else:
            logits = custom_forward(x)

        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=-1,
            reduction='mean')

        return logits, loss
