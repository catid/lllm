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

    n_embd: int = 768
    bias: bool = False
    dropout: float = 0.2
    n_head: int = 12
    n_layer: int = 12

    ffn_mult: int = 4

    d_state: int = 128
    d_conv: int = 4
    expand: int = 2
    headdim: int = 64

#
#Model TODO:
#* Mamba2 interleaved with SWA layers
#* SWA + Primer Spatial D-Conv 3x1: https://arxiv.org/pdf/2109.08668v2 (Figure 4)
#* Produce 2 tokens at once using 2x MLP heads

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

# https://github.com/huggingface/transformers/blob/c28d04e9e252a1a099944e325685f14d242ecdcd/src/transformers/models/gpt2/modeling_gpt2.py#L454
def _init_weights(
    module,
    n_layer,
    initializer_range=0.02,  # Now only used for embedding layer.
    rescale_prenorm_residual=True,
    n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)

class Mamba2Block(nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()

        self.mixer = Mamba2(
                    d_model=args.n_embd,
                    d_state=args.d_state,
                    d_conv=args.d_conv,
                    expand=args.expand,
                    headdim=args.headdim)

        #_init_weights(self.mixer, layer_id)

    def forward(self, x):
        return self.mixer(x)

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

class Block(nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        self.ln = FastSimpleRMSNorm(args.n_embd)
        self.attn = Mamba2Block(args, layer_id)
        self.mlp = RWKV_CMix_x060(args, layer_id)

    def forward(self, x):
        x = x + self.attn(self.ln(x))
        x = x + self.mlp(self.ln(x))
        return x

class LatentLanguage(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

        self.embedding = nn.Embedding(cfg.n_vocab, cfg.n_embd)

        self.layers = nn.ModuleList()

        for layer_id in range(cfg.n_layer):
            self.layers.append(Block(cfg, layer_id))

        self.lm_head = nn.Linear(cfg.n_embd, cfg.n_vocab)

        # Tie vocab weights
        self.lm_head.weight = self.embedding.weight

        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x, targets):
        B, N = x.size()

        # True: should take part in attention
        #attn_mask = (x != -1).unsqueeze(1).unsqueeze(2)

        x = x.masked_fill(x == -1, 0) # replace padding with some other value

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
