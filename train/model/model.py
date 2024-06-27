import torch
from torch import nn
import torch.nn.functional as F

from flash_attn import flash_attn_func
from .srmsnorm import FastSimpleRMSNorm
from mamba_ssm import Mamba2

import math
import bitsandbytes as bnb
from dataclasses import dataclass

from torch.utils.checkpoint import checkpoint

@dataclass
class LatentLanguageConfig:
    n_vocab: int = 0 # Set to tokenizer n_vocab
    padding_token: int = 0

    n_embd: int = 64

    d_model: int = 512
    bias: bool = False
    dropout: float = 0.2
    n_head: int = 8
    n_layer: int = 8
    window_size: int = (1024, 0)

    ffn_mult: int = 4

    d_state: int = 128
    d_conv: int = 4
    expand: int = 2
    headdim: int = 64

    bnb_embedding: bool = False

# activation functions

class ReLUSquared(nn.Module):
    """ Introduced by Noam's Primer paper: https://arxiv.org/pdf/2109.08668v2 """
    def forward(self, x):
        return F.relu(x) ** 2

# FFN layer

class RWKV_CMix_x060(nn.Module):
    def __init__(self, args, layer_id, d_in=None, d_out=None):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        if d_in is None:
            d_in = args.d_model
        if d_out is None:
            d_out = args.d_model
        d_inner = d_in * args.ffn_mult

        with torch.no_grad():  # fancy init of time_mix
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, d_in)
            for i in range(d_in):
                ddd[0, 0, i] = i / d_in
            self.time_maa_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_r = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))

        self.key = nn.Linear(d_in, d_inner, bias=False)
        self.receptance = nn.Linear(d_in, d_out, bias=False)
        self.value = nn.Linear(d_inner, d_out, bias=False)

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
        x = x.reshape(B * heads, head_dim, seq_len)
        x = self.conv(x)
        # Crop the right most kernel_size - 1 results since we padded both sides
        x = x[:, :, :-(self.kernel_size - 1)]
        # Reshape to [B, heads, head_dim, seq_len]
        x = x.reshape(B, heads, head_dim, seq_len)
        # Permute to [B, heads, seq_len, head_dim]
        x = x.permute(0, 1, 3, 2)
        return x

# Multi-query attention: https://arxiv.org/pdf/1911.02150
# Primer-EZ DConv: https://arxiv.org/pdf/2109.08668v2
class MultiQueryAttentionDConv(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.d_model % config.n_head == 0
        self.head_dim = config.d_model // config.n_head  # head dimension
        self.n_head = config.n_head
        self.d_model = config.d_model
        self.dropout = config.dropout
        self.window_size = config.window_size if hasattr(config, 'window_size') else (-1, -1)
        # query projections for all heads, key and value projections for one head
        self.qkv = nn.Linear(config.d_model, config.d_model + 2 * self.head_dim, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        # regularization
        self.resid_dropout = nn.Dropout(config.dropout)
        # causal depthwise convolutions for Q, K, and V
        self.q_conv = SpatialDepthWiseConvolution(self.head_dim)
        self.k_conv = SpatialDepthWiseConvolution(self.head_dim)
        self.v_conv = SpatialDepthWiseConvolution(self.head_dim)

    def forward(self, x, mask=None):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (d_model)

        # calculate query for all heads, and key & value for a single head
        qkv = self.qkv(x)
        q = qkv[..., :self.d_model]
        k = qkv[..., self.d_model:self.d_model + C // self.n_head]
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
        self.ln = FastSimpleRMSNorm(args.d_model)
        self.mamba = Mamba2(
                    d_model=args.d_model,
                    d_state=args.d_state,
                    d_conv=args.d_conv,
                    expand=args.expand,
                    headdim=args.headdim)
        self.attn = MultiQueryAttentionDConv(args)
        self.ffn = RWKV_CMix_x060(args, layer_id)

    def forward(self, x):
        x = x + self.mamba(self.ln(x))
        x = x + self.attn(self.ln(x))
        x = x + self.ffn(self.ln(x))
        return x

class EmbeddingLayer(nn.Module):
    def __init__(self, args):
        super(EmbeddingLayer, self).__init__()

        if args.bnb_embedding:
            self.embedding = bnb.nn.StableEmbedding(args.n_vocab, args.d_model)
        else:
            self.embedding = nn.Embedding(args.n_vocab, args.d_model)
        self.lm_head = nn.Linear(args.d_model, args.n_vocab)

        # Tie vocab weights
        self.lm_head.weight = self.embedding.weight

    def forward_encode(self, inputs):
        return self.embedding(inputs)

    def forward_decode(self, x):
        return self.lm_head(x)

# https://github.com/huggingface/transformers/blob/c28d04e9e252a1a099944e325685f14d242ecdcd/src/transformers/models/gpt2/modeling_gpt2.py#L454
def _init_weights(
    module,
    n_layer,
    initializer_range=0.02,  # Now only used for embedding layer.
    rescale_prenorm_residual=True,
    n_residuals_per_layer=3, # Mamba2, MHA, FFN
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding) or isinstance(module, bnb.nn.StableEmbedding):
        nn.init.normal_(module.weight, std=initializer_range)
    elif isinstance(module, MultiQueryAttentionDConv):
        nn.init.orthogonal_(module.qkv.weight)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)

# Predicts next two tokens
class LatentLanguage(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args

        self.embed = EmbeddingLayer(args)

        self.layers = nn.ModuleList()

        for layer_id in range(args.n_layer):
            self.layers.append(Block(args, layer_id))

        self.ln = FastSimpleRMSNorm(args.d_model)
        self.pred1 = RWKV_CMix_x060(args, args.n_layer)
        self.pred2 = RWKV_CMix_x060(args, args.n_layer)

        self.drop = nn.Dropout(args.dropout)

        # Initialize weights
        self.apply(lambda module: _init_weights(module, args.n_layer))

    def override_config(self, mng):
        mng.override_config(self.pred1.time_maa_k, "optim_bits", 32)
        mng.override_config(self.pred1.time_maa_r, "optim_bits", 32)

        mng.override_config(self.pred2.time_maa_k, "optim_bits", 32)
        mng.override_config(self.pred2.time_maa_r, "optim_bits", 32)

        for block in self.layers:
            mng.override_config(block.ffn.time_maa_k, "optim_bits", 32)
            mng.override_config(block.ffn.time_maa_r, "optim_bits", 32)

            mng.override_config(block.mamba.A_log, "optim_bits", 32)
            mng.override_config(block.mamba.D, "optim_bits", 32)

    def forward(self, x, targets_1, targets_2):
        B, N = x.size()

        # True: should take part in attention
        #attn_mask = (x != -1).unsqueeze(1).unsqueeze(2)

        x = x.masked_fill(x == -1, self.args.padding_token)

        x = self.embed.forward_encode(x)

        x = self.drop(x)

        num_layers = len(self.layers)

        # Apply layers in [0, 1, 2, 3, 4, 4, 3, 2, 1, 5] order (for L=6)
        layer_order = (
            list(range(num_layers - 1)) +  # Forward pass excluding last layer
            list(range(num_layers - 2, 0, -1)) +  # Backward pass excluding first and last layer
            [num_layers - 1]  # Last layer
        )

        for i in layer_order:
            x = self.layers[i](x)
            #x = checkpoint(self.layers[i], x, use_reentrant=False)

        # Apply output prediction heads
        x = self.ln(x)
        x1 = self.pred1(x)
        x2 = self.pred2(x)

        # Use the same lm_head for both predictions
        logits_1 = self.embed.forward_decode(x1)
        logits_2 = self.embed.forward_decode(x2)

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
