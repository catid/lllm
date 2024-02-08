import torch
import torch.nn as nn
import torch.nn.functional as F

from types import SimpleNamespace

from mambaformer import MamaFormer

class LatentLM(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.embed = nn.Embedding(args.vocab_size, args.full_dim)

        full_args = SimpleNamespace()
        full_args.dim = args.full_dim
        full_args.mamba_state = args.full_mamba_state
        full_args.mamba_conv = args.full_mamba_conv
        full_args.mamba_expand = args.full_mamba_expand
        full_args.heads = args.full_heads
        full_args.dropout = args.dropout
        full_args.ff_mult = args.ff_mult
        full_args.ff_bias = args.ff_bias

        self.full_layer = MamaFormer(full_args)
        self.ds0_proj = nn.Linear(full_args.dim * 2, full_args.dim)

        self.half_layer = MamaFormer(full_args)
        self.ds1_proj = nn.Linear(full_args.dim * 2, full_args.dim)

        self.quarter_layer = MamaFormer(full_args)
        self.ds2_proj = nn.Linear(full_args.dim * 2, full_args.dim)

    def forward(self, idx):
        x = self.embed(idx)

        # FIXME: transformer.drop()?

