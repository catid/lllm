import torch
import torch.nn as nn
import torch.nn.functional as F

from mambaformer import MamaFormer

class LatentLM(nn.Module):
    def __init__(self, full_args, half_args, quarter_args, body_args, body_n):
        super().__init__()
        self.full_layer = MamaFormer(full_args)
        self.half_layer = MamaFormer(half_args)
        self.quarter_layer = MamaFormer(quarter_args)
        self.

    def forward(self, x):
