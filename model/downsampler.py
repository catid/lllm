import torch
import torch.nn as nn
import torch.nn.functional as F

# Embedding
# Lightning Transformer
# Project pairs of output to 1 token (2x downsample)
# Lightning Transformer
# Project pairs of output to 1 token (2x downsample)
# Lightning Transformer
# Project output to 2x fewer features

class Downsampler(nn.Module):
    def __init__(self, dim):
        super().__init__()

    def forward(self, x):
