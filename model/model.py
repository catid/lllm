import torch
from torch import nn

import random

from model.mamba import MambaBlock
from model.ngram import Ngram

from dataclasses import dataclass

@dataclass
class LatentLanguageConfig:
    dim: int = 256
    state: 16
    conv: 2
    expand: 4
    vocab_size: -1

class TokenMerger(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.down = nn.Linear(dim * 2, dim)

    def forward(self, x):
        batch_size, num_tokens, num_features = x.shape
        assert num_tokens % 2 == 0, "num_tokens must be even"

        x = x.view(batch_size, num_tokens // 2, num_features * 2)
        x = self.down(x)

        return x

class TokenExpander(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.up = nn.Linear(dim, dim * 2)

    def forward(self, x):
        batch_size, num_tokens, num_features = x.shape

        x = self.up(x)
        x = x.view(batch_size, num_tokens * 2, num_features // 2)  

        return x

class LatentLanguage(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.embedding = nn.Embedding(cfg.vocab_size, cfg.dim)

        # 1:1 with input tokens
        self.encoder = nn.Sequential(
            self.embedding,
            MambaBlock(0, cfg.dim, cfg.state, cfg.conv, cfg.expand),
            Ngram(cfg.dim, ngram=1),
            Ngram(cfg.dim, ngram=2),
            Ngram(cfg.dim, ngram=3),
            MambaBlock(1, cfg.dim, cfg.state, cfg.conv, cfg.expand),
            TokenMerger(cfg.dim),
            MambaBlock(2, cfg.dim, cfg.state, cfg.conv, cfg.expand),
            TokenMerger(cfg.dim),
            MambaBlock(3, cfg.dim, cfg.state, cfg.conv, cfg.expand),
            nn.Linear(cfg.dim, cfg.dim // 2),
            MambaBlock(4, cfg.dim // 2, cfg.state, cfg.conv, cfg.expand),
        )

        layer_index = 5
        self.body_layers = []

        for _ in range(cfg.layers):
            # Add pairs of layers to self.body_layers
            self.body_layers.append(
                nn.Sequential(
                    MambaBlock(layer_index, cfg.dim // 2, cfg.state, cfg.conv, cfg.expand),
                    MambaBlock(layer_index + 1, cfg.dim // 2, cfg.state, cfg.conv, cfg.expand),
                )
            )
            layer_index += 2

        self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size)

        self.decoder = nn.Sequential(
            MambaBlock(layer_index, cfg.dim // 2, cfg.state, cfg.conv, cfg.expand),
            nn.Linear(cfg.dim // 2, cfg.dim),
            MambaBlock(layer_index + 1, cfg.dim, cfg.state, cfg.conv, cfg.expand),
            TokenExpander(cfg.dim),
            MambaBlock(layer_index + 2, cfg.dim, cfg.state, cfg.conv, cfg.expand),
            TokenExpander(cfg.dim),
            MambaBlock(layer_index + 3, cfg.dim, cfg.state, cfg.conv, cfg.expand),
            self.lm_head,
        )

        # Tie vocab weights
        self.lm_head.weight = self.embedding.weight

    def forward(self, x):
        x = self.encoder(x)

        random.shuffle(self.body_layers)

        for layer in self.body_layers:
            x = layer(x)

        return self.decoder(x)
