import torch
from torch import nn

import random

from model.mamba import MambaBlock
from model.ngram import Ngram

class LatentLanguage(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.embedding = nn.Embedding(args.vocab_size, args.dim)

        # 1:1 with input tokens
        self.full_start = nn.Sequential(
            self.embedding,
            MambaBlock(0, args.dim, args.state, args.conv, args.expand),
            Ngram(args.dim, ngram=1),
            Ngram(args.dim, ngram=2),
            Ngram(args.dim, ngram=3),
            MambaBlock(1, args.dim, args.state, args.conv, args.expand),
        )

        self.down1 = nn.Linear(args.dim * 2, args.dim)
        self.half_start = MambaBlock(2, args.dim, args.state, args.conv, args.expand)
        self.down2 = nn.Linear(args.dim * 2, args.dim)

        self.quarter_start = nn.Sequential(
            MambaBlock(3, args.dim, args.state, args.conv, args.expand),
            nn.Linear(args.dim, args.dim // 2),
            MambaBlock(4, args.dim // 2, args.state, args.conv, args.expand),
        )

        layer_index = 5
        self.body_layers = []

        for _ in range(args.layers):
            # Add pairs of layers to self.body_layers
            self.body_layers.append(
                nn.Sequential(
                    MambaBlock(layer_index, args.dim // 2, args.state, args.conv, args.expand),
                    MambaBlock(layer_index + 1, args.dim // 2, args.state, args.conv, args.expand),
                )
            )
            layer_index += 2

        self.body_end = MambaBlock(layer_index, args.dim // 2, args.state, args.conv, args.expand)
        layer_index += 1

        self.up1 = nn.Linear(args.dim // 2, args.dim)

        self.quarter_end = MambaBlock(layer_index, args.dim, args.state, args.conv, args.expand)
        layer_index += 1

        self.up2 = nn.Linear(args.dim, args.dim * 2)

        self.half_end = MambaBlock(layer_index, args.dim, args.state, args.conv, args.expand)
        layer_index += 1

        self.up3 = nn.Linear(args.dim, args.dim * 2)

        self.full_end = MambaBlock(layer_index, args.dim, args.state, args.conv, args.expand)
        layer_index += 1

        self.lm_head = nn.Linear(args.dim, args.vocab_size)

        # Tie vocab weights
        self.lm_head.weight = self.embedding.weight

    def forward(self, x):
        x = self.full_start(x)

        random.shuffle(self.body_layers)

        for layer in self.body_layers:
            x = layer(x)

        return self.lm_head(x)
