import torch
from torch import nn

import random

from model.mamba import MambaBlock
from model.ngram import Ngram

class TokenMerger(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.down = nn.Linear(dim * 2, dim)

    def forward(self, x):
        batch_size, num_tokens, num_features = tensor.shape
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
    def __init__(self, args):
        super().__init__()

        self.embedding = nn.Embedding(args.vocab_size, args.dim)

        # 1:1 with input tokens
        self.encoder = nn.Sequential(
            self.embedding,
            MambaBlock(0, args.dim, args.state, args.conv, args.expand),
            Ngram(args.dim, ngram=1),
            Ngram(args.dim, ngram=2),
            Ngram(args.dim, ngram=3),
            MambaBlock(1, args.dim, args.state, args.conv, args.expand),
            TokenMerger(args.dim),
            MambaBlock(2, args.dim, args.state, args.conv, args.expand),
            TokenMerger(args.dim),
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

        self.lm_head = nn.Linear(args.dim, args.vocab_size)

        self.decoder = nn.Sequential(
            MambaBlock(layer_index, args.dim // 2, args.state, args.conv, args.expand),
            nn.Linear(args.dim // 2, args.dim),
            MambaBlock(layer_index + 1, args.dim, args.state, args.conv, args.expand),
            TokenExpander(args.dim),
            MambaBlock(layer_index + 2, args.dim, args.state, args.conv, args.expand),
            TokenExpander(args.dim),
            MambaBlock(layer_index + 3, args.dim, args.state, args.conv, args.expand),
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
