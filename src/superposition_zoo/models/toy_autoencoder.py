"""Phase 0 model: the original Elhage et al. (2022) toy superposition setup.

No sequence structure at all -- a single linear encoder into a bottleneck,
ReLU, and a linear decoder back out. This is the foundation everything else
in the repo builds on: reproduce this exactly before adding any sequence
mixing.
"""

from __future__ import annotations

import torch
from torch import nn


class ToyAutoencoder(nn.Module):
    """Encoder -> ReLU bottleneck -> decoder, reconstructing sparse features."""

    def __init__(self, n_features: int, d_hidden: int, bias: bool = True):
        super().__init__()
        self.n_features = n_features
        self.d_hidden = d_hidden
        self.encoder = nn.Linear(n_features, d_hidden, bias=bias)
        self.decoder = nn.Linear(d_hidden, n_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = torch.relu(self.encoder(x))
        return self.decoder(hidden)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
