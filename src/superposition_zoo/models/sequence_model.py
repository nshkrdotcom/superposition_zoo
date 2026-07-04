"""Phase 1 model: embed -> N x mixing block -> decode.

Every mixing primitive in the zoo plugs into the same shell, so swapping
``mixing_primitive_name`` is the only thing that changes between a run using
standard attention and a run using DeltaNet or any other registered
primitive -- everything else (embedding width, MLP ratio, depth, decode
head) stays identical, which is what makes a matched-budget comparison
meaningful (doc 5 §2 principle 2).
"""

from __future__ import annotations

import torch
from torch import nn

from superposition_zoo.mixing import REGISTRY

_CONTROL_CHANNELS = 2  # [is_pointer_flag, normalized_offset], see recall_task.py


class MixingBlock(nn.Module):
    """Pre-norm residual block: mixing primitive + a small MLP."""

    def __init__(self, d_model: int, mixing_primitive_name: str, mixing_kwargs: dict | None = None):
        super().__init__()
        mixing_kwargs = mixing_kwargs or {}
        self.norm1 = nn.LayerNorm(d_model)
        self.mixing = REGISTRY.build(mixing_primitive_name, d_model=d_model, **mixing_kwargs)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.mixing(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class SequenceModel(nn.Module):
    def __init__(
        self,
        n_features: int,
        d_model: int,
        mixing_primitive_name: str,
        n_layers: int = 1,
        mixing_kwargs: dict | None = None,
    ):
        super().__init__()
        self.n_features = n_features
        self.d_model = d_model
        self.embed = nn.Linear(n_features + _CONTROL_CHANNELS, d_model)
        self.blocks = nn.ModuleList(
            [
                MixingBlock(d_model, mixing_primitive_name, mixing_kwargs)
                for _ in range(n_layers)
            ]
        )
        self.decode = nn.Linear(d_model, n_features)

    def forward(self, input_features: torch.Tensor, control: torch.Tensor) -> torch.Tensor:
        x = torch.cat([input_features, control], dim=-1)
        x = self.embed(x)
        for block in self.blocks:
            x = block(x)
        return self.decode(x)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
