"""Phase 1 model: embed -> N x mixing block -> decode.

Every mixing primitive in the zoo plugs into the same shell, so swapping
``mixing_primitive_name`` is the only thing that changes between a run using
standard attention and a run using DeltaNet or any other registered
primitive -- everything else (embedding width, MLP ratio, depth, decode
head) stays identical, which is what makes a matched-budget comparison
meaningful (doc 5 §2 principle 2).

The recall benchmark (``recall_task.py``) is now a *content-addressed*
pointer/copy task (MQAR-style): a pointer's key matches its true source
position's key exactly, and retrieval means finding the earlier position
with a matching key. An earlier version encoded the source as a relative
positional offset instead, which requires the model to perform positional
arithmetic rather than content matching -- that turned out to be much
harder to learn in practice (recall accuracy stayed near zero even after
adding positional encoding, verified by actually training the system, not
by any unit test), so the benchmark itself was redesigned rather than the
model tuned harder to compensate. See ``recall_task.py`` for the full
account.

Positional encoding is still added at the embedding step below, since even
content-addressed matching benefits from the model knowing "earlier vs.
later" beyond what the causal mask alone conveys, and some mixing
primitives (e.g. the SSM) have no other route to positional information at
all.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from superposition_zoo.mixing import REGISTRY
from superposition_zoo.recall_task import CONTROL_CHANNELS as _CONTROL_CHANNELS


def sinusoidal_positional_encoding(
    seq_len: int, d_model: int, device: torch.device | None = None
) -> torch.Tensor:
    """Fixed (non-learned) sinusoidal positional encoding, as in Vaswani et al. (2017).

    Parameter-free by design: a learned positional embedding table would add
    parameters inconsistently across configs with different ``seq_len``,
    complicating the matched-budget comparisons this repo cares about.

    Args:
        seq_len: sequence length.
        d_model: embedding dimension; must be even.

    Returns:
        ``(seq_len, d_model)``, values in ``[-1, 1]``.
    """
    if d_model % 2 != 0:
        raise ValueError(f"d_model must be even, got {d_model}")
    position = torch.arange(seq_len, device=device, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2, device=device, dtype=torch.float32) * (-math.log(10000.0) / d_model)
    )
    pe = torch.zeros(seq_len, d_model, device=device)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


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
        seq_len = x.shape[1]
        x = x + sinusoidal_positional_encoding(seq_len, self.d_model, device=x.device).unsqueeze(0)
        for block in self.blocks:
            x = block(x)
        return self.decode(x)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
