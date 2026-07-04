"""A minimal selective linear recurrence (S4/Mamba-lite).

Mechanically about as different from attention as a "sequence mixing
primitive" can be while staying in that category: a per-channel linear
recurrent dynamical system (state evolves by a learned, input-dependent
gate) instead of pairwise content comparison across all positions. This is
one of the most actively debated open questions in interpretability right
now -- whether SSMs represent and route information fundamentally
differently from attention -- so a result here has standing interest beyond
this repo's own curiosity (doc 4 §7).

This is deliberately the simplest correct selective SSM, not a full
S4/Mamba reproduction: a single scalar state per channel (no expanded state
dimension), a GRU-style convex retain/write gate, and a straightforward
sequential recurrence rather than a parallel scan. Both simplifications are
honest scope choices for a first version, not hidden limitations.
"""

from __future__ import annotations

import torch
from torch import nn

from superposition_zoo.mixing.base import MixingPrimitive


def selective_scan(a: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """The per-channel recurrence ``h_t = a_t * h_{t-1} + c_t`` (``h_{-1} = 0``).

    Args:
        a: ``(batch, seq_len, d_model)`` retain gate, expected in ``[0, 1]``
            for a stable/bounded recurrence (not enforced here so this
            function can be unit-tested with arbitrary values).
        c: ``(batch, seq_len, d_model)`` per-step input contribution.

    Returns:
        ``(batch, seq_len, d_model)`` state trajectory.
    """
    batch, seq_len, d_model = a.shape
    h = torch.zeros(batch, d_model, device=a.device, dtype=a.dtype)
    outputs = []
    for t in range(seq_len):
        h = a[:, t] * h + c[:, t]
        outputs.append(h)
    return torch.stack(outputs, dim=1)


class MinimalSSM(MixingPrimitive):
    def __init__(self, d_model: int, **kwargs):
        super().__init__()
        self.d_model = d_model
        self.a_proj = nn.Linear(d_model, d_model)
        self.b_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = torch.sigmoid(self.a_proj(x))
        b = self.b_proj(x)
        c = (1.0 - a) * b
        h = selective_scan(a, c)
        return self.out_proj(h)
