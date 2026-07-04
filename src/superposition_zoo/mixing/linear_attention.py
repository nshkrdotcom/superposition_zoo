"""Causal linear (kernelized) attention.

Mechanically close to standard attention (same qkv/out projections, so a
matched-parameter comparison against ``StandardAttention`` is exact and
free), but replaces the softmax with a positive feature map and a cumulative
sum, which is known in the associative-recall literature to struggle at
exact recall relative to softmax attention -- a good, well-precedented
minimal-diff contrast case (doc 4 §7).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from superposition_zoo.mixing.base import MixingPrimitive


def causal_linear_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """Causal linear attention core, computed via cumulative sums.

    Args:
        q, k, v: ``(batch, n_heads, seq_len, head_dim)``, already passed
            through a positive feature map (this function does not apply
            one itself, so it can be unit-tested against a hand-computed
            closed form independent of the feature-map choice).
        eps: denominator stabilizer.

    Returns:
        ``(batch, n_heads, seq_len, head_dim)``.
    """
    kv = torch.einsum("bhtd,bhte->bhtde", k, v)
    kv_cumsum = kv.cumsum(dim=2)
    k_cumsum = k.cumsum(dim=2)

    numerator = torch.einsum("bhtd,bhtde->bhte", q, kv_cumsum)
    denominator = torch.einsum("bhtd,bhtd->bht", q, k_cumsum).unsqueeze(-1) + eps
    return numerator / denominator


class LinearAttention(MixingPrimitive):
    def __init__(self, d_model: int, n_heads: int = 4, eps: float = 1e-6, **kwargs):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by n_heads={n_heads}")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.eps = eps
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    @staticmethod
    def _feature_map(x: torch.Tensor) -> torch.Tensor:
        return F.elu(x) + 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(self.d_model, dim=-1)
        q = q.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        q = self._feature_map(q)
        k = self._feature_map(k)

        y = causal_linear_attention(q, k, v, eps=self.eps)
        y = y.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)
        return self.out_proj(y)
