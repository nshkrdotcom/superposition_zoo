"""Hard (discrete, top-1) causal routing attention.

Tests the "legible by construction" hypothesis from doc 4 §7/§11 directly:
instead of a continuous softmax blend over all earlier positions, force each
query to route from exactly one earlier position. There is no blend to be
ambiguous about, so if discreteness alone improves feature isolation under
superposition, this primitive should show it relative to standard attention
at a matched parameter budget (same qkv/out projection shapes -- zero extra
parameters).

Training uses the straight-through Gumbel-softmax trick (forward pass is a
true one-hot selection, backward pass uses the soft relaxation's gradient)
so the discrete choice remains learnable. Evaluation uses a deterministic
hard argmax with no stochastic relaxation -- the strict-causality contract
(doc 5 §7) is defined and tested against this deterministic regime, since
the training-time relaxation is a gradient-estimation device, not part of
the model's actual causal computation.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from superposition_zoo.mixing.base import MixingPrimitive


def causal_hard_routing_weights(
    scores: torch.Tensor, training: bool, temperature: float = 1.0
) -> torch.Tensor:
    """Turn raw attention scores into hard, causal, one-hot routing weights.

    Args:
        scores: ``(..., query_len, key_len)`` with ``query_len == key_len``.
            Causal masking is applied internally -- callers should pass raw,
            unmasked scores.
        training: if ``True``, use the straight-through Gumbel-softmax
            relaxation (stochastic, differentiable); if ``False``, use a
            deterministic hard argmax (no randomness).
        temperature: Gumbel-softmax temperature (only used when training).

    Returns:
        A tensor the same shape as ``scores`` where each query row is a
        one-hot vector over causally-valid key positions.
    """
    seq_len = scores.shape[-1]
    causal_mask = torch.triu(
        torch.ones(seq_len, seq_len, dtype=torch.bool, device=scores.device), diagonal=1
    )
    masked_scores = scores.masked_fill(causal_mask, float("-inf"))

    if training:
        return F.gumbel_softmax(masked_scores, tau=temperature, hard=True, dim=-1)

    idx = masked_scores.argmax(dim=-1, keepdim=True)
    weights = torch.zeros_like(masked_scores)
    weights.scatter_(-1, idx, 1.0)
    return weights


class HardRoutingAttention(MixingPrimitive):
    def __init__(self, d_model: int, n_heads: int = 4, temperature: float = 1.0, **kwargs):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by n_heads={n_heads}")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.temperature = temperature
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(self.d_model, dim=-1)
        q = q.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        scores = torch.einsum("bhid,bhjd->bhij", q, k) / (self.head_dim**0.5)
        weights = causal_hard_routing_weights(scores, training=self.training, temperature=self.temperature)
        y = torch.einsum("bhij,bhjd->bhid", weights, v)

        y = y.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)
        return self.out_proj(y)
