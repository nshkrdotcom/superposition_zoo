"""DeltaNet: delta-rule associative memory (Yang et al., 2024).

Structurally the "minimal diff from linear attention": same qkv/read-out
shape, but the memory update is a corrective delta-rule write
(``state += beta * k (v - state^T k)``) instead of linear attention's pure
additive accumulation (``state += k v^T``). This gives real overwrite/erase
dynamics -- writing to a key direction that already holds a value corrects
it, rather than just adding more mass to it -- which is exactly the
structural distinction doc 4 §7 wants tested against plain linear attention.

Implemented as an explicit sequential recurrence over time. The delta rule
is not a simple cumulative sum (each step's state update nonlinearly
depends on the *current* state, unlike linear attention's additive
accumulation), so it cannot be vectorized into a cumsum the way
``linear_attention.py`` is. At the toy sequence lengths this repo targets
(tens to low hundreds of positions), a plain Python loop is fast enough and,
critically, obviously correct -- a fully chunked/parallel-across-time
implementation (as in the original DeltaNet paper) is a legitimate future
optimization, not a requirement for a first correct version.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from superposition_zoo.mixing.base import MixingPrimitive


def delta_rule_recurrence(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, beta: torch.Tensor
) -> torch.Tensor:
    """The delta-rule associative-memory recurrence, computed sequentially.

    Args:
        q, k, v: ``(batch, n_heads, seq_len, head_dim)``. This function does
            not normalize ``k`` itself (that is a module-level choice, see
            :class:`DeltaNet`), so it can be unit-tested against a
            hand-computed closed form independent of that choice.
        beta: ``(batch, n_heads, seq_len)`` write-strength in ``[0, 1]``.

    Returns:
        ``(batch, n_heads, seq_len, head_dim)``.
    """
    batch, n_heads, seq_len, head_dim = q.shape
    state = torch.zeros(batch, n_heads, head_dim, head_dim, device=q.device, dtype=q.dtype)
    outputs = []

    for t in range(seq_len):
        k_t = k[:, :, t]
        v_t = v[:, :, t]
        q_t = q[:, :, t]
        beta_t = beta[:, :, t]

        predicted = torch.einsum("bhij,bhi->bhj", state, k_t)
        delta = v_t - predicted
        update = beta_t.unsqueeze(-1).unsqueeze(-1) * torch.einsum("bhi,bhj->bhij", k_t, delta)
        state = state + update

        out_t = torch.einsum("bhij,bhi->bhj", state, q_t)
        outputs.append(out_t)

    return torch.stack(outputs, dim=2)


class DeltaNet(MixingPrimitive):
    def __init__(self, d_model: int, n_heads: int = 4, eps: float = 1e-6, **kwargs):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by n_heads={n_heads}")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.eps = eps
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.beta_proj = nn.Linear(d_model, n_heads)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(self.d_model, dim=-1)
        q = q.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        # unit-norm keys is standard delta-rule practice: it keeps the
        # (I - beta k k^T) contraction well-behaved instead of blowing up.
        k = F.normalize(k, dim=-1, eps=self.eps)

        beta = torch.sigmoid(self.beta_proj(x)).transpose(1, 2)  # (batch, n_heads, seq_len)

        y = delta_rule_recurrence(q, k, v, beta)
        y = y.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)
        return self.out_proj(y)
