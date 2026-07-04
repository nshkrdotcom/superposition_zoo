"""Standard causal softmax self-attention -- the zoo's reference primitive.

This doubles as the positive control for the recall dimension (doc 4 §7/§8):
it is already known, from the associative-recall literature (Zoology/MQAR),
to be strong at exact recall, so any other primitive's recall failure can be
compared against a case known to work.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from superposition_zoo.mixing.base import MixingPrimitive


class StandardAttention(MixingPrimitive):
    def __init__(self, d_model: int, n_heads: int = 4, **kwargs):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by n_heads={n_heads}")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(self.d_model, dim=-1)
        q = q.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)
        return self.out_proj(y)
