"""Shared test contract for every mixing primitive.

Not a test module itself (no ``test_`` prefix, pytest will not collect it) --
imported by each primitive-specific test file so every primitive is checked
against the same shape/causality contract.
"""

from __future__ import annotations

import torch
from torch import nn


def assert_shape_preserved(module: nn.Module, d_model: int, batch: int = 2, seq_len: int = 16) -> None:
    x = torch.randn(batch, seq_len, d_model)
    y = module(x)
    assert y.shape == (batch, seq_len, d_model)


def assert_strictly_causal(
    module: nn.Module, d_model: int, batch: int = 2, seq_len: int = 16, atol: float = 1e-5
) -> None:
    """A change to the last position must never change any earlier position's output."""
    torch.manual_seed(0)
    x = torch.randn(batch, seq_len, d_model)
    with torch.no_grad():
        y1 = module(x)

        x2 = x.clone()
        x2[:, -1, :] += 10.0
        y2 = module(x2)

    assert torch.allclose(y1[:, :-1, :], y2[:, :-1, :], atol=atol), (
        "output at earlier positions changed when a later position was perturbed -- "
        "primitive is not strictly causal"
    )


def assert_gradients_flow(module: nn.Module, d_model: int, batch: int = 2, seq_len: int = 8) -> None:
    x = torch.randn(batch, seq_len, d_model, requires_grad=True)
    y = module(x)
    y.pow(2).sum().backward()
    assert x.grad is not None
    assert torch.any(x.grad != 0.0)
    for name, param in module.named_parameters():
        assert param.grad is not None, f"no gradient reached parameter {name}"
