from __future__ import annotations

import pytest
import torch

from superposition_zoo.models.sequence_model import sinusoidal_positional_encoding


def test_shape():
    pe = sinusoidal_positional_encoding(seq_len=10, d_model=16)
    assert pe.shape == (10, 16)


def test_different_positions_are_distinguishable():
    pe = sinusoidal_positional_encoding(seq_len=8, d_model=16)
    for i in range(8):
        for j in range(8):
            if i != j:
                assert not torch.allclose(pe[i], pe[j])


def test_values_are_bounded():
    pe = sinusoidal_positional_encoding(seq_len=20, d_model=32)
    assert torch.all(pe >= -1.0)
    assert torch.all(pe <= 1.0)


def test_deterministic():
    pe1 = sinusoidal_positional_encoding(seq_len=10, d_model=16)
    pe2 = sinusoidal_positional_encoding(seq_len=10, d_model=16)
    assert torch.equal(pe1, pe2)


def test_rejects_odd_d_model():
    with pytest.raises(ValueError):
        sinusoidal_positional_encoding(seq_len=10, d_model=15)
