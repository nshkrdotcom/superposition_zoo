from __future__ import annotations

import pytest
import torch
from mixing_helpers import assert_gradients_flow, assert_shape_preserved, assert_strictly_causal

from superposition_zoo.mixing.standard_attention import StandardAttention


def test_shape_preserved():
    assert_shape_preserved(StandardAttention(d_model=16, n_heads=4), d_model=16)


def test_strictly_causal():
    assert_strictly_causal(StandardAttention(d_model=16, n_heads=4), d_model=16)


def test_gradients_flow():
    assert_gradients_flow(StandardAttention(d_model=16, n_heads=4), d_model=16)


def test_rejects_d_model_not_divisible_by_n_heads():
    with pytest.raises(ValueError):
        StandardAttention(d_model=10, n_heads=3)


def test_param_count_matches_hand_computation():
    d_model = 8
    model = StandardAttention(d_model=d_model, n_heads=2)
    # qkv_proj: (d_model -> 3*d_model) weight + bias; out_proj: (d_model -> d_model) weight + bias
    expected = (d_model * 3 * d_model + 3 * d_model) + (d_model * d_model + d_model)
    assert model.num_parameters() == expected


def test_registered_under_standard_attention_name():
    from superposition_zoo.mixing import REGISTRY

    instance = REGISTRY.build("standard_attention", d_model=16)
    assert isinstance(instance, StandardAttention)


def test_attends_to_earlier_position_not_just_identity():
    # a single content position followed by zeros: attention should let the
    # later position's output depend on the earlier one (unlike a purely
    # local/identity mapping).
    torch.manual_seed(0)
    model = StandardAttention(d_model=8, n_heads=2)
    x = torch.zeros(1, 4, 8)
    x[0, 0] = torch.randn(8)
    with torch.no_grad():
        y = model(x)
    x_perturbed = x.clone()
    x_perturbed[0, 0] += 5.0
    with torch.no_grad():
        y_perturbed = model(x_perturbed)
    assert not torch.allclose(y[0, 1:], y_perturbed[0, 1:])
