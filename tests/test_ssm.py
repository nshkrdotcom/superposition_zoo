from __future__ import annotations

import torch
from mixing_helpers import assert_gradients_flow, assert_shape_preserved, assert_strictly_causal

from superposition_zoo.mixing.ssm import MinimalSSM, selective_scan


def test_shape_preserved():
    assert_shape_preserved(MinimalSSM(d_model=16), d_model=16)


def test_strictly_causal():
    assert_strictly_causal(MinimalSSM(d_model=16), d_model=16)


def test_gradients_flow():
    assert_gradients_flow(MinimalSSM(d_model=16), d_model=16)


def test_registered_under_ssm_name():
    from superposition_zoo.mixing import REGISTRY

    instance = REGISTRY.build("ssm", d_model=16)
    assert isinstance(instance, MinimalSSM)


def test_selective_scan_matches_hand_computed_two_steps():
    a = torch.tensor([[[0.5], [0.25]]])  # (batch=1, seq_len=2, d_model=1)
    c = torch.tensor([[[2.0], [3.0]]])
    h = selective_scan(a, c)
    # h_0 = a_0 * 0 + c_0 = 2.0
    # h_1 = a_1 * h_0 + c_1 = 0.25 * 2.0 + 3.0 = 3.5
    assert torch.allclose(h[0, 0], torch.tensor([2.0]))
    assert torch.allclose(h[0, 1], torch.tensor([3.5]))


def test_selective_scan_zero_retention_equals_current_input_exactly():
    # a_t == 0 everywhere means no memory at all: h_t == c_t exactly.
    a = torch.zeros(2, 5, 4)
    c = torch.randn(2, 5, 4)
    h = selective_scan(a, c)
    assert torch.allclose(h, c)


def test_selective_scan_full_retention_with_zero_input_stays_at_initial_state():
    # a_t == 1 and c_t == 0 everywhere: state never changes from its zero init.
    a = torch.ones(2, 5, 4)
    c = torch.zeros(2, 5, 4)
    h = selective_scan(a, c)
    assert torch.allclose(h, torch.zeros_like(h))
