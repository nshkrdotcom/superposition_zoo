from __future__ import annotations

import torch
from mixing_helpers import assert_gradients_flow, assert_shape_preserved, assert_strictly_causal

from superposition_zoo.mixing.delta_net import DeltaNet, delta_rule_recurrence


def test_shape_preserved():
    assert_shape_preserved(DeltaNet(d_model=16, n_heads=4), d_model=16)


def test_strictly_causal():
    assert_strictly_causal(DeltaNet(d_model=16, n_heads=4), d_model=16)


def test_gradients_flow():
    assert_gradients_flow(DeltaNet(d_model=16, n_heads=4), d_model=16)


def test_registered_under_delta_net_name():
    from superposition_zoo.mixing import REGISTRY

    instance = REGISTRY.build("delta_net", d_model=16)
    assert isinstance(instance, DeltaNet)


def test_zero_beta_means_state_never_updates_output_stays_zero():
    # beta=0 at every step: the delta-rule update is a no-op, so the state
    # stays exactly zero and every output must be exactly zero regardless
    # of q/k/v.
    q = torch.randn(2, 3, 5, 4)
    k = torch.randn(2, 3, 5, 4)
    v = torch.randn(2, 3, 5, 4)
    beta = torch.zeros(2, 3, 5)
    out = delta_rule_recurrence(q, k, v, beta)
    assert torch.allclose(out, torch.zeros_like(out))


def test_delta_rule_recurrence_matches_hand_computed_two_steps():
    # B=1, H=1, Dh=2, T=2 -- worked by hand from the recurrence:
    #   predicted_t = state^T k_t  (contracting the key index)
    #   delta_t = v_t - predicted_t
    #   state <- state + beta_t * outer(k_t, delta_t)
    #   out_t = state^T q_t   (read using the just-updated state)
    q = torch.tensor([[[[1.0, 0.0], [0.5, 0.5]]]])
    k = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
    v = torch.tensor([[[[3.0, 4.0], [1.0, 1.0]]]])
    beta = torch.tensor([[[0.5, 1.0]]])

    out = delta_rule_recurrence(q, k, v, beta)

    expected_step0 = torch.tensor([1.5, 2.0])
    expected_step1 = torch.tensor([1.25, 1.5])
    assert torch.allclose(out[0, 0, 0], expected_step0, atol=1e-5)
    assert torch.allclose(out[0, 0, 1], expected_step1, atol=1e-5)


def test_beta_is_squashed_into_unit_interval_by_the_module():
    # the module applies sigmoid to its beta projection; construct a case
    # with an extreme bias and confirm outputs stay finite (no runaway
    # unbounded overwrite from beta outside [0, 1]).
    model = DeltaNet(d_model=8, n_heads=2)
    with torch.no_grad():
        model.beta_proj.bias.fill_(100.0)
    x = torch.randn(2, 6, 8)
    y = model(x)
    assert torch.all(torch.isfinite(y))
