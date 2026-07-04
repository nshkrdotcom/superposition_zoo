from __future__ import annotations

import torch
from mixing_helpers import assert_gradients_flow, assert_shape_preserved, assert_strictly_causal

from superposition_zoo.mixing.hard_routing import HardRoutingAttention, causal_hard_routing_weights


def test_shape_preserved():
    assert_shape_preserved(HardRoutingAttention(d_model=16, n_heads=4), d_model=16)


def test_strictly_causal_in_eval_mode():
    # eval mode uses deterministic hard argmax (no gumbel-softmax noise), so
    # the strict-causality invariant holds exactly, as it must for any
    # legitimate mixing primitive (doc 5 §7). Training-mode stochastic
    # relaxation is a gradient-estimation device, not part of the model's
    # actual causal computation contract, and is tested separately below.
    model = HardRoutingAttention(d_model=16, n_heads=4)
    model.eval()
    assert_strictly_causal(model, d_model=16)


def test_gradients_flow_in_training_mode_via_straight_through():
    model = HardRoutingAttention(d_model=16, n_heads=4)
    model.train()
    assert_gradients_flow(model, d_model=16)


def test_registered_under_hard_routing_name():
    from superposition_zoo.mixing import REGISTRY

    instance = REGISTRY.build("hard_routing", d_model=16)
    assert isinstance(instance, HardRoutingAttention)


def test_param_count_matches_standard_attention_exactly():
    from superposition_zoo.mixing.standard_attention import StandardAttention

    hard = HardRoutingAttention(d_model=32, n_heads=4)
    standard = StandardAttention(d_model=32, n_heads=4)
    assert hard.num_parameters() == standard.num_parameters()


def test_causal_hard_routing_weights_are_one_hot_per_row_in_eval_mode():
    torch.manual_seed(0)
    scores = torch.randn(2, 3, 5, 5)  # (batch, heads, query, key)
    weights = causal_hard_routing_weights(scores, training=False, temperature=1.0)
    row_sums = weights.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums))
    nonzero_per_row = (weights > 0).sum(dim=-1)
    assert torch.all(nonzero_per_row == 1)


def test_causal_hard_routing_weights_respect_causal_mask():
    torch.manual_seed(0)
    scores = torch.randn(1, 1, 6, 6)
    weights = causal_hard_routing_weights(scores, training=False, temperature=1.0)
    for i in range(6):
        chosen = weights[0, 0, i].nonzero().item()
        assert chosen <= i


def test_causal_hard_routing_weights_pick_the_argmax_when_deterministic():
    scores = torch.tensor([[[[1.0, -1e9, -1e9], [0.2, 5.0, -1e9], [0.1, 0.1, 9.0]]]])
    weights = causal_hard_routing_weights(scores, training=False, temperature=1.0)
    assert weights[0, 0, 0].argmax().item() == 0
    assert weights[0, 0, 1].argmax().item() == 1
    assert weights[0, 0, 2].argmax().item() == 2
