from __future__ import annotations

import torch
import torch.nn.functional as F
from mixing_helpers import assert_gradients_flow, assert_shape_preserved, assert_strictly_causal

from superposition_zoo.mixing.linear_attention import LinearAttention, causal_linear_attention


def test_shape_preserved():
    assert_shape_preserved(LinearAttention(d_model=16, n_heads=4), d_model=16)


def test_strictly_causal():
    assert_strictly_causal(LinearAttention(d_model=16, n_heads=4), d_model=16)


def test_gradients_flow():
    assert_gradients_flow(LinearAttention(d_model=16, n_heads=4), d_model=16)


def test_param_count_matches_standard_attention_exactly():
    # Linear attention reuses the same qkv_proj/out_proj shapes as standard
    # attention -- a matched-budget comparison between the two is free.
    from superposition_zoo.mixing.standard_attention import StandardAttention

    linear = LinearAttention(d_model=32, n_heads=4)
    standard = StandardAttention(d_model=32, n_heads=4)
    assert linear.num_parameters() == standard.num_parameters()


def test_registered_under_linear_attention_name():
    from superposition_zoo.mixing import REGISTRY

    instance = REGISTRY.build("linear_attention", d_model=16)
    assert isinstance(instance, LinearAttention)


def test_causal_linear_attention_core_first_position_equals_v0():
    # At the first position there is nothing else in the causal window, so
    # (for negligible eps relative to the q.k inner product) the output must
    # reduce to v_0 exactly, regardless of q_0/k_0 -- a known-correct-answer
    # unit test for the core math, not just a smoke test.
    torch.manual_seed(0)
    q = F.elu(torch.randn(2, 3, 5, 4)) + 1.0
    k = F.elu(torch.randn(2, 3, 5, 4)) + 1.0
    v = torch.randn(2, 3, 5, 4)
    out = causal_linear_attention(q, k, v, eps=1e-6)
    assert torch.allclose(out[:, :, 0, :], v[:, :, 0, :], atol=1e-4)


def test_causal_linear_attention_core_matches_hand_computed_two_steps():
    # B=1, H=1, T=2, Dh=2. Compute step 1's output by hand from the
    # closed-form causal linear-attention recurrence.
    q = torch.tensor([[[[1.0, 0.0], [0.5, 0.5]]]])
    k = torch.tensor([[[[1.0, 1.0], [2.0, 0.0]]]])
    v = torch.tensor([[[[3.0, 4.0], [1.0, 1.0]]]])
    eps = 1e-6

    out = causal_linear_attention(q, k, v, eps=eps)

    # step 0: S_0 = k_0 outer v_0, Z_0 = k_0
    s0 = torch.outer(k[0, 0, 0], v[0, 0, 0])
    z0 = k[0, 0, 0]
    expected_0 = (q[0, 0, 0] @ s0) / (q[0, 0, 0] @ z0 + eps)
    assert torch.allclose(out[0, 0, 0], expected_0, atol=1e-5)

    # step 1: S_1 = S_0 + k_1 outer v_1, Z_1 = Z_0 + k_1
    s1 = s0 + torch.outer(k[0, 0, 1], v[0, 0, 1])
    z1 = z0 + k[0, 0, 1]
    expected_1 = (q[0, 0, 1] @ s1) / (q[0, 0, 1] @ z1 + eps)
    assert torch.allclose(out[0, 0, 1], expected_1, atol=1e-5)
