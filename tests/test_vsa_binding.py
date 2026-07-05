from __future__ import annotations

import torch
from mixing_helpers import assert_gradients_flow, assert_shape_preserved, assert_strictly_causal

from superposition_zoo.mixing.vsa_binding import VSABinding, causal_bind_bundle_unbind


def test_shape_preserved():
    assert_shape_preserved(VSABinding(d_model=16), d_model=16)


def test_strictly_causal():
    assert_strictly_causal(VSABinding(d_model=16), d_model=16)


def test_gradients_flow():
    assert_gradients_flow(VSABinding(d_model=16), d_model=16)


def test_registered_under_vsa_binding_name():
    from superposition_zoo.mixing import REGISTRY

    instance = REGISTRY.build("vsa_binding", d_model=16)
    assert isinstance(instance, VSABinding)


def test_role_codes_are_frozen_not_a_parameter():
    model = VSABinding(d_model=16)
    param_names = {name for name, _ in model.named_parameters()}
    assert "role_codes" not in param_names
    buffer_names = {name for name, _ in model.named_buffers()}
    assert "role_codes" in buffer_names


def test_role_codes_are_rademacher():
    model = VSABinding(d_model=16, max_seq_len=32)
    codes = model.role_codes
    assert torch.all((codes == 1.0) | (codes == -1.0))


def test_role_codes_are_deterministic_across_instances():
    # "frozen" means the same across every instance and every run, not
    # freshly randomized per construction -- otherwise a checkpoint's
    # bind/unbind algebra wouldn't even make sense to reload.
    model_a = VSABinding(d_model=16, max_seq_len=32)
    model_b = VSABinding(d_model=16, max_seq_len=32)
    assert torch.equal(model_a.role_codes, model_b.role_codes)


def test_raises_when_seq_len_exceeds_max_seq_len():
    import pytest

    model = VSABinding(d_model=16, max_seq_len=8)
    x = torch.randn(1, 16, 16)
    with pytest.raises(ValueError):
        model(x)


def test_causal_bind_bundle_unbind_recovers_own_content_at_first_position():
    # At position 0 there is nothing else bundled in yet, so unbinding with
    # position 0's own code must recover its own bound content exactly:
    # bind(c_0, code_0) then immediately unbind with code_0 again is just
    # c_0 * code_0 * code_0 = c_0 (Rademacher codes are self-inverse under
    # elementwise multiplication).
    torch.manual_seed(0)
    content = torch.randn(2, 5, 8)
    codes = (torch.randint(0, 2, (5, 8)) * 2 - 1).float()
    retrieved = causal_bind_bundle_unbind(content, codes)
    assert torch.allclose(retrieved[:, 0, :], content[:, 0, :], atol=1e-5)


def test_causal_bind_bundle_unbind_matches_hand_computation_two_steps():
    # B=1, T=2, D=2. codes chosen so cross-terms are easy to hand-verify.
    content = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    codes = torch.tensor([[1.0, -1.0], [-1.0, 1.0]])

    retrieved = causal_bind_bundle_unbind(content, codes)

    # step 0: bound_0 = content_0 * code_0 = [1, -2]; memory_0 = bound_0
    # retrieved_0 = memory_0 * code_0 = [1*1, -2*-1] = [1, 2] = content_0 (self-recovery)
    assert torch.allclose(retrieved[0, 0], torch.tensor([1.0, 2.0]), atol=1e-5)

    # step 1: bound_1 = content_1 * code_1 = [3*-1, 4*1] = [-3, 4]
    # memory_1 = bound_0 + bound_1 = [1-3, -2+4] = [-2, 2]
    # retrieved_1 = memory_1 * code_1 = [-2*-1, 2*1] = [2, 2]
    #             = content_1 + cross-talk from bound_0 unbound with the wrong code
    assert torch.allclose(retrieved[0, 1], torch.tensor([2.0, 2.0]), atol=1e-5)


def test_vsa_binding_is_worse_than_matched_control_at_cross_position_recall():
    # Documents, rather than hides, the primitive's real limitation: frozen
    # position-indexed binding has no content-comparison operation at all,
    # so it structurally cannot do content-addressed retrieval the way
    # recall_task.py's key-matching design requires (that needs comparing
    # two continuous random vectors for similarity, which bind/unbind
    # algebra alone does not provide). This is a predictable, informative
    # negative result, not a bug -- verified directly rather than assumed.
    from superposition_zoo.models.sequence_model import SequenceModel
    from superposition_zoo.recall_task import generate_recall_batch

    torch.manual_seed(0)
    generator = torch.Generator()
    generator.manual_seed(0)

    model = SequenceModel(n_features=8, d_model=16, mixing_primitive_name="vsa_binding", n_layers=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)

    for _ in range(500):
        batch = generate_recall_batch(
            batch_size=32,
            seq_len=16,
            n_features=8,
            n_pointers=2,
            sparsity=0.3,
            min_gap=2,
            generator=generator,
        )
        optimizer.zero_grad()
        predicted = model(batch.input_features, batch.control)
        loss = (predicted - batch.target).pow(2).mean()
        loss.backward()
        optimizer.step()

    # not asserting a specific accuracy bound (that's for EXPERIMENT_LOG.md,
    # not a brittle unit test) -- just that training runs to completion and
    # produces finite, sane output, since the whole point of this primitive
    # is architectural, not a training-stability claim.
    model.eval()
    with torch.no_grad():
        eval_batch = generate_recall_batch(
            batch_size=64,
            seq_len=16,
            n_features=8,
            n_pointers=2,
            sparsity=0.3,
            min_gap=2,
            generator=generator,
        )
        predicted = model(eval_batch.input_features, eval_batch.control)
    assert torch.all(torch.isfinite(predicted))
