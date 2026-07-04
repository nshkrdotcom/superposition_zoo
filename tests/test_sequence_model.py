from __future__ import annotations

import pytest
import torch

from superposition_zoo.metrics.recall import retrieval_accuracy
from superposition_zoo.models.sequence_model import SequenceModel
from superposition_zoo.recall_task import CONTROL_CHANNELS, generate_recall_batch


@pytest.mark.parametrize("primitive_name", ["standard_attention", "linear_attention", "delta_net"])
def test_shape_preserved_across_primitives(primitive_name):
    model = SequenceModel(n_features=10, d_model=16, mixing_primitive_name=primitive_name, n_layers=1)
    input_features = torch.randn(3, 12, 10)
    control = torch.zeros(3, 12, CONTROL_CHANNELS)
    output = model(input_features, control)
    assert output.shape == (3, 12, 10)


def test_multiple_layers_stack_correctly():
    model = SequenceModel(n_features=8, d_model=16, mixing_primitive_name="standard_attention", n_layers=3)
    assert len(model.blocks) == 3
    input_features = torch.randn(2, 10, 8)
    control = torch.zeros(2, 10, CONTROL_CHANNELS)
    output = model(input_features, control)
    assert output.shape == (2, 10, 8)


def test_strictly_causal():
    model = SequenceModel(n_features=6, d_model=16, mixing_primitive_name="standard_attention", n_layers=2)
    torch.manual_seed(0)
    input_features = torch.randn(2, 10, 6)
    control = torch.zeros(2, 10, CONTROL_CHANNELS)
    with torch.no_grad():
        y1 = model(input_features, control)

        perturbed = input_features.clone()
        perturbed[:, -1, :] += 10.0
        y2 = model(perturbed, control)

    assert torch.allclose(y1[:, :-1, :], y2[:, :-1, :], atol=1e-5)


def test_gradients_flow():
    model = SequenceModel(n_features=6, d_model=16, mixing_primitive_name="delta_net", n_layers=1)
    input_features = torch.randn(2, 8, 6, requires_grad=True)
    control = torch.zeros(2, 8, CONTROL_CHANNELS)
    output = model(input_features, control)
    output.pow(2).sum().backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, f"no gradient reached {name}"


def test_unknown_primitive_raises_not_implemented():
    with pytest.raises(NotImplementedError):
        SequenceModel(n_features=6, d_model=16, mixing_primitive_name="quantum_attention", n_layers=1)


def test_num_parameters_is_positive_and_reported():
    model = SequenceModel(n_features=6, d_model=16, mixing_primitive_name="standard_attention", n_layers=1)
    assert model.num_parameters() > 0


def test_standard_attention_solves_the_easy_recall_regime():
    # doc 4 §4/§8: standard attention is the positive control for the
    # recall dimension. A model that cannot learn to recall a single
    # nearby pointer on a low-sparsity, short-sequence task within a
    # short training run indicates a bug in the benchmark or model
    # wiring, not a real finding. This checks retrieval_accuracy directly
    # (not just aggregate loss, which is dominated by the majority of
    # trivial content positions and can look fine even when recall itself
    # has not been learned at all). Learning this task shows a sharp,
    # grokking-like phase transition (near-zero accuracy, then a jump to
    # ~99% within a few hundred steps) rather than a smooth ramp -- verified
    # empirically before picking these hyperparameters, which sit
    # comfortably past the observed transition point with margin.
    torch.manual_seed(0)
    generator = torch.Generator()
    generator.manual_seed(0)

    model = SequenceModel(n_features=8, d_model=32, mixing_primitive_name="standard_attention", n_layers=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)

    for _ in range(2000):
        batch = generate_recall_batch(
            batch_size=64,
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

    model.eval()
    with torch.no_grad():
        eval_batch = generate_recall_batch(
            batch_size=256,
            seq_len=16,
            n_features=8,
            n_pointers=2,
            sparsity=0.3,
            min_gap=2,
            generator=generator,
        )
        predicted = model(eval_batch.input_features, eval_batch.control)
    result = retrieval_accuracy(eval_batch.target, predicted, eval_batch.is_pointer, threshold=0.05)
    assert result["accuracy"] > 0.8, result
