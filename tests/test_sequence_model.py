from __future__ import annotations

import pytest
import torch

from superposition_zoo.models.sequence_model import SequenceModel
from superposition_zoo.recall_task import generate_recall_batch


@pytest.mark.parametrize("primitive_name", ["standard_attention", "linear_attention", "delta_net"])
def test_shape_preserved_across_primitives(primitive_name):
    model = SequenceModel(n_features=10, d_model=16, mixing_primitive_name=primitive_name, n_layers=1)
    input_features = torch.randn(3, 12, 10)
    control = torch.zeros(3, 12, 2)
    output = model(input_features, control)
    assert output.shape == (3, 12, 10)


def test_multiple_layers_stack_correctly():
    model = SequenceModel(n_features=8, d_model=16, mixing_primitive_name="standard_attention", n_layers=3)
    assert len(model.blocks) == 3
    input_features = torch.randn(2, 10, 8)
    control = torch.zeros(2, 10, 2)
    output = model(input_features, control)
    assert output.shape == (2, 10, 8)


def test_strictly_causal():
    model = SequenceModel(n_features=6, d_model=16, mixing_primitive_name="standard_attention", n_layers=2)
    torch.manual_seed(0)
    input_features = torch.randn(2, 10, 6)
    control = torch.zeros(2, 10, 2)
    with torch.no_grad():
        y1 = model(input_features, control)

        perturbed = input_features.clone()
        perturbed[:, -1, :] += 10.0
        y2 = model(perturbed, control)

    assert torch.allclose(y1[:, :-1, :], y2[:, :-1, :], atol=1e-5)


def test_gradients_flow():
    model = SequenceModel(n_features=6, d_model=16, mixing_primitive_name="delta_net", n_layers=1)
    input_features = torch.randn(2, 8, 6, requires_grad=True)
    control = torch.zeros(2, 8, 2)
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
    # wiring, not a real finding.
    torch.manual_seed(0)
    generator = torch.Generator()
    generator.manual_seed(0)

    model = SequenceModel(n_features=8, d_model=32, mixing_primitive_name="standard_attention", n_layers=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    losses = []
    for _ in range(300):
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
        losses.append(loss.item())

    early_loss = sum(losses[:10]) / 10
    late_loss = sum(losses[-10:]) / 10
    assert late_loss < early_loss / 3
