from __future__ import annotations

import torch

from superposition_zoo.models.toy_autoencoder import ToyAutoencoder


def test_forward_shape_preserved():
    model = ToyAutoencoder(n_features=20, d_hidden=5)
    x = torch.rand(16, 20)
    y = model(x)
    assert y.shape == (16, 20)


def test_bottleneck_forces_dimensionality_reduction():
    model = ToyAutoencoder(n_features=20, d_hidden=5)
    assert model.encoder.out_features == 5
    assert model.decoder.in_features == 5


def test_gradients_flow_to_all_parameters():
    model = ToyAutoencoder(n_features=10, d_hidden=3)
    x = torch.rand(8, 10)
    y = model(x)
    loss = y.pow(2).sum()
    loss.backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, f"no gradient reached {name}"
        assert torch.any(param.grad != 0.0), f"gradient is exactly zero for {name}"


def test_num_parameters_matches_hand_computation():
    n_features, d_hidden = 12, 4
    model = ToyAutoencoder(n_features=n_features, d_hidden=d_hidden, bias=True)
    # encoder: (n_features -> d_hidden) weight + bias; decoder: (d_hidden -> n_features) weight + bias
    expected = (n_features * d_hidden + d_hidden) + (d_hidden * n_features + n_features)
    assert model.num_parameters() == expected


def test_can_disable_bias():
    n_features, d_hidden = 12, 4
    model = ToyAutoencoder(n_features=n_features, d_hidden=d_hidden, bias=False)
    expected = (n_features * d_hidden) + (d_hidden * n_features)
    assert model.num_parameters() == expected


def test_perfect_identity_case_is_learnable_in_a_few_steps():
    # sanity: with d_hidden == n_features the model should trivially learn identity-ish
    # reconstruction on dense (non-sparse) input within a handful of steps.
    torch.manual_seed(0)
    model = ToyAutoencoder(n_features=6, d_hidden=6)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    x = torch.rand(64, 6)
    initial_loss = None
    final_loss = None
    for step in range(200):
        optimizer.zero_grad()
        y = model(x)
        loss = (x - y).pow(2).mean()
        if step == 0:
            initial_loss = loss.item()
        loss.backward()
        optimizer.step()
        final_loss = loss.item()
    assert final_loss < initial_loss / 5
