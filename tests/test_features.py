from __future__ import annotations

import pytest
import torch

from superposition_zoo.features import generate_features


def test_shapes(rng):
    values, active_mask = generate_features(n_features=10, batch_size=32, sparsity=0.5, generator=rng)
    assert values.shape == (32, 10)
    assert active_mask.shape == (32, 10)
    assert active_mask.dtype == torch.bool


def test_inactive_features_are_exactly_zero(rng):
    values, active_mask = generate_features(n_features=10, batch_size=64, sparsity=0.7, generator=rng)
    assert torch.all(values[~active_mask] == 0.0)


def test_active_features_are_in_unit_interval(rng):
    values, active_mask = generate_features(n_features=10, batch_size=64, sparsity=0.3, generator=rng)
    active_values = values[active_mask]
    assert torch.all(active_values >= 0.0)
    assert torch.all(active_values <= 1.0)


def test_sparsity_rate_matches_target_within_tolerance(rng):
    n_features = 50
    batch_size = 4000
    sparsity = 0.9
    _, active_mask = generate_features(
        n_features=n_features, batch_size=batch_size, sparsity=sparsity, generator=rng
    )
    observed_density = active_mask.float().mean().item()
    expected_density = 1.0 - sparsity
    assert abs(observed_density - expected_density) < 0.02


def test_zero_sparsity_means_all_active(rng):
    _, active_mask = generate_features(n_features=5, batch_size=200, sparsity=0.0, generator=rng)
    assert torch.all(active_mask)


def test_full_sparsity_means_all_inactive(rng):
    values, active_mask = generate_features(n_features=5, batch_size=200, sparsity=1.0, generator=rng)
    assert not torch.any(active_mask)
    assert torch.all(values == 0.0)


def test_deterministic_given_same_generator_seed():
    gen1 = torch.Generator()
    gen1.manual_seed(42)
    gen2 = torch.Generator()
    gen2.manual_seed(42)
    v1, m1 = generate_features(n_features=8, batch_size=16, sparsity=0.5, generator=gen1)
    v2, m2 = generate_features(n_features=8, batch_size=16, sparsity=0.5, generator=gen2)
    assert torch.equal(v1, v2)
    assert torch.equal(m1, m2)


def test_rejects_invalid_scalar_sparsity(rng):
    with pytest.raises(ValueError):
        generate_features(n_features=5, batch_size=10, sparsity=1.5, generator=rng)
    with pytest.raises(ValueError):
        generate_features(n_features=5, batch_size=10, sparsity=-0.1, generator=rng)


def test_per_feature_sparsity_tensor(rng):
    # feature 0 always active (sparsity 0), feature 1 always inactive (sparsity 1)
    sparsity = torch.tensor([0.0, 1.0, 0.5, 0.5, 0.5])
    _, active_mask = generate_features(n_features=5, batch_size=500, sparsity=sparsity, generator=rng)
    assert torch.all(active_mask[:, 0])
    assert not torch.any(active_mask[:, 1])


def test_rejects_mismatched_sparsity_tensor_length(rng):
    sparsity = torch.tensor([0.0, 1.0])
    with pytest.raises(ValueError):
        generate_features(n_features=5, batch_size=10, sparsity=sparsity, generator=rng)
