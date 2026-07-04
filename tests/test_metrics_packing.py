from __future__ import annotations

import math

import pytest
import torch

from superposition_zoo.metrics.packing import (
    capacity_summary,
    importance_weighted_loss,
    interference_matrix,
    per_feature_reconstruction_error,
)


def test_per_feature_reconstruction_error_zero_when_perfect():
    true_values = torch.rand(20, 5)
    error = per_feature_reconstruction_error(true_values, true_values.clone())
    assert error.shape == (5,)
    assert torch.allclose(error, torch.zeros(5))


def test_per_feature_reconstruction_error_matches_hand_computation():
    true_values = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    reconstructed = torch.tensor([[0.0, 0.0], [0.0, 0.0]])
    error = per_feature_reconstruction_error(true_values, reconstructed)
    # feature 0 squared errors: [1.0, 0.0] -> mean 0.5
    # feature 1 squared errors: [0.0, 1.0] -> mean 0.5
    assert torch.allclose(error, torch.tensor([0.5, 0.5]))


def test_interference_matrix_detects_engineered_interference():
    active_mask = torch.tensor(
        [
            [True, True, False],
            [True, False, False],
            [False, True, False],
            [False, False, False],
        ]
    )
    per_example_squared_error = torch.zeros(4, 3)
    # feature 0's error is exactly 1.0 whenever feature 1 is active, else 0.0
    per_example_squared_error[:, 0] = torch.tensor([1.0, 0.0, 1.0, 0.0])

    result = interference_matrix(active_mask, per_example_squared_error)

    assert result.shape == (3, 3)
    assert result[0, 1] == 1.0
    # feature 2 is never active in this batch: no contrast, must report NaN not a fabricated number
    assert math.isnan(result[0, 2].item())


def test_interference_matrix_diagonal_is_zero():
    active_mask = torch.rand(50, 4) > 0.5
    per_example_squared_error = torch.rand(50, 4)
    result = interference_matrix(active_mask, per_example_squared_error)
    assert torch.all(torch.diagonal(result) == 0.0)


def test_capacity_summary_all_well_reconstructed():
    per_feature_mse = torch.zeros(10)
    summary = capacity_summary(per_feature_mse, threshold=0.05)
    assert summary["num_features"] == 10
    assert summary["num_well_reconstructed"] == 10
    assert summary["fraction_well_reconstructed"] == 1.0


def test_capacity_summary_none_well_reconstructed():
    per_feature_mse = torch.full((10,), 1.0)
    summary = capacity_summary(per_feature_mse, threshold=0.05)
    assert summary["num_well_reconstructed"] == 0
    assert summary["fraction_well_reconstructed"] == 0.0


def test_capacity_summary_partial():
    per_feature_mse = torch.tensor([0.0, 0.0, 1.0, 1.0])
    summary = capacity_summary(per_feature_mse, threshold=0.05)
    assert summary["num_well_reconstructed"] == 2
    assert summary["fraction_well_reconstructed"] == 0.5


def test_importance_weighted_loss_uniform_importance_matches_summed_mse():
    true_values = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    reconstructed = torch.zeros(2, 2)
    importance = torch.ones(2)
    loss = importance_weighted_loss(true_values, reconstructed, importance)
    # per-example squared errors: [1, 0] and [0, 1] -> summed over features = 1.0 each
    # mean over batch = 1.0
    assert loss.item() == pytest.approx(1.0)


def test_importance_weighted_loss_emphasizes_high_importance_features():
    true_values = torch.tensor([[1.0, 1.0]])
    reconstructed = torch.tensor([[0.0, 0.0]])
    low_importance = torch.tensor([1.0, 0.1])
    high_importance = torch.tensor([1.0, 10.0])
    loss_low = importance_weighted_loss(true_values, reconstructed, low_importance)
    loss_high = importance_weighted_loss(true_values, reconstructed, high_importance)
    assert loss_high.item() > loss_low.item()


def test_importance_weighted_loss_is_scalar_and_differentiable():
    true_values = torch.rand(8, 5)
    reconstructed = torch.rand(8, 5, requires_grad=True)
    importance = torch.linspace(1.0, 0.1, 5)
    loss = importance_weighted_loss(true_values, reconstructed, importance)
    assert loss.ndim == 0
    loss.backward()
    assert reconstructed.grad is not None
    assert torch.any(reconstructed.grad != 0.0)
