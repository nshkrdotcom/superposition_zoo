from __future__ import annotations

import math

import pytest
import torch

from superposition_zoo.metrics.interference_report import position_masked_interference
from superposition_zoo.metrics.packing import interference_matrix, mean_absolute_interference


def test_position_masked_interference_matches_manual_masking():
    torch.manual_seed(0)
    target = torch.rand(4, 10, 3)
    predicted = torch.rand(4, 10, 3)
    active_mask = torch.rand(4, 10, 3) > 0.5
    position_mask = torch.rand(4, 10) > 0.5

    result = position_masked_interference(target, predicted, active_mask, position_mask)

    expected_error = (target[position_mask] - predicted[position_mask]) ** 2
    expected = interference_matrix(active_mask[position_mask], expected_error)

    torch.testing.assert_close(result, expected, equal_nan=True)


def test_position_masked_interference_shape():
    target = torch.rand(2, 8, 5)
    predicted = torch.rand(2, 8, 5)
    active_mask = torch.rand(2, 8, 5) > 0.5
    position_mask = torch.ones(2, 8, dtype=torch.bool)
    result = position_masked_interference(target, predicted, active_mask, position_mask)
    assert result.shape == (5, 5)


def test_mean_absolute_interference_ignores_diagonal_and_nan():
    mat = torch.tensor(
        [
            [0.0, 2.0, float("nan")],
            [-3.0, 0.0, 1.0],
            [float("nan"), 5.0, 0.0],
        ]
    )
    # off-diagonal, non-NaN values: 2.0, -3.0, 1.0, 5.0 -> abs mean = (2+3+1+5)/4 = 2.75
    result = mean_absolute_interference(mat)
    assert result == pytest.approx(2.75)


def test_mean_absolute_interference_all_nan_returns_nan():
    mat = torch.full((3, 3), float("nan"))
    mat.fill_diagonal_(0.0)
    result = mean_absolute_interference(mat)
    assert math.isnan(result)
