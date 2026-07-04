from __future__ import annotations

import torch

from superposition_zoo.metrics.recall import retrieval_accuracy, retrieval_accuracy_by_distance


def test_retrieval_accuracy_all_correct():
    target = torch.zeros(2, 4, 3)
    predicted = target.clone()
    is_pointer = torch.tensor([[False, True, False, True], [False, False, True, False]])
    result = retrieval_accuracy(target, predicted, is_pointer, threshold=0.01)
    assert result["num_pointers"] == 3
    assert result["num_correct"] == 3
    assert result["accuracy"] == 1.0


def test_retrieval_accuracy_all_wrong():
    target = torch.zeros(2, 4, 3)
    predicted = torch.full((2, 4, 3), 5.0)
    is_pointer = torch.tensor([[False, True, False, True], [False, False, True, False]])
    result = retrieval_accuracy(target, predicted, is_pointer, threshold=0.01)
    assert result["num_correct"] == 0
    assert result["accuracy"] == 0.0


def test_retrieval_accuracy_ignores_non_pointer_positions():
    target = torch.zeros(1, 3, 2)
    predicted = torch.zeros(1, 3, 2)
    # make non-pointer positions wildly wrong -- must not affect the metric
    predicted[0, 0] = 999.0
    predicted[0, 2] = -999.0
    is_pointer = torch.tensor([[False, True, False]])
    result = retrieval_accuracy(target, predicted, is_pointer, threshold=0.01)
    assert result["num_pointers"] == 1
    assert result["accuracy"] == 1.0


def test_retrieval_accuracy_no_pointers_returns_nan_not_fabricated():
    target = torch.zeros(1, 3, 2)
    predicted = torch.zeros(1, 3, 2)
    is_pointer = torch.zeros(1, 3, dtype=torch.bool)
    result = retrieval_accuracy(target, predicted, is_pointer, threshold=0.01)
    assert result["num_pointers"] == 0
    import math

    assert math.isnan(result["accuracy"])


def test_retrieval_accuracy_by_distance_buckets_correctly():
    # batch of 1, seq_len 5: pointer at t=3 with source=1 (distance 2, correct),
    # pointer at t=4 with source=0 (distance 4, wrong)
    target = torch.zeros(1, 5, 2)
    predicted = torch.zeros(1, 5, 2)
    predicted[0, 4] = 999.0  # wrong at distance-4 pointer
    is_pointer = torch.tensor([[False, False, False, True, True]])
    source_position = torch.tensor([[-1, -1, -1, 1, 0]])

    by_distance = retrieval_accuracy_by_distance(
        target, predicted, is_pointer, source_position, threshold=0.01
    )

    assert by_distance[2] == 1.0
    assert by_distance[4] == 0.0
