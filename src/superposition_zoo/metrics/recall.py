"""Ground-truth-anchored retrieval/recall metrics (Phase 1).

A pointer position's prediction is scored directly against the known true
source content -- no probing, no inference of what the "correct" retrieval
would have looked like.
"""

from __future__ import annotations

import torch


def retrieval_accuracy(
    target: torch.Tensor,
    predicted: torch.Tensor,
    is_pointer: torch.Tensor,
    threshold: float,
) -> dict[str, float | int]:
    """Fraction of pointer positions whose prediction is within ``threshold``.

    "Within threshold" means the per-position squared error (summed over
    the feature dimension) is strictly below ``threshold``. Only pointer
    positions are scored; non-pointer positions cannot affect this metric.

    Args:
        target: ``(batch, seq_len, n_features)``.
        predicted: ``(batch, seq_len, n_features)``.
        is_pointer: ``(batch, seq_len)`` bool.
        threshold: error threshold for "correct".

    Returns:
        A dict with ``num_pointers``, ``num_correct``, and ``accuracy``.
        ``accuracy`` is ``NaN`` (not a fabricated ``0.0`` or ``1.0``) when
        there are no pointer positions to score.
    """
    squared_error = ((target - predicted) ** 2).sum(dim=-1)
    pointer_errors = squared_error[is_pointer]
    num_pointers = int(is_pointer.sum().item())

    if num_pointers == 0:
        return {"num_pointers": 0, "num_correct": 0, "accuracy": float("nan")}

    num_correct = int((pointer_errors < threshold).sum().item())
    return {
        "num_pointers": num_pointers,
        "num_correct": num_correct,
        "accuracy": num_correct / num_pointers,
    }


def retrieval_accuracy_by_distance(
    target: torch.Tensor,
    predicted: torch.Tensor,
    is_pointer: torch.Tensor,
    source_position: torch.Tensor,
    threshold: float,
) -> dict[int, float]:
    """Retrieval accuracy grouped by ``distance = t - source_position[t]``.

    This is the standard MQAR-style degradation-by-distance breakdown: how
    much recall accuracy drops as the model has to reach further back.

    Returns:
        A dict mapping each observed integer distance to the accuracy among
        pointer positions at that exact distance.
    """
    squared_error = ((target - predicted) ** 2).sum(dim=-1)
    batch_size, seq_len = is_pointer.shape

    correct_by_distance: dict[int, list[bool]] = {}
    for b in range(batch_size):
        for t in range(seq_len):
            if not bool(is_pointer[b, t]):
                continue
            source = int(source_position[b, t].item())
            distance = t - source
            is_correct = bool(squared_error[b, t].item() < threshold)
            correct_by_distance.setdefault(distance, []).append(is_correct)

    return {
        distance: sum(flags) / len(flags) for distance, flags in correct_by_distance.items()
    }
