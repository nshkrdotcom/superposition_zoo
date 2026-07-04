"""Cross-seed reliability scoring.

Doc 4 §8's requirement that a "reliability score" be a first-class output
of every comparison, not an afterthought: a mean effect size alone does not
say whether that effect showed up in every seed or only some of them.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch


def reliability_summary(values: Sequence[float]) -> dict[str, float | int]:
    """Mean/spread of a metric across seeds.

    Args:
        values: one value per seed.

    Returns:
        A dict with ``n_seeds``, ``mean``, ``std`` (``0.0`` for a single
        seed rather than an undefined/NaN value), ``min``, ``max``.
    """
    if len(values) == 0:
        raise ValueError("values must be non-empty")
    tensor = torch.as_tensor(list(values), dtype=torch.float32)
    std = tensor.std(unbiased=True).item() if tensor.numel() > 1 else 0.0
    return {
        "n_seeds": tensor.numel(),
        "mean": tensor.mean().item(),
        "std": std,
        "min": tensor.min().item(),
        "max": tensor.max().item(),
    }


def direction_agreement(values: Sequence[float], baseline_values: Sequence[float]) -> float:
    """Fraction of seeds where ``value > baseline`` (paired per seed).

    This operationalizes "not 80% of the time we see this and don't know
    why" into an explicit, reported number: how consistently, across seeds,
    a primitive beats its baseline on a given metric.

    Args:
        values: one value per seed for the primitive under test.
        baseline_values: the paired baseline (e.g. standard attention) value
            for the same seeds, in the same order.

    Returns:
        Fraction in ``[0, 1]``.
    """
    if len(values) != len(baseline_values):
        raise ValueError(
            f"values and baseline_values must have the same length (paired per seed), "
            f"got {len(values)} and {len(baseline_values)}"
        )
    v = torch.as_tensor(list(values), dtype=torch.float32)
    b = torch.as_tensor(list(baseline_values), dtype=torch.float32)
    return (v > b).float().mean().item()
