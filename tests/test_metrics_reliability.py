from __future__ import annotations

import pytest

from superposition_zoo.metrics.reliability import direction_agreement, reliability_summary


def test_reliability_summary_basic_stats():
    result = reliability_summary([1.0, 2.0, 3.0, 4.0, 5.0])
    assert result["n_seeds"] == 5
    assert result["mean"] == pytest.approx(3.0)
    assert result["min"] == 1.0
    assert result["max"] == 5.0
    assert result["std"] > 0.0


def test_reliability_summary_single_seed_has_zero_std():
    result = reliability_summary([2.5])
    assert result["n_seeds"] == 1
    assert result["mean"] == 2.5
    assert result["std"] == 0.0


def test_direction_agreement_all_agree():
    values = [1.0, 2.0, 3.0]
    baseline = [0.5, 1.0, 1.5]
    assert direction_agreement(values, baseline) == 1.0


def test_direction_agreement_all_disagree():
    values = [0.5, 1.0, 1.5]
    baseline = [1.0, 2.0, 3.0]
    assert direction_agreement(values, baseline) == 0.0


def test_direction_agreement_mixed():
    values = [1.0, 0.5, 3.0]
    baseline = [0.5, 1.0, 1.5]
    # seed 0: 1.0 > 0.5 (agree), seed 1: 0.5 < 1.0 (disagree), seed 2: 3.0 > 1.5 (agree)
    assert direction_agreement(values, baseline) == pytest.approx(2 / 3)


def test_direction_agreement_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        direction_agreement([1.0, 2.0], [1.0])


def test_reliability_summary_rejects_empty_input():
    with pytest.raises(ValueError):
        reliability_summary([])
