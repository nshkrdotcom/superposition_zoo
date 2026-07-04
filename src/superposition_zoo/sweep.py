"""The zoo sweep driver: primitives x seeds -> a comparison table.

Replication is not optional (doc 5 §2 principle 3): callers always pass a
list of seeds, and :func:`summarize_sweep` always reports spread and
direction-agreement alongside the mean, not a bare point estimate.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pandas as pd

from superposition_zoo.config import RunConfig
from superposition_zoo.metrics.reliability import direction_agreement, reliability_summary
from superposition_zoo.train import train


def run_sweep(
    primitives: list[str],
    seeds: list[int],
    base_config: RunConfig,
    runs_root: Path | None = None,
    device: str = "cpu",
) -> pd.DataFrame:
    """Train every ``(primitive, seed)`` combination and collect a result table.

    Args:
        primitives: mixing-primitive registry names to compare.
        seeds: seeds to replicate every primitive across.
        base_config: a :class:`RunConfig` giving everything *except*
            ``model.mixing_primitive_name`` and ``train.seed``, which this
            function overrides per row.
        runs_root: if given, each run's artifacts are written to
            ``runs_root / f"{primitive}_seed{seed}"``.
        device: torch device string, forwarded to :func:`train`.

    Returns:
        A ``pandas.DataFrame`` with one row per ``(primitive, seed)``.
    """
    rows = []
    for primitive in primitives:
        for seed in seeds:
            config = dataclasses.replace(
                base_config,
                model=dataclasses.replace(base_config.model, mixing_primitive_name=primitive),
                train=dataclasses.replace(base_config.train, seed=seed),
            )
            run_dir = runs_root / f"{primitive}_seed{seed}" if runs_root is not None else None
            result = train(config, run_dir=run_dir, device=device)
            rows.append(
                {
                    "primitive": primitive,
                    "seed": seed,
                    "final_loss": result.final_loss,
                    "num_parameters": result.num_parameters,
                    "recall_accuracy": result.recall_metrics["accuracy"],
                    "fraction_well_reconstructed": result.packing_metrics["fraction_well_reconstructed"],
                }
            )
    return pd.DataFrame(rows)


def summarize_sweep(
    df: pd.DataFrame, metric: str, baseline_primitive: str | None = None
) -> dict[str, dict]:
    """Per-primitive reliability summary for one metric column of a sweep table.

    Args:
        df: a sweep result table from :func:`run_sweep`.
        metric: which column to summarize (e.g. ``"final_loss"``).
        baseline_primitive: if given, every non-baseline primitive also gets
            a ``direction_agreement_vs_baseline`` entry (paired per seed).

    Returns:
        A dict keyed by primitive name.
    """
    summary: dict[str, dict] = {}
    baseline_values = None
    if baseline_primitive is not None:
        baseline_rows = df[df["primitive"] == baseline_primitive].sort_values("seed")
        baseline_values = baseline_rows[metric].tolist()

    for primitive, group in df.groupby("primitive"):
        group_sorted = group.sort_values("seed")
        entry = reliability_summary(group_sorted[metric].tolist())
        if baseline_primitive is not None and primitive != baseline_primitive:
            entry["direction_agreement_vs_baseline"] = direction_agreement(
                group_sorted[metric].tolist(), baseline_values
            )
        summary[primitive] = entry
    return summary
