from __future__ import annotations

from superposition_zoo.config import FeatureConfig, ModelConfig, RecallTaskConfig, RunConfig, TrainConfig
from superposition_zoo.sweep import run_sweep


def _base_config() -> RunConfig:
    return RunConfig(
        name="sweep_smoke",
        features=FeatureConfig(n_features=6, sparsity=0.3),
        recall_task=RecallTaskConfig(seq_len=12, n_pointers=2, min_gap=2),
        model=ModelConfig(d_model=16, mixing_primitive_name="standard_attention", n_layers=1),
        train=TrainConfig(steps=15, batch_size=8, lr=1e-3, seed=0),
    )


def test_sweep_produces_one_row_per_primitive_and_seed():
    df = run_sweep(
        primitives=["standard_attention", "linear_attention"],
        seeds=[0, 1],
        base_config=_base_config(),
    )
    assert len(df) == 4
    assert set(df["primitive"]) == {"standard_attention", "linear_attention"}
    assert set(df["seed"]) == {0, 1}


def test_sweep_reports_expected_columns_with_no_nans():
    df = run_sweep(primitives=["standard_attention"], seeds=[0, 1], base_config=_base_config())
    expected_columns = {
        "primitive",
        "seed",
        "final_loss",
        "num_parameters",
        "recall_accuracy",
        "content_fraction_well_reconstructed",
        "pointer_fraction_well_reconstructed",
    }
    assert expected_columns.issubset(df.columns)
    assert df[list(expected_columns)].notna().all().all()


def test_sweep_persists_run_artifacts_when_runs_root_given(tmp_path):
    run_sweep(
        primitives=["standard_attention"],
        seeds=[0],
        base_config=_base_config(),
        runs_root=tmp_path,
    )
    assert (tmp_path / "standard_attention_seed0" / "summary.json").exists()


def test_summarize_sweep_reports_reliability_across_seeds():
    from superposition_zoo.sweep import summarize_sweep

    df = run_sweep(
        primitives=["standard_attention", "linear_attention"],
        seeds=[0, 1, 2],
        base_config=_base_config(),
    )
    summary = summarize_sweep(df, metric="final_loss", baseline_primitive="standard_attention")

    assert "standard_attention" in summary
    assert "linear_attention" in summary
    assert "mean" in summary["linear_attention"]
    assert "std" in summary["linear_attention"]
    # baseline vs. itself must show perfect agreement (every seed "ties",
    # i.e. does not beat itself) -- a sanity check on the plumbing, not a
    # claim about the metric's direction.
    assert "direction_agreement_vs_baseline" in summary["linear_attention"]
    assert "direction_agreement_vs_baseline" not in summary["standard_attention"]
