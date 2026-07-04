from __future__ import annotations

from superposition_zoo.config import (
    FeatureConfig,
    ModelConfig,
    Phase0Config,
    RecallTaskConfig,
    RunConfig,
    TrainConfig,
)
from superposition_zoo.train import train, train_phase0


def _tiny_run_config(mixing_primitive_name: str = "standard_attention", seed: int = 0) -> RunConfig:
    return RunConfig(
        name="tiny_test",
        features=FeatureConfig(n_features=6, sparsity=0.3),
        recall_task=RecallTaskConfig(seq_len=12, n_pointers=2, min_gap=2),
        model=ModelConfig(d_model=16, mixing_primitive_name=mixing_primitive_name, n_layers=1),
        train=TrainConfig(steps=20, batch_size=8, lr=1e-3, seed=seed),
    )


def test_train_tiny_run_creates_expected_artifacts(tmp_path):
    config = _tiny_run_config()
    run_dir = tmp_path / "tiny_test"

    result = train(config, run_dir=run_dir)

    assert (run_dir / "metrics.jsonl").exists()
    assert (run_dir / "checkpoint.pt").exists()
    assert (run_dir / "summary.json").exists()
    assert len(result.losses) == 20
    assert result.final_loss >= 0.0
    assert result.num_parameters > 0
    assert "accuracy" in result.recall_metrics
    assert "fraction_well_reconstructed" in result.packing_metrics
    # regression: packing metrics must be computed per (batch*time) example
    # over the n_features feature dimension, not accidentally treating the
    # sequence-length dimension as if it were a feature count.
    assert result.packing_metrics["num_features"] == config.features.n_features
    assert 0.0 <= result.packing_metrics["fraction_well_reconstructed"] <= 1.0
    assert 0.0 <= result.recall_metrics["accuracy"] <= 1.0


def test_train_without_run_dir_persists_nothing(tmp_path):
    config = _tiny_run_config()
    result = train(config, run_dir=None)
    assert len(result.losses) == 20
    assert not (tmp_path / "tiny_test").exists()


def test_train_is_deterministic_given_same_seed(tmp_path):
    config_a = _tiny_run_config(seed=42)
    config_b = _tiny_run_config(seed=42)
    result_a = train(config_a, run_dir=None)
    result_b = train(config_b, run_dir=None)
    assert result_a.losses == result_b.losses


def test_train_loss_decreases_on_easy_recall_regime():
    config = RunConfig(
        name="easy_regime",
        features=FeatureConfig(n_features=8, sparsity=0.3),
        recall_task=RecallTaskConfig(seq_len=16, n_pointers=2, min_gap=2),
        model=ModelConfig(d_model=32, mixing_primitive_name="standard_attention", n_layers=1),
        train=TrainConfig(steps=300, batch_size=32, lr=1e-3, seed=0),
    )
    result = train(config, run_dir=None)
    early = sum(result.losses[:10]) / 10
    late = sum(result.losses[-10:]) / 10
    assert late < early / 3


def test_train_works_for_every_registered_primitive():
    from superposition_zoo.mixing import REGISTRY

    for name in REGISTRY.available():
        if name == "vsa_binding":
            continue  # intentionally unimplemented stub, see mixing/vsa_binding.py
        config = _tiny_run_config(mixing_primitive_name=name)
        result = train(config, run_dir=None)
        assert len(result.losses) == 20, name


def _tiny_phase0_config(seed: int = 0) -> Phase0Config:
    return Phase0Config(
        name="tiny_phase0",
        n_features=10,
        d_hidden=4,
        sparsity=0.7,
        steps=20,
        batch_size=32,
        lr=1e-2,
        seed=seed,
    )


def test_train_phase0_creates_expected_artifacts(tmp_path):
    config = _tiny_phase0_config()
    run_dir = tmp_path / "tiny_phase0"

    result = train_phase0(config, run_dir=run_dir)

    assert (run_dir / "metrics.jsonl").exists()
    assert (run_dir / "checkpoint.pt").exists()
    assert (run_dir / "summary.json").exists()
    assert len(result["losses"]) == 20
    assert "capacity_summary" in result
    assert result["num_parameters"] > 0


def test_train_phase0_loss_decreases():
    config = Phase0Config(
        name="phase0_learns",
        n_features=15,
        d_hidden=5,
        sparsity=0.7,
        steps=400,
        batch_size=64,
        lr=1e-2,
        seed=0,
    )
    result = train_phase0(config, run_dir=None)
    early = sum(result["losses"][:10]) / 10
    late = sum(result["losses"][-10:]) / 10
    assert late < early
