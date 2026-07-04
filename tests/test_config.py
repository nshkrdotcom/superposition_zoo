from __future__ import annotations

import textwrap

from superposition_zoo.config import load_config


def test_load_config_parses_nested_structure(tmp_path):
    config_path = tmp_path / "run.yaml"
    config_path.write_text(
        textwrap.dedent(
            """
            name: my_run
            features:
              n_features: 10
              sparsity: 0.6
            recall_task:
              seq_len: 32
              n_pointers: 4
              min_gap: 2
            model:
              d_model: 32
              mixing_primitive_name: standard_attention
              n_layers: 2
              mixing_kwargs:
                n_heads: 4
            train:
              steps: 100
              batch_size: 16
              lr: 0.001
              seed: 7
            retrieval_threshold: 0.02
            """
        )
    )

    config = load_config(config_path)

    assert config.name == "my_run"
    assert config.features.n_features == 10
    assert config.features.sparsity == 0.6
    assert config.recall_task.seq_len == 32
    assert config.recall_task.n_pointers == 4
    assert config.recall_task.min_gap == 2
    assert config.model.d_model == 32
    assert config.model.mixing_primitive_name == "standard_attention"
    assert config.model.n_layers == 2
    assert config.model.mixing_kwargs == {"n_heads": 4}
    assert config.train.steps == 100
    assert config.train.batch_size == 16
    assert config.train.lr == 0.001
    assert config.train.seed == 7
    assert config.retrieval_threshold == 0.02


def test_load_config_applies_defaults_for_optional_fields(tmp_path):
    config_path = tmp_path / "minimal.yaml"
    config_path.write_text(
        textwrap.dedent(
            """
            name: minimal_run
            features:
              n_features: 6
              sparsity: 0.5
            recall_task:
              seq_len: 16
              n_pointers: 2
            model:
              d_model: 16
              mixing_primitive_name: linear_attention
            train:
              steps: 10
              batch_size: 8
            """
        )
    )

    config = load_config(config_path)

    assert config.recall_task.min_gap == 1
    assert config.model.n_layers == 1
    assert config.model.mixing_kwargs == {}
    assert config.train.lr == 1e-3
    assert config.train.seed == 0
    assert config.retrieval_threshold == 0.05


def test_load_phase0_config(tmp_path):
    from superposition_zoo.config import load_phase0_config

    config_path = tmp_path / "phase0.yaml"
    config_path.write_text(
        textwrap.dedent(
            """
            name: phase0_run
            n_features: 20
            d_hidden: 5
            sparsity: 0.9
            steps: 500
            batch_size: 64
            lr: 0.001
            seed: 3
            threshold: 0.05
            """
        )
    )

    config = load_phase0_config(config_path)

    assert config.name == "phase0_run"
    assert config.n_features == 20
    assert config.d_hidden == 5
    assert config.sparsity == 0.9
    assert config.steps == 500
    assert config.batch_size == 64
    assert config.lr == 0.001
    assert config.seed == 3
    assert config.threshold == 0.05
