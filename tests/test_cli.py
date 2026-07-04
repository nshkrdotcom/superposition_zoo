from __future__ import annotations

import textwrap

from superposition_zoo.cli import main


def test_cli_train_end_to_end(tmp_path, monkeypatch, capsys):
    config_path = tmp_path / "run.yaml"
    config_path.write_text(
        textwrap.dedent(
            """
            name: cli_test_run
            features:
              n_features: 6
              sparsity: 0.3
            recall_task:
              seq_len: 12
              n_pointers: 2
              min_gap: 2
            model:
              d_model: 16
              mixing_primitive_name: standard_attention
            train:
              steps: 10
              batch_size: 8
            """
        )
    )
    runs_root = tmp_path / "runs"

    monkeypatch.setattr(
        "sys.argv",
        ["szoo", "train", "--config", str(config_path), "--runs-root", str(runs_root)],
    )
    main()

    captured = capsys.readouterr()
    assert "cli_test_run" in captured.out
    assert (runs_root / "cli_test_run" / "summary.json").exists()


def test_cli_phase0_end_to_end(tmp_path, monkeypatch, capsys):
    config_path = tmp_path / "phase0.yaml"
    config_path.write_text(
        textwrap.dedent(
            """
            name: cli_phase0_test
            n_features: 10
            d_hidden: 4
            sparsity: 0.7
            steps: 10
            batch_size: 16
            """
        )
    )
    runs_root = tmp_path / "runs"

    monkeypatch.setattr(
        "sys.argv",
        ["szoo", "phase0", "--config", str(config_path), "--runs-root", str(runs_root)],
    )
    main()

    captured = capsys.readouterr()
    assert "cli_phase0_test" in captured.out
    assert (runs_root / "cli_phase0_test" / "summary.json").exists()


def test_cli_sweep_end_to_end(tmp_path, monkeypatch, capsys):
    config_path = tmp_path / "sweep.yaml"
    config_path.write_text(
        textwrap.dedent(
            """
            name: cli_sweep_test
            features:
              n_features: 6
              sparsity: 0.3
            recall_task:
              seq_len: 12
              n_pointers: 2
              min_gap: 2
            model:
              d_model: 16
              mixing_primitive_name: standard_attention
            train:
              steps: 8
              batch_size: 8
            """
        )
    )
    runs_root = tmp_path / "runs"

    monkeypatch.setattr(
        "sys.argv",
        [
            "szoo",
            "sweep",
            "--config",
            str(config_path),
            "--primitives",
            "standard_attention,linear_attention",
            "--seeds",
            "0,1",
            "--runs-root",
            str(runs_root),
        ],
    )
    main()

    captured = capsys.readouterr()
    assert "Reliability summary" in captured.out
    assert (runs_root / "cli_sweep_test" / "sweep_results.csv").exists()
