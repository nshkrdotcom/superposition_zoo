"""``uv run szoo ...`` entry point.

Thin argparse wrapper over :mod:`superposition_zoo.train` and
:mod:`superposition_zoo.sweep`. No queue, no daemon, no ledger (doc 5 §10) --
a sweep at this repo's scale finishes in an afternoon on a single GPU, so
there is nothing here for a scheduler to do.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from superposition_zoo.causal_check import causal_check_report, load_trained_model
from superposition_zoo.config import load_config, load_phase0_config
from superposition_zoo.recall_task import generate_recall_batch
from superposition_zoo.sweep import run_sweep, summarize_sweep
from superposition_zoo.train import train, train_phase0


def _cmd_phase0(args: argparse.Namespace) -> None:
    config = load_phase0_config(args.config)
    run_dir = Path(args.runs_root) / config.name
    result = train_phase0(config, run_dir=run_dir, device=args.device)
    print(f"phase0 run '{config.name}' final_loss={result['final_loss']:.6f}")
    print(f"capacity_summary={result['capacity_summary']}")
    print(f"artifacts written to {run_dir}")


def _cmd_train(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    run_dir = Path(args.runs_root) / config.name
    result = train(config, run_dir=run_dir, device=args.device)
    print(
        f"run '{config.name}' ({config.model.mixing_primitive_name}) "
        f"final_loss={result.final_loss:.6f}"
    )
    print(f"recall_metrics={result.recall_metrics}")
    print(f"packing_metrics={result.packing_metrics}")
    print(f"artifacts written to {run_dir}")


def _cmd_sweep(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    primitives = args.primitives.split(",")
    seeds = [int(s) for s in args.seeds.split(",")]
    runs_root = Path(args.runs_root) / config.name

    df = run_sweep(
        primitives=primitives, seeds=seeds, base_config=config, runs_root=runs_root, device=args.device
    )
    runs_root.mkdir(parents=True, exist_ok=True)
    df.to_csv(runs_root / "sweep_results.csv", index=False)

    print(df.to_string(index=False))
    summary = summarize_sweep(df, metric="final_loss", baseline_primitive=primitives[0])
    print("\nReliability summary (final_loss, lower is better):")
    for primitive, stats in summary.items():
        print(f"  {primitive}: {stats}")
    print(f"\nfull results written to {runs_root / 'sweep_results.csv'}")


def _cmd_causal_check(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    # a fresh, held-out generator seed so the check isn't run on
    # training-time-seen examples
    generator = torch.Generator()
    generator.manual_seed(config.train.seed + 10_000)

    model = load_trained_model(
        checkpoint_path=args.checkpoint,
        n_features=config.features.n_features,
        d_model=config.model.d_model,
        mixing_primitive_name=config.model.mixing_primitive_name,
        n_layers=config.model.n_layers,
        mixing_kwargs=config.model.mixing_kwargs,
        device=args.device,
    )
    batch = generate_recall_batch(
        batch_size=args.batch_size,
        seq_len=config.recall_task.seq_len,
        n_features=config.features.n_features,
        n_pointers=config.recall_task.n_pointers,
        sparsity=config.features.sparsity,
        min_gap=config.recall_task.min_gap,
        generator=generator,
    )
    report = causal_check_report(model, batch, n_checks=args.n_checks, generator=generator)
    print(f"causal check for {config.model.mixing_primitive_name} ({args.checkpoint}):")
    for key, value in report.items():
        print(f"  {key}: {value}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="szoo")
    subparsers = parser.add_subparsers(dest="command", required=True)

    phase0_parser = subparsers.add_parser(
        "phase0", help="Train the Phase 0 toy-superposition autoencoder"
    )
    phase0_parser.add_argument("--config", required=True)
    phase0_parser.add_argument("--runs-root", default="runs", dest="runs_root")
    phase0_parser.add_argument("--device", default="cpu")
    phase0_parser.set_defaults(func=_cmd_phase0)

    train_parser = subparsers.add_parser(
        "train", help="Train a single Phase 1 (cross-token recall) run"
    )
    train_parser.add_argument("--config", required=True)
    train_parser.add_argument("--runs-root", default="runs", dest="runs_root")
    train_parser.add_argument("--device", default="cpu")
    train_parser.set_defaults(func=_cmd_train)

    sweep_parser = subparsers.add_parser("sweep", help="Compare mixing primitives across seeds")
    sweep_parser.add_argument("--config", required=True)
    sweep_parser.add_argument("--primitives", required=True, help="comma-separated registry names")
    sweep_parser.add_argument("--seeds", required=True, help="comma-separated seeds")
    sweep_parser.add_argument("--runs-root", default="runs", dest="runs_root")
    sweep_parser.add_argument("--device", default="cpu")
    sweep_parser.set_defaults(func=_cmd_sweep)

    causal_check_parser = subparsers.add_parser(
        "causal-check",
        help="Ground-truth activation patching against a real trained checkpoint",
    )
    causal_check_parser.add_argument("--config", required=True)
    causal_check_parser.add_argument("--checkpoint", required=True)
    causal_check_parser.add_argument("--n-checks", type=int, default=50, dest="n_checks")
    causal_check_parser.add_argument("--batch-size", type=int, default=256, dest="batch_size")
    causal_check_parser.add_argument("--device", default="cpu")
    causal_check_parser.set_defaults(func=_cmd_causal_check)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
