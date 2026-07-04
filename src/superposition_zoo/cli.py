"""``uv run szoo ...`` entry point.

Thin argparse wrapper over :mod:`superposition_zoo.train` and
:mod:`superposition_zoo.sweep`. No queue, no daemon, no ledger (doc 5 §10) --
a sweep at this repo's scale finishes in an afternoon on a single GPU, so
there is nothing here for a scheduler to do.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from superposition_zoo.config import load_config, load_phase0_config
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

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
