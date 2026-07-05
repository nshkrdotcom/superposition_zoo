"""Single-run training: config in, checkpoint + metrics.jsonl + summary out.

Two entry points: :func:`train` (Phase 1, cross-token recall) and
:func:`train_phase0` (Phase 0, the plain toy-superposition reproduction).
Modeled loosely on ``attention_lab``'s ``train()`` shape (config-in,
artifacts-out, a summary dict/dataclass) -- that shape was good; only the
content around it (FineWeb, HellaSwag, the queue) doesn't transfer here.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

from superposition_zoo.config import Phase0Config, RunConfig
from superposition_zoo.features import generate_features
from superposition_zoo.metrics.packing import (
    capacity_summary,
    importance_weighted_loss,
    per_feature_reconstruction_error,
    split_packing_summary,
)
from superposition_zoo.metrics.recall import retrieval_accuracy
from superposition_zoo.models.sequence_model import SequenceModel
from superposition_zoo.models.toy_autoencoder import ToyAutoencoder
from superposition_zoo.recall_task import generate_recall_batch


@dataclass
class RunResult:
    run_name: str
    mixing_primitive_name: str
    seed: int
    final_loss: float
    losses: list[float] = field(repr=False)
    packing_metrics: dict
    recall_metrics: dict
    num_parameters: int


def _write_metrics_jsonl(path: Path, losses: list[float]) -> None:
    with path.open("w") as f:
        for step, loss in enumerate(losses):
            f.write(json.dumps({"step": step, "loss": loss}) + "\n")


def train(config: RunConfig, run_dir: str | Path | None = None, device: str = "cpu") -> RunResult:
    """Train a :class:`SequenceModel` on the Phase 1 recall benchmark.

    Args:
        config: a :class:`~superposition_zoo.config.RunConfig`.
        run_dir: if given, ``metrics.jsonl``, ``checkpoint.pt``, and
            ``summary.json`` are written there. Accepts a plain string as
            well as a ``Path`` -- converted up front so a caller's typo
            fails immediately rather than after a full (possibly
            expensive) training run finishes and only the final artifact
            write blows up.
        device: torch device string.

    Returns:
        A :class:`RunResult`.
    """
    run_dir = Path(run_dir) if run_dir is not None else None
    generator = torch.Generator()
    generator.manual_seed(config.train.seed)
    torch.manual_seed(config.train.seed)

    model = SequenceModel(
        n_features=config.features.n_features,
        d_model=config.model.d_model,
        mixing_primitive_name=config.model.mixing_primitive_name,
        n_layers=config.model.n_layers,
        mixing_kwargs=config.model.mixing_kwargs,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.train.lr)

    losses: list[float] = []
    for _ in range(config.train.steps):
        batch = generate_recall_batch(
            batch_size=config.train.batch_size,
            seq_len=config.recall_task.seq_len,
            n_features=config.features.n_features,
            n_pointers=config.recall_task.n_pointers,
            sparsity=config.features.sparsity,
            min_gap=config.recall_task.min_gap,
            generator=generator,
        )
        input_features = batch.input_features.to(device)
        control = batch.control.to(device)
        target = batch.target.to(device)

        optimizer.zero_grad()
        predicted = model(input_features, control)
        loss = (predicted - target).pow(2).mean()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    model.eval()
    with torch.no_grad():
        eval_batch = generate_recall_batch(
            batch_size=max(config.train.batch_size, 64),
            seq_len=config.recall_task.seq_len,
            n_features=config.features.n_features,
            n_pointers=config.recall_task.n_pointers,
            sparsity=config.features.sparsity,
            min_gap=config.recall_task.min_gap,
            generator=generator,
        )
        predicted = model(eval_batch.input_features.to(device), eval_batch.control.to(device)).cpu()

    packing_metrics = split_packing_summary(
        eval_batch.target, predicted, eval_batch.is_pointer, threshold=config.retrieval_threshold
    )
    recall_metrics = retrieval_accuracy(
        eval_batch.target, predicted, eval_batch.is_pointer, threshold=config.retrieval_threshold
    )

    result = RunResult(
        run_name=config.name,
        mixing_primitive_name=config.model.mixing_primitive_name,
        seed=config.train.seed,
        final_loss=losses[-1],
        losses=losses,
        packing_metrics=packing_metrics,
        recall_metrics=recall_metrics,
        num_parameters=model.num_parameters(),
    )

    if run_dir is not None:
        run_dir.mkdir(parents=True, exist_ok=True)
        _write_metrics_jsonl(run_dir / "metrics.jsonl", losses)
        torch.save(model.state_dict(), run_dir / "checkpoint.pt")
        (run_dir / "summary.json").write_text(json.dumps(asdict(result), indent=2))

    return result


def train_phase0(config: Phase0Config, run_dir: str | Path | None = None, device: str = "cpu") -> dict:
    """Train a :class:`ToyAutoencoder` on the Phase 0 superposition benchmark.

    Returns a plain dict (not a dataclass) since Phase 0 has no recall
    metrics -- keeping its result shape visibly different from Phase 1's
    :class:`RunResult` avoids implying they are interchangeable.

    ``run_dir`` accepts a plain string as well as a ``Path`` -- see
    :func:`train`'s docstring for why this is converted up front.
    """
    run_dir = Path(run_dir) if run_dir is not None else None
    generator = torch.Generator()
    generator.manual_seed(config.seed)
    torch.manual_seed(config.seed)

    importance = torch.linspace(1.0, 0.1, config.n_features)

    model = ToyAutoencoder(n_features=config.n_features, d_hidden=config.d_hidden).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    losses: list[float] = []
    for _ in range(config.steps):
        values, _ = generate_features(
            n_features=config.n_features,
            batch_size=config.batch_size,
            sparsity=config.sparsity,
            generator=generator,
        )
        values = values.to(device)
        optimizer.zero_grad()
        reconstructed = model(values)
        loss = importance_weighted_loss(values, reconstructed, importance.to(device))
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    model.eval()
    with torch.no_grad():
        eval_values, _ = generate_features(
            n_features=config.n_features, batch_size=2000, sparsity=config.sparsity, generator=generator
        )
        eval_reconstructed = model(eval_values.to(device)).cpu()
    per_feature_mse = per_feature_reconstruction_error(eval_values, eval_reconstructed)
    summary = capacity_summary(per_feature_mse, threshold=config.threshold)

    result = {
        "run_name": config.name,
        "final_loss": losses[-1],
        "losses": losses,
        "capacity_summary": summary,
        "num_parameters": model.num_parameters(),
    }

    if run_dir is not None:
        run_dir.mkdir(parents=True, exist_ok=True)
        _write_metrics_jsonl(run_dir / "metrics.jsonl", losses)
        torch.save(model.state_dict(), run_dir / "checkpoint.pt")
        (run_dir / "summary.json").write_text(json.dumps(result, indent=2))

    return result
