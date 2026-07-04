"""Config schemas: plain dataclasses read from YAML.

Deliberately much simpler than ``attention_lab``'s config system -- no
manifest hashing, no shard verification, because there is no real dataset
here to drift (doc 5 §5).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class FeatureConfig:
    n_features: int
    sparsity: float


@dataclass
class RecallTaskConfig:
    seq_len: int
    n_pointers: int
    min_gap: int = 1


@dataclass
class ModelConfig:
    d_model: int
    mixing_primitive_name: str
    n_layers: int = 1
    mixing_kwargs: dict = field(default_factory=dict)


@dataclass
class TrainConfig:
    steps: int
    batch_size: int
    lr: float = 1e-3
    seed: int = 0


@dataclass
class RunConfig:
    name: str
    features: FeatureConfig
    recall_task: RecallTaskConfig
    model: ModelConfig
    train: TrainConfig
    retrieval_threshold: float = 0.05


@dataclass
class Phase0Config:
    name: str
    n_features: int
    d_hidden: int
    sparsity: float
    steps: int
    batch_size: int
    lr: float = 1e-3
    seed: int = 0
    threshold: float = 0.05


def load_config(path: str | Path) -> RunConfig:
    """Load a Phase 1 (cross-token recall) run config from YAML."""
    raw = yaml.safe_load(Path(path).read_text())
    return RunConfig(
        name=raw["name"],
        features=FeatureConfig(**raw["features"]),
        recall_task=RecallTaskConfig(**raw["recall_task"]),
        model=ModelConfig(**raw["model"]),
        train=TrainConfig(**raw["train"]),
        retrieval_threshold=raw.get("retrieval_threshold", 0.05),
    )


def load_phase0_config(path: str | Path) -> Phase0Config:
    """Load a Phase 0 (toy superposition) run config from YAML."""
    raw = yaml.safe_load(Path(path).read_text())
    return Phase0Config(**raw)
