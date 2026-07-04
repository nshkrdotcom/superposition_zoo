from __future__ import annotations

import pytest
import torch


@pytest.fixture
def rng() -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(1234)
    return generator
