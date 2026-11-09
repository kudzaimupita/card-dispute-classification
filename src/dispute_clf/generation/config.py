"""Generation settings loaded from ``configs/generation.yaml``."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, PositiveInt

Effort = Literal["low", "medium", "high", "xhigh", "max"]


class GenerationPaths(BaseModel):
    model_config = ConfigDict(frozen=True)

    labels: Path
    attributes: Path
    slots: Path
    batch: Path
    generated: Path
    failures: Path


class GenerationConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    model: str
    effort: Effort
    max_tokens: PositiveInt
    slots_per_label: PositiveInt
    messages_per_slot: PositiveInt
    seed: int
    paths: GenerationPaths


def load_generation_config(path: Path) -> GenerationConfig:
    with path.open(encoding="utf-8") as handle:
        return GenerationConfig.model_validate(yaml.safe_load(handle))
