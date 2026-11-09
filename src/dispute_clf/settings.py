"""Settings for preprocessing and the TF-IDF baseline, loaded from YAML."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, PositiveInt


class PreprocessingPaths(BaseModel):
    model_config = ConfigDict(frozen=True)

    generated: Path
    clean: Path
    dedup_report: Path
    splits_dir: Path
    split_report: Path


class PreprocessingConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    near_threshold: float = Field(gt=0, le=1)
    semantic_threshold: float = Field(gt=0, le=1)
    noise_seed: int
    split_seed: int
    val_fraction: float = Field(ge=0, lt=1)
    test_fraction: float = Field(ge=0, lt=1)
    paths: PreprocessingPaths


class TfidfPaths(BaseModel):
    model_config = ConfigDict(frozen=True)

    splits_dir: Path
    output_dir: Path


class TfidfConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    c_grid: tuple[float, ...]
    cv_folds: PositiveInt
    min_df: PositiveInt
    seed: int
    bootstrap_resamples: PositiveInt
    paths: TfidfPaths


def load_preprocessing_config(path: Path) -> PreprocessingConfig:
    with path.open(encoding="utf-8") as handle:
        return PreprocessingConfig.model_validate(yaml.safe_load(handle))


def load_tfidf_config(path: Path) -> TfidfConfig:
    with path.open(encoding="utf-8") as handle:
        return TfidfConfig.model_validate(yaml.safe_load(handle))
