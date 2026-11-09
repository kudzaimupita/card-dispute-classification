"""Label set and label schema loaded from ``configs/labels.yaml``."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, model_validator


class Label(StrEnum):
    """The five target classes. Order is the canonical index order for models."""

    FRAUD = "fraud"
    GOODS_NOT_RECEIVED = "goods_not_received"
    DUPLICATE_CHARGE = "duplicate_charge"
    SUBSCRIPTION_CANCELLATION = "subscription_cancellation"
    OTHER = "other"

    @classmethod
    def ordered(cls) -> list[Label]:
        return list(cls)


class LabelSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: Label
    name: str
    definition: str
    includes: tuple[str, ...] = ()
    excludes: tuple[str, ...] = ()


class LabelSchema(BaseModel):
    model_config = ConfigDict(frozen=True)

    labels: tuple[LabelSpec, ...]
    boundary_rules: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _covers_every_label_once(self) -> LabelSchema:
        ids = [spec.id for spec in self.labels]
        if sorted(ids) != sorted(Label.ordered()) or len(ids) != len(set(ids)):
            raise ValueError(f"label schema must define each label exactly once, got {ids}")
        return self

    def spec(self, label: Label) -> LabelSpec:
        return next(spec for spec in self.labels if spec.id is label)


def load_label_schema(path: Path) -> LabelSchema:
    with path.open(encoding="utf-8") as handle:
        return LabelSchema.model_validate(yaml.safe_load(handle))
