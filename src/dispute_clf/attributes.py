"""Attribute grid for controlled generation, loaded from ``configs/attributes.yaml``."""

from __future__ import annotations

import random
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, field_validator

from dispute_clf.noise import TypoLevel


class Attributes(BaseModel):
    """One sampled combination of attributes, recorded with every generated message."""

    model_config = ConfigDict(frozen=True)

    persona: str
    channel: str
    length: str
    english_variety: str
    currency: str
    merchant_type: str
    tone: str
    typo_level: TypoLevel


class AttributeGrid(BaseModel):
    model_config = ConfigDict(frozen=True)

    persona: tuple[str, ...]
    channel: tuple[str, ...]
    length: dict[str, str]
    english_variety: tuple[str, ...]
    currency: tuple[str, ...]
    merchant_type: tuple[str, ...]
    tone: tuple[str, ...]
    typo_level: tuple[TypoLevel, ...]

    @field_validator("persona", "channel", "english_variety", "currency", "merchant_type", "tone")
    @classmethod
    def _non_empty(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if not values:
            raise ValueError("attribute value lists must not be empty")
        return values

    def sample(self, rng: random.Random) -> Attributes:
        return Attributes(
            persona=rng.choice(self.persona),
            channel=rng.choice(self.channel),
            length=rng.choice(sorted(self.length)),
            english_variety=rng.choice(self.english_variety),
            currency=rng.choice(self.currency),
            merchant_type=rng.choice(self.merchant_type),
            tone=rng.choice(self.tone),
            typo_level=rng.choice(self.typo_level),
        )

    def describe_length(self, key: str) -> str:
        return self.length[key]


def load_attribute_grid(path: Path) -> AttributeGrid:
    with path.open(encoding="utf-8") as handle:
        return AttributeGrid.model_validate(yaml.safe_load(handle))
