"""Real-text evaluation sets built from public datasets."""

from __future__ import annotations

from pathlib import Path

import yaml
from datasets import load_dataset
from pydantic import BaseModel, ConfigDict, model_validator

from dispute_clf.labels import Label


class ExternalExample(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    text: str
    label: Label
    source: str


class IntentMapping(BaseModel):
    model_config = ConfigDict(frozen=True)

    dataset: str
    split: str
    mapping: dict[Label, tuple[str, ...]]
    output: Path

    @model_validator(mode="after")
    def _intents_map_to_one_label(self) -> IntentMapping:
        intents = [intent for group in self.mapping.values() for intent in group]
        if len(intents) != len(set(intents)):
            raise ValueError("an intent is mapped to more than one label")
        return self

    def label_of(self, intent: str) -> Label | None:
        return next((label for label, group in self.mapping.items() if intent in group), None)


def load_intent_mapping(path: Path) -> IntentMapping:
    with path.open(encoding="utf-8") as handle:
        return IntentMapping.model_validate(yaml.safe_load(handle))


def map_rows(texts: list[str], intents: list[str], mapping: IntentMapping) -> list[ExternalExample]:
    examples = []
    for index, (text, intent) in enumerate(zip(texts, intents, strict=True)):
        label = mapping.label_of(intent)
        if label is not None:
            examples.append(
                ExternalExample(
                    id=f"{mapping.split}-{index:05d}",
                    text=" ".join(text.split()),
                    label=label,
                    source=intent,
                )
            )
    return examples


def build_mapped_set(mapping: IntentMapping) -> list[ExternalExample]:
    split = load_dataset(mapping.dataset, split=mapping.split)
    names = split.features["label"].names
    return map_rows(list(split["text"]), [names[i] for i in split["label"]], mapping)
