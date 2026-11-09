"""Record types that flow through the pipeline, and JSONL persistence."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, ConfigDict

from dispute_clf.attributes import Attributes
from dispute_clf.labels import Label


class LabelledText(Protocol):
    """Anything with an id, a text and a gold label: generated or external examples."""

    @property
    def id(self) -> str: ...

    @property
    def text(self) -> str: ...

    @property
    def label(self) -> Label: ...


class Provenance(BaseModel):
    """Where a generated message came from. Enough to reproduce or audit it."""

    model_config = ConfigDict(frozen=True)

    model: str
    batch_id: str
    slot_id: str
    prompt_sha256: str
    generated_at: datetime


class Message(BaseModel):
    """A labelled message.

    ``group`` identifies the generation request that produced the message; all messages in
    a group share a label and attributes and are kept in the same split.
    """

    model_config = ConfigDict(frozen=True)

    id: str
    text: str
    label: Label
    group: str
    attributes: Attributes
    provenance: Provenance
    text_clean: str | None = None


def write_jsonl(path: Path, records: Iterable[BaseModel]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(record.model_dump_json() + "\n")
            count += 1
    return count


def read_jsonl[T: BaseModel](path: Path, model: type[T]) -> list[T]:
    with path.open(encoding="utf-8") as handle:
        return [model.model_validate_json(line) for line in handle if line.strip()]
