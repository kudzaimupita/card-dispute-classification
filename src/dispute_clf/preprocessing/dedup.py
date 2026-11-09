"""Three-stage deduplication: exact, lexical near-duplicate and semantic.

Stages run in order of cost. Within each stage the first message by id is kept, so results
are deterministic. Every removal is recorded with the stage and the message it duplicates,
so counts can be reported per generator and per label.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from datasketch import MinHash, MinHashLSH
from pydantic import BaseModel, ConfigDict
from semhash import SemHash

from dispute_clf.records import Message

_NON_WORD = re.compile(r"[^\w\s]")
_SPACES = re.compile(r"\s+")


class Removal(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    stage: str
    duplicate_of: str


class DedupResult(BaseModel):
    kept: list[Message]
    removed: list[Removal]

    def count(self, stage: str) -> int:
        return sum(removal.stage == stage for removal in self.removed)


def normalise(text: str) -> str:
    return _SPACES.sub(" ", _NON_WORD.sub(" ", text.lower())).strip()


def drop_exact(messages: Sequence[Message]) -> DedupResult:
    seen: dict[str, str] = {}
    kept: list[Message] = []
    removed: list[Removal] = []
    for message in sorted(messages, key=lambda m: m.id):
        key = normalise(message.text)
        if key in seen:
            removed.append(Removal(id=message.id, stage="exact", duplicate_of=seen[key]))
        else:
            seen[key] = message.id
            kept.append(message)
    return DedupResult(kept=kept, removed=removed)


def _minhash(text: str, *, shingle: int, num_perm: int) -> MinHash:
    normalised = normalise(text)
    grams = {normalised[i : i + shingle] for i in range(max(len(normalised) - shingle + 1, 1))}
    signature = MinHash(num_perm=num_perm)
    for gram in grams:
        signature.update(gram.encode())
    return signature


def drop_near_duplicates(
    messages: Sequence[Message], *, threshold: float = 0.85, shingle: int = 5, num_perm: int = 128
) -> DedupResult:
    """Remove messages whose character-shingle Jaccard similarity exceeds ``threshold``."""
    index = MinHashLSH(threshold=threshold, num_perm=num_perm)
    kept: list[Message] = []
    removed: list[Removal] = []
    for message in sorted(messages, key=lambda m: m.id):
        signature = _minhash(message.text, shingle=shingle, num_perm=num_perm)
        matches = sorted(index.query(signature))
        if matches:
            removed.append(Removal(id=message.id, stage="near", duplicate_of=matches[0]))
        else:
            index.insert(message.id, signature)
            kept.append(message)
    return DedupResult(kept=kept, removed=removed)


def drop_semantic(messages: Sequence[Message], *, threshold: float = 0.9) -> DedupResult:
    """Remove messages whose embedding similarity to an earlier message exceeds ``threshold``."""
    ordered = sorted(messages, key=lambda m: m.id)
    records = [{"id": message.id, "text": message.text} for message in ordered]
    result = SemHash.from_records(records=records, columns=["text"]).self_deduplicate(
        threshold=threshold
    )
    removed_ids = {
        duplicate.record["id"]: duplicate.duplicate_of["id"] for duplicate in result.filtered
    }
    return DedupResult(
        kept=[message for message in ordered if message.id not in removed_ids],
        removed=[
            Removal(id=removed, stage="semantic", duplicate_of=original)
            for removed, original in sorted(removed_ids.items())
        ],
    )


def deduplicate(
    messages: Sequence[Message],
    *,
    near_threshold: float = 0.85,
    semantic_threshold: float | None = 0.9,
) -> DedupResult:
    """Run all stages. Set ``semantic_threshold`` to ``None`` to skip the semantic stage."""
    exact = drop_exact(messages)
    near = drop_near_duplicates(exact.kept, threshold=near_threshold)
    removed = exact.removed + near.removed
    kept = near.kept
    if semantic_threshold is not None:
        semantic = drop_semantic(kept, threshold=semantic_threshold)
        kept, removed = semantic.kept, removed + semantic.removed
    return DedupResult(kept=kept, removed=removed)


def cross_split_overlap(
    reference: Sequence[Message], candidates: Sequence[Message], *, threshold: float = 0.9
) -> list[str]:
    """Return ids of ``candidates`` that are semantic near-duplicates of any ``reference``."""
    index = SemHash.from_records(
        records=[{"id": m.id, "text": m.text} for m in reference], columns=["text"]
    )
    result = index.deduplicate(
        records=[{"id": m.id, "text": m.text} for m in candidates], threshold=threshold
    )
    return sorted(duplicate.record["id"] for duplicate in result.filtered)
