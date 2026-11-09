"""Apply each message's sampled typo level after deduplication."""

from __future__ import annotations

import zlib
from collections.abc import Iterable

from dispute_clf.noise import add_typos
from dispute_clf.records import Message


def message_seed(message_id: str, seed: int) -> int:
    return zlib.crc32(f"{seed}:{message_id}".encode())


def apply_typos(messages: Iterable[Message], *, seed: int) -> list[Message]:
    """Return copies with noisy ``text`` and the original kept in ``text_clean``."""
    return [
        message.model_copy(
            update={
                "text": add_typos(
                    message.text,
                    message.attributes.typo_level,
                    seed=message_seed(message.id, seed),
                ),
                "text_clean": message.text,
            }
        )
        for message in messages
    ]
