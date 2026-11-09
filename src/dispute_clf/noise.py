"""Seeded typographic noise.

Noise is applied after generation so that the typo distribution is identical for every
generating model and can be ablated. Only alphabetic words of three or more letters are
altered; numbers, amounts, currency codes and punctuation are left untouched.
"""

from __future__ import annotations

import random
import re
from collections.abc import Callable
from enum import StrEnum


class TypoLevel(StrEnum):
    NONE = "none"
    LIGHT = "light"
    HEAVY = "heavy"


# Probability that an eligible word receives one edit.
WORD_EDIT_RATE: dict[TypoLevel, float] = {
    TypoLevel.NONE: 0.0,
    TypoLevel.LIGHT: 0.05,
    TypoLevel.HEAVY: 0.15,
}

_WORD = re.compile(r"[A-Za-z]{3,}")

_QWERTY_ROWS = ("qwertyuiop", "asdfghjkl", "zxcvbnm")
_NEIGHBOURS: dict[str, str] = {
    char: row[max(i - 1, 0) : i] + row[i + 1 : i + 2]
    for row in _QWERTY_ROWS
    for i, char in enumerate(row)
}


def _swap(word: str, rng: random.Random) -> str:
    i = rng.randrange(len(word) - 1)
    return word[:i] + word[i + 1] + word[i] + word[i + 2 :]


def _delete(word: str, rng: random.Random) -> str:
    i = rng.randrange(len(word))
    return word[:i] + word[i + 1 :]


def _double(word: str, rng: random.Random) -> str:
    i = rng.randrange(len(word))
    return word[: i + 1] + word[i] + word[i + 1 :]


def _neighbour(word: str, rng: random.Random) -> str:
    i = rng.randrange(len(word))
    original = word[i]
    candidates = _NEIGHBOURS.get(original.lower(), "")
    if not candidates:
        return word
    replacement = rng.choice(candidates)
    if original.isupper():
        replacement = replacement.upper()
    return word[:i] + replacement + word[i + 1 :]


_EDITS: tuple[Callable[[str, random.Random], str], ...] = (_swap, _delete, _double, _neighbour)


def add_typos(text: str, level: TypoLevel, seed: int) -> str:
    """Return ``text`` with seeded keyboard-style typos at the given level."""
    rate = WORD_EDIT_RATE[level]
    if rate == 0.0:
        return text
    rng = random.Random(seed)

    def edit(match: re.Match[str]) -> str:
        word = match.group(0)
        if rng.random() >= rate:
            return word
        return rng.choice(_EDITS)(word, rng)

    return _WORD.sub(edit, text)
