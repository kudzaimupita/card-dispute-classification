"""Label-stratified group split.

Each generation request (group) produces several messages with one label and one set of
attributes. Whole groups are assigned to a split so that near-paraphrases from the same
request never appear on both sides of a train/evaluation boundary. Groups are split per
label, which keeps the label distribution equal across splits.
"""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Sequence
from enum import StrEnum

from dispute_clf.labels import Label
from dispute_clf.records import Message


class Split(StrEnum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


def split_by_group(
    messages: Sequence[Message],
    *,
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
    seed: int,
) -> dict[Split, list[Message]]:
    if not 0 <= val_fraction + test_fraction < 1:
        raise ValueError("val_fraction + test_fraction must be in [0, 1)")

    groups_by_label: dict[Label, set[str]] = defaultdict(set)
    label_of_group: dict[str, Label] = {}
    for message in messages:
        previous = label_of_group.setdefault(message.group, message.label)
        if previous is not message.label:
            raise ValueError(f"group {message.group} mixes labels {previous} and {message.label}")
        groups_by_label[message.label].add(message.group)

    rng = random.Random(seed)
    split_of_group: dict[str, Split] = {}
    for label in Label.ordered():
        groups = sorted(groups_by_label.get(label, set()))
        rng.shuffle(groups)
        n_val = round(len(groups) * val_fraction)
        n_test = round(len(groups) * test_fraction)
        for index, group in enumerate(groups):
            if index < n_val:
                split_of_group[group] = Split.VAL
            elif index < n_val + n_test:
                split_of_group[group] = Split.TEST
            else:
                split_of_group[group] = Split.TRAIN

    splits: dict[Split, list[Message]] = {split: [] for split in Split}
    for message in sorted(messages, key=lambda m: m.id):
        splits[split_of_group[message.group]].append(message)
    return splits


def leaked_groups(splits: dict[Split, list[Message]]) -> set[str]:
    """Return groups that appear in more than one split. Empty for a valid split."""
    seen: dict[str, Split] = {}
    leaked: set[str] = set()
    for split, messages in splits.items():
        for message in messages:
            if seen.setdefault(message.group, split) is not split:
                leaked.add(message.group)
    return leaked
