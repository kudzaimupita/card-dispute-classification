from collections import Counter

import pytest

from dispute_clf.labels import Label
from dispute_clf.preprocessing.split import Split, leaked_groups, split_by_group


@pytest.fixture
def corpus(make_message):
    messages = []
    for label in Label:
        for group_index in range(20):
            group = f"{label.value}-{group_index:04d}"
            messages += [make_message(f"{group} {k}", label, group, k) for k in range(4)]
    return messages


def test_no_group_appears_in_two_splits(corpus) -> None:
    assert leaked_groups(split_by_group(corpus, seed=1)) == set()


def test_each_split_is_label_balanced(corpus) -> None:
    splits = split_by_group(corpus, seed=1)
    for split in Split:
        counts = Counter(m.label for m in splits[split])
        assert len(set(counts.values())) == 1, counts
    assert len(splits[Split.TRAIN]) == 5 * 16 * 4


def test_split_rejects_groups_with_mixed_labels(make_message) -> None:
    messages = [make_message("a", Label.FRAUD, "g", 0), make_message("b", Label.OTHER, "g", 1)]
    with pytest.raises(ValueError, match="mixes labels"):
        split_by_group(messages, seed=0)
