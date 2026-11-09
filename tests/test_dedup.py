import pytest

from dispute_clf.preprocessing.dedup import (
    cross_split_overlap,
    deduplicate,
    drop_exact,
    drop_near_duplicates,
    normalise,
)


def test_normalise_ignores_case_punctuation_and_spacing() -> None:
    assert normalise("  Charged TWICE!!  for   my order. ") == "charged twice for my order"


def test_exact_stage_keeps_first_by_id(make_message) -> None:
    messages = [
        make_message("Charged twice!", idx=1),
        make_message("charged twice", idx=0),
        make_message("Where is my parcel?", idx=2),
    ]
    result = drop_exact(messages)
    assert [m.id for m in result.kept] == ["s0-0", "s0-2"]
    assert result.removed[0].duplicate_of == "s0-0"


def test_near_stage_removes_small_edits(make_message) -> None:
    base = "I was charged twice for the same hotel booking in Lisbon last weekend"
    messages = [
        make_message(base, idx=0),
        make_message(base + " please", idx=1),
        make_message("My parcel from the electronics shop never arrived at all", idx=2),
    ]
    result = drop_near_duplicates(messages, threshold=0.8)
    assert [m.id for m in result.kept] == ["s0-0", "s0-2"]
    assert result.count("near") == 1


def test_deduplicate_without_semantic_stage(make_message) -> None:
    messages = [make_message("same text", idx=0), make_message("Same text.", idx=1)]
    result = deduplicate(messages, semantic_threshold=None)
    assert len(result.kept) == 1 and result.count("exact") == 1


@pytest.mark.semantic
def test_semantic_overlap_between_splits(make_message) -> None:
    train = [make_message("my card was charged twice for one coffee", idx=0)]
    test = [
        make_message("my card was charged twice for one coffee!", group="s1", idx=0),
        make_message("the airline cancelled my flight and kept the money", group="s1", idx=1),
    ]
    assert cross_split_overlap(train, test) == ["s1-0"]
