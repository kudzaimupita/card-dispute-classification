from pathlib import Path

import pytest

from dispute_clf.external import IntentMapping, map_rows
from dispute_clf.labels import Label


def _mapping(**groups: tuple[str, ...]) -> IntentMapping:
    return IntentMapping(
        dataset="local/test",
        split="test",
        mapping={Label(k): v for k, v in groups.items()},
        output=Path("unused.jsonl"),
    )


def test_map_rows_keeps_mapped_intents_only() -> None:
    mapping = _mapping(fraud=("card_payment_not_recognised",), other=("exchange_rate",))
    examples = map_rows(
        ["I don't know this payment", "What rate  do you use?", "Activate my card"],
        ["card_payment_not_recognised", "exchange_rate", "activate_my_card"],
        mapping,
    )
    assert [(e.label, e.text) for e in examples] == [
        (Label.FRAUD, "I don't know this payment"),
        (Label.OTHER, "What rate do you use?"),
    ]


def test_mapping_rejects_an_intent_under_two_labels() -> None:
    with pytest.raises(ValueError, match="more than one label"):
        _mapping(fraud=("x",), other=("x",))
