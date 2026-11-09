from datetime import UTC, datetime
from pathlib import Path

import pytest

from dispute_clf.attributes import AttributeGrid, Attributes, load_attribute_grid
from dispute_clf.labels import Label, LabelSchema, load_label_schema
from dispute_clf.noise import TypoLevel
from dispute_clf.records import Message, Provenance

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


@pytest.fixture(scope="session")
def label_schema() -> LabelSchema:
    return load_label_schema(CONFIGS / "labels.yaml")


@pytest.fixture(scope="session")
def attribute_grid() -> AttributeGrid:
    return load_attribute_grid(CONFIGS / "attributes.yaml")


@pytest.fixture
def attributes() -> Attributes:
    return Attributes(
        persona="retired pensioner",
        channel="email to the bank",
        length="short",
        english_variety="British English",
        currency="GBP",
        merchant_type="airline",
        tone="calm and polite",
        typo_level=TypoLevel.NONE,
    )


@pytest.fixture
def make_message(attributes: Attributes):
    def factory(text: str, label: Label = Label.OTHER, group: str = "s0", idx: int = 0) -> Message:
        return Message(
            id=f"{group}-{idx}",
            text=text,
            label=label,
            group=group,
            attributes=attributes,
            provenance=Provenance(
                model="test-model",
                batch_id="batch-test",
                slot_id=group,
                prompt_sha256="0" * 64,
                generated_at=datetime(2026, 11, 10, tzinfo=UTC),
            ),
        )

    return factory
