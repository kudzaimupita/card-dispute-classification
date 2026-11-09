from pathlib import Path

import pytest
import yaml

from dispute_clf.labels import Label, LabelSchema, load_label_schema


def test_schema_defines_every_label_once(label_schema: LabelSchema) -> None:
    assert [spec.id for spec in label_schema.labels] == Label.ordered()


def test_spec_lookup_returns_matching_label(label_schema: LabelSchema) -> None:
    assert label_schema.spec(Label.FRAUD).id is Label.FRAUD


def test_schema_rejects_missing_label(tmp_path: Path, label_schema: LabelSchema) -> None:
    data = label_schema.model_dump(mode="json")
    data["labels"] = data["labels"][:-1]
    path = tmp_path / "labels.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(ValueError, match="exactly once"):
        load_label_schema(path)
