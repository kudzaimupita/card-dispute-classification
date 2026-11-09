"""End-to-end run of the offline pipeline stages on a small generated corpus."""

import json
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from dispute_clf.cli import app
from dispute_clf.labels import Label
from dispute_clf.records import write_jsonl

VOCAB = {
    Label.FRAUD: ["unrecognised", "stolen", "someone", "used", "card", "details"],
    Label.GOODS_NOT_RECEIVED: ["order", "never", "arrived", "delivery", "parcel", "missing"],
    Label.DUPLICATE_CHARGE: ["charged", "twice", "same", "payment", "double", "again"],
    Label.SUBSCRIPTION_CANCELLATION: ["cancelled", "subscription", "still", "billing", "trial"],
    Label.OTHER: ["declined", "pin", "fees", "exchange", "rate", "statement"],
}


def _text(label: Label, group: int, variant: int) -> str:
    start = (group + variant) % 4
    words = " ".join(VOCAB[label][start : start + 3])
    return f"{words} reference {group}{variant}x{label.value}"


@pytest.mark.semantic
def test_clean_split_and_train(tmp_path: Path, make_message) -> None:
    messages = [
        make_message(
            _text(label, g, k),
            label,
            f"{label.value}-{g:04d}",
            k,
        )
        for label in Label
        for g in range(20)
        for k in range(3)
    ]
    write_jsonl(tmp_path / "raw" / "generated.jsonl", messages)

    pre = {
        "near_threshold": 0.95,
        "semantic_threshold": 0.99,
        "noise_seed": 1,
        "split_seed": 2,
        "val_fraction": 0.2,
        "test_fraction": 0.2,
        "paths": {
            "generated": str(tmp_path / "raw" / "generated.jsonl"),
            "clean": str(tmp_path / "processed" / "clean.jsonl"),
            "dedup_report": str(tmp_path / "processed" / "dedup_report.json"),
            "splits_dir": str(tmp_path / "processed"),
            "split_report": str(tmp_path / "processed" / "split_report.json"),
        },
    }
    tfidf = {
        "c_grid": [1.0],
        "cv_folds": 3,
        "min_df": 1,
        "seed": 0,
        "bootstrap_resamples": 50,
        "paths": {"splits_dir": str(tmp_path / "processed"), "output_dir": str(tmp_path / "out")},
    }
    (tmp_path / "pre.yaml").write_text(yaml.safe_dump(pre), encoding="utf-8")
    (tmp_path / "tfidf.yaml").write_text(yaml.safe_dump(tfidf), encoding="utf-8")

    runner = CliRunner()
    for args in (
        ["clean", "--config", str(tmp_path / "pre.yaml")],
        ["split", "--config", str(tmp_path / "pre.yaml")],
        ["train-tfidf", "--config", str(tmp_path / "tfidf.yaml")],
    ):
        result = runner.invoke(app, args)
        assert result.exit_code == 0, result.output

    report = json.loads((tmp_path / "out" / "metrics_val.json").read_text(encoding="utf-8"))
    assert report["system"] == "tfidf_logreg"
    assert 0.0 <= report["scores"]["macro_f1"]["estimate"] <= 1.0
