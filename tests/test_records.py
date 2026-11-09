from pathlib import Path

from dispute_clf.labels import Label
from dispute_clf.records import Message, read_jsonl, write_jsonl


def test_jsonl_round_trip(tmp_path: Path, make_message) -> None:
    messages = [make_message("first", Label.FRAUD, idx=0), make_message("second", idx=1)]
    path = tmp_path / "nested" / "messages.jsonl"

    assert write_jsonl(path, messages) == 2
    assert read_jsonl(path, Message) == messages
