import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from anthropic.types.messages import MessageBatchIndividualResponse

from dispute_clf.attributes import AttributeGrid
from dispute_clf.generation.batch import BatchRecord, build_requests, collect, parse_payload
from dispute_clf.generation.config import load_generation_config
from dispute_clf.generation.planner import GenerationSlot, plan_slots
from dispute_clf.generation.prompts import build_system_prompt, build_user_prompt
from dispute_clf.labels import LabelSchema

ROOT = Path(__file__).resolve().parents[1]
BATCH = BatchRecord(
    batch_id="msgbatch_test",
    model="claude-sonnet-5-5",
    n_requests=3,
    submitted_at=datetime(2026, 11, 10, tzinfo=UTC),
)


def _succeeded(
    custom_id: str, text: str, stop_reason: str = "end_turn"
) -> MessageBatchIndividualResponse:
    return MessageBatchIndividualResponse.model_validate(
        {
            "custom_id": custom_id,
            "result": {
                "type": "succeeded",
                "message": {
                    "id": "msg_1",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-sonnet-5-5",
                    "content": [{"type": "text", "text": text}],
                    "stop_reason": stop_reason,
                    "stop_sequence": None,
                    "usage": {"input_tokens": 10, "output_tokens": 20},
                },
            },
        }
    )


@pytest.fixture
def slots(attribute_grid: AttributeGrid) -> list[GenerationSlot]:
    return plan_slots(attribute_grid, slots_per_label=1, messages_per_slot=2, seed=0)


def test_requests_carry_model_effort_and_schema(
    slots: list[GenerationSlot], label_schema: LabelSchema, attribute_grid: AttributeGrid
) -> None:
    config = load_generation_config(ROOT / "configs" / "generation.yaml")
    requests = build_requests(slots, config, label_schema, attribute_grid)

    assert [r["custom_id"] for r in requests] == [s.slot_id for s in slots]
    params = requests[0]["params"]
    assert params["model"] == config.model
    assert params["output_config"]["effort"] == config.effort
    assert params["output_config"]["format"]["type"] == "json_schema"
    assert "temperature" not in params


def test_parse_payload_normalises_whitespace_and_drops_empty() -> None:
    payload = json.dumps({"messages": ["  charged   twice ", "", 3, "where is\nmy parcel"]})
    assert parse_payload(payload) == ["charged twice", "where is my parcel"]


def test_parse_payload_rejects_wrong_shape() -> None:
    with pytest.raises(ValueError, match="messages"):
        parse_payload(json.dumps({"items": []}))


def test_collect_keeps_successes_and_reports_failures(
    slots: list[GenerationSlot], label_schema: LabelSchema, attribute_grid: AttributeGrid
) -> None:
    system = build_system_prompt(label_schema)
    users = {s.slot_id: build_user_prompt(s, label_schema, attribute_grid) for s in slots}
    ok, refused, broken = slots[0], slots[1], slots[2]
    results = [
        _succeeded(ok.slot_id, json.dumps({"messages": ["one", "two", "three"]})),
        _succeeded(refused.slot_id, "", stop_reason="refusal"),
        _succeeded(broken.slot_id, "not json"),
    ]

    outcome = collect(
        results,
        {s.slot_id: s for s in slots},
        batch=BATCH,
        system_prompt=system,
        user_prompts=users,
    )

    assert [m.id for m in outcome.messages] == [f"{ok.slot_id}-0", f"{ok.slot_id}-1"]
    assert all(m.label is ok.label and m.group == ok.slot_id for m in outcome.messages)
    assert {(f.slot_id, f.reason) for f in outcome.failures} == {
        (refused.slot_id, "refusal"),
        (broken.slot_id, "invalid_payload"),
    }
