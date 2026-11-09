"""Build, submit and collect Message Batches for corpus generation.

Batches run asynchronously at half the standard token price. Results arrive in any order
and are keyed by ``custom_id``, which is the slot id.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from datetime import datetime

import anthropic
from anthropic.types import Message as ApiMessage
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages import MessageBatchIndividualResponse
from anthropic.types.messages.batch_create_params import Request
from pydantic import BaseModel, ConfigDict

from dispute_clf.attributes import AttributeGrid
from dispute_clf.generation.config import GenerationConfig
from dispute_clf.generation.planner import GenerationSlot
from dispute_clf.generation.prompts import (
    OUTPUT_SCHEMA,
    build_system_prompt,
    build_user_prompt,
    prompt_sha256,
)
from dispute_clf.labels import LabelSchema
from dispute_clf.records import Message, Provenance


class BatchRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    batch_id: str
    model: str
    n_requests: int
    submitted_at: datetime


class GenerationFailure(BaseModel):
    model_config = ConfigDict(frozen=True)

    slot_id: str
    reason: str
    detail: str = ""


class GenerationOutcome(BaseModel):
    messages: list[Message]
    failures: list[GenerationFailure]


def build_requests(
    slots: Iterable[GenerationSlot],
    config: GenerationConfig,
    schema: LabelSchema,
    grid: AttributeGrid,
) -> list[Request]:
    system = build_system_prompt(schema)
    return [
        Request(
            custom_id=slot.slot_id,
            params=MessageCreateParamsNonStreaming(
                model=config.model,
                max_tokens=config.max_tokens,
                system=system,
                messages=[{"role": "user", "content": build_user_prompt(slot, schema, grid)}],
                output_config={
                    "effort": config.effort,
                    "format": {"type": "json_schema", "schema": OUTPUT_SCHEMA},
                },
            ),
        )
        for slot in slots
    ]


def submit(
    client: anthropic.Anthropic, requests: list[Request], submitted_at: datetime
) -> BatchRecord:
    batch = client.messages.batches.create(requests=requests)
    model = requests[0]["params"]["model"] if requests else ""
    return BatchRecord(
        batch_id=batch.id, model=model, n_requests=len(requests), submitted_at=submitted_at
    )


def parse_payload(text: str) -> list[str]:
    """Extract non-empty, whitespace-normalised messages from a structured-output payload."""
    data = json.loads(text)
    items = data.get("messages") if isinstance(data, dict) else None
    if not isinstance(items, list):
        raise ValueError("payload has no 'messages' list")
    return [" ".join(item.split()) for item in items if isinstance(item, str) and item.strip()]


def _text_of(message: ApiMessage) -> str:
    return next((block.text for block in message.content if block.type == "text"), "")


def collect(
    results: Iterable[MessageBatchIndividualResponse],
    slots: dict[str, GenerationSlot],
    *,
    batch: BatchRecord,
    system_prompt: str,
    user_prompts: dict[str, str],
) -> GenerationOutcome:
    """Turn batch results into labelled messages and an explicit list of failures."""
    messages: list[Message] = []
    failures: list[GenerationFailure] = []

    for item in results:
        slot = slots.get(item.custom_id)
        if slot is None:
            failures.append(GenerationFailure(slot_id=item.custom_id, reason="unknown_slot"))
            continue

        result = item.result
        if result.type != "succeeded":
            failures.append(GenerationFailure(slot_id=slot.slot_id, reason=result.type))
            continue

        response = result.message
        if response.stop_reason in {"refusal", "max_tokens"}:
            failures.append(
                GenerationFailure(slot_id=slot.slot_id, reason=str(response.stop_reason))
            )
            continue

        try:
            texts = parse_payload(_text_of(response))
        except (json.JSONDecodeError, ValueError) as error:
            failures.append(
                GenerationFailure(slot_id=slot.slot_id, reason="invalid_payload", detail=str(error))
            )
            continue

        provenance = Provenance(
            model=response.model,
            batch_id=batch.batch_id,
            slot_id=slot.slot_id,
            prompt_sha256=prompt_sha256(system_prompt, user_prompts[slot.slot_id]),
            generated_at=batch.submitted_at,
        )
        messages.extend(
            Message(
                id=f"{slot.slot_id}-{index}",
                text=text,
                label=slot.label,
                group=slot.slot_id,
                attributes=slot.attributes,
                provenance=provenance,
            )
            for index, text in enumerate(texts[: slot.n_messages])
        )

    messages.sort(key=lambda message: message.id)
    failures.sort(key=lambda failure: failure.slot_id)
    return GenerationOutcome(messages=messages, failures=failures)
