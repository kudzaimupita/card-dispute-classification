"""Prompts and output schema for attribute-controlled message generation."""

from __future__ import annotations

import hashlib
from typing import Any

from dispute_clf.attributes import AttributeGrid
from dispute_clf.generation.planner import GenerationSlot
from dispute_clf.labels import LabelSchema

OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "messages": {
            "type": "array",
            "items": {"type": "string"},
            "description": "The customer messages, one string per message.",
        }
    },
    "required": ["messages"],
    "additionalProperties": False,
}


def _describe_labels(schema: LabelSchema) -> str:
    blocks = []
    for spec in schema.labels:
        lines = [f"{spec.id.value} ({spec.name}): {spec.definition}"]
        lines += [f"  includes: {item}" for item in spec.includes]
        lines += [f"  excludes: {item}" for item in spec.excludes]
        blocks.append("\n".join(lines))
    rules = "\n".join(f"- {rule}" for rule in schema.boundary_rules)
    return "\n\n".join(blocks) + f"\n\nBoundary rules:\n{rules}"


def build_system_prompt(schema: LabelSchema) -> str:
    return (
        "You write realistic customer messages for a research dataset on classifying "
        "card-payment support requests. The data is synthetic: invent every name, merchant, "
        "date, reference and amount, and never reproduce real personal data.\n\n"
        "The dataset has five categories:\n\n"
        f"{_describe_labels(schema)}\n\n"
        "Each request asks for messages in one target category. Every message must belong to "
        "that category under the definitions and boundary rules, even when it is phrased "
        "indirectly or mentions other issues; read as if a real customer wrote it; and differ "
        "from the others in situation and wording. Do not name the category. Write with "
        "correct spelling; typing errors are added separately."
    )


def build_user_prompt(slot: GenerationSlot, schema: LabelSchema, grid: AttributeGrid) -> str:
    spec = schema.spec(slot.label)
    attrs = slot.attributes
    return (
        f"Write {slot.n_messages} distinct messages in the category "
        f"{spec.id.value} ({spec.name}).\n\n"
        "Customer profile for these messages:\n"
        f"- Persona: {attrs.persona}\n"
        f"- Channel: {attrs.channel}\n"
        f"- Length: {grid.describe_length(attrs.length)}\n"
        f"- Variety of English: {attrs.english_variety}\n"
        f"- Currency for any amounts: {attrs.currency}\n"
        f"- Type of merchant involved: {attrs.merchant_type}\n"
        f"- Tone: {attrs.tone}\n\n"
        f"How the problem is expressed: {grid.describe_difficulty(attrs.difficulty)}\n\n"
        f"Vary the situation, the details given and the sentence structure across the "
        f"{slot.n_messages} messages."
    )


def prompt_sha256(system: str, user: str) -> str:
    return hashlib.sha256(f"{system}\n\n{user}".encode()).hexdigest()
