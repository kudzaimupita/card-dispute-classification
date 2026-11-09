from dispute_clf.attributes import AttributeGrid
from dispute_clf.generation.planner import plan_slots
from dispute_clf.generation.prompts import (
    OUTPUT_SCHEMA,
    build_system_prompt,
    build_user_prompt,
    prompt_sha256,
)
from dispute_clf.labels import Label, LabelSchema


def test_system_prompt_defines_every_label(label_schema: LabelSchema) -> None:
    prompt = build_system_prompt(label_schema)
    assert all(label.value in prompt for label in Label)


def test_user_prompt_states_target_and_hides_typo_level(
    label_schema: LabelSchema, attribute_grid: AttributeGrid
) -> None:
    slot = plan_slots(attribute_grid, slots_per_label=1, messages_per_slot=8, seed=0)[0]
    prompt = build_user_prompt(slot, label_schema, attribute_grid)
    assert slot.label.value in prompt
    assert "8 distinct messages" in prompt
    assert "typo" not in prompt.lower()


def test_output_schema_is_closed() -> None:
    assert OUTPUT_SCHEMA["additionalProperties"] is False
    assert OUTPUT_SCHEMA["required"] == ["messages"]


def test_prompt_hash_is_stable() -> None:
    assert prompt_sha256("a", "b") == prompt_sha256("a", "b") != prompt_sha256("a", "c")
