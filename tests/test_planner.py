from collections import Counter

from dispute_clf.attributes import AttributeGrid
from dispute_clf.generation.planner import plan_slots
from dispute_clf.labels import Label


def test_slots_are_balanced_across_labels(attribute_grid: AttributeGrid) -> None:
    slots = plan_slots(attribute_grid, slots_per_label=4, messages_per_slot=8, seed=1)
    assert Counter(slot.label for slot in slots) == dict.fromkeys(Label, 4)


def test_plan_is_deterministic(attribute_grid: AttributeGrid) -> None:
    first = plan_slots(attribute_grid, slots_per_label=3, messages_per_slot=8, seed=9)
    second = plan_slots(attribute_grid, slots_per_label=3, messages_per_slot=8, seed=9)
    assert first == second


def test_slot_ids_are_unique_and_valid_custom_ids(attribute_grid: AttributeGrid) -> None:
    slots = plan_slots(attribute_grid, slots_per_label=12, messages_per_slot=8, seed=2)
    ids = [slot.slot_id for slot in slots]
    assert len(ids) == len(set(ids))
    assert all(len(i) <= 64 and i.replace("_", "").replace("-", "").isalnum() for i in ids)
