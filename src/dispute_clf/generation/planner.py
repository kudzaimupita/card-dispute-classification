"""Plan generation slots: one slot is one request for several messages of one label."""

from __future__ import annotations

import random

from pydantic import BaseModel, ConfigDict

from dispute_clf.attributes import AttributeGrid, Attributes
from dispute_clf.labels import Label


class GenerationSlot(BaseModel):
    model_config = ConfigDict(frozen=True)

    slot_id: str
    label: Label
    attributes: Attributes
    n_messages: int
    seed: int


def plan_slots(
    grid: AttributeGrid,
    *,
    slots_per_label: int,
    messages_per_slot: int,
    seed: int,
    labels: list[Label] | None = None,
) -> list[GenerationSlot]:
    """Return a label-balanced, deterministic list of slots.

    Slot ids are valid batch ``custom_id`` values and double as split groups.
    """
    rng = random.Random(seed)
    slots: list[GenerationSlot] = []
    for label in labels or Label.ordered():
        for index in range(slots_per_label):
            slots.append(
                GenerationSlot(
                    slot_id=f"{label.value}-{index:04d}",
                    label=label,
                    attributes=grid.sample(rng),
                    n_messages=messages_per_slot,
                    seed=rng.randrange(2**31),
                )
            )
    return slots
