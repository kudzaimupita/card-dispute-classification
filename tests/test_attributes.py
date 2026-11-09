import random

from dispute_clf.attributes import AttributeGrid


def test_sampling_is_deterministic_for_a_seed(attribute_grid: AttributeGrid) -> None:
    first = [attribute_grid.sample(random.Random(3)) for _ in range(5)]
    second = [attribute_grid.sample(random.Random(3)) for _ in range(5)]
    assert first == second


def test_sampled_values_come_from_the_grid(attribute_grid: AttributeGrid) -> None:
    rng = random.Random(0)
    for _ in range(50):
        attrs = attribute_grid.sample(rng)
        assert attrs.persona in attribute_grid.persona
        assert attrs.length in attribute_grid.length
        assert attrs.typo_level in attribute_grid.typo_level
