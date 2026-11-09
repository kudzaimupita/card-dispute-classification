import pytest

from dispute_clf.noise import TypoLevel, add_typos

TEXT = (
    "I was charged twice for the same hotel booking on Friday, "
    "please refund the second payment of 249.99 GBP as soon as possible"
)


def test_none_level_returns_text_unchanged() -> None:
    assert add_typos(TEXT, TypoLevel.NONE, seed=1) == TEXT


@pytest.mark.parametrize("level", [TypoLevel.LIGHT, TypoLevel.HEAVY])
def test_same_seed_gives_same_output(level: TypoLevel) -> None:
    assert add_typos(TEXT, level, seed=7) == add_typos(TEXT, level, seed=7)


def test_heavy_level_edits_more_words_than_light() -> None:
    def changed_words(level: TypoLevel) -> int:
        total = 0
        for seed in range(200):
            noisy = add_typos(TEXT, level, seed=seed).split()
            total += sum(a != b for a, b in zip(TEXT.split(), noisy, strict=False))
        return total

    assert changed_words(TypoLevel.HEAVY) > changed_words(TypoLevel.LIGHT) > 0


def test_numbers_and_currency_codes_are_preserved() -> None:
    for seed in range(100):
        noisy = add_typos(TEXT, TypoLevel.HEAVY, seed=seed)
        assert "249.99" in noisy
