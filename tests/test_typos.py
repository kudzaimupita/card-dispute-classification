from dispute_clf.noise import TypoLevel
from dispute_clf.preprocessing.typos import apply_typos


def test_apply_typos_keeps_clean_text_and_is_deterministic(make_message) -> None:
    message = make_message("Please refund the duplicate payment for my gym membership")
    heavy = message.model_copy(
        update={"attributes": message.attributes.model_copy(update={"typo_level": TypoLevel.HEAVY})}
    )
    first = apply_typos([heavy, message], seed=3)
    second = apply_typos([heavy, message], seed=3)

    assert first == second
    assert first[0].text_clean == heavy.text
    assert first[1].text == message.text  # level "none" leaves text unchanged
