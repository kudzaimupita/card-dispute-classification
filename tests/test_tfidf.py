from dispute_clf.baselines.tfidf import fit, predict
from dispute_clf.labels import Label

PHRASES = {
    Label.FRAUD: "I do not recognise this payment, someone used my card",
    Label.GOODS_NOT_RECEIVED: "my order never arrived and the shop ignores me",
    Label.DUPLICATE_CHARGE: "the same payment was taken twice from my account",
    Label.SUBSCRIPTION_CANCELLATION: "they keep billing me after I cancelled the membership",
    Label.OTHER: "why was my card declined at the till today",
}


def test_tfidf_learns_separable_toy_data(make_message) -> None:
    train = [
        make_message(f"{PHRASES[label]} {variant}", label, f"{label.value}-{group}", variant)
        for label in Label
        for group in range(6)
        for variant in range(3)
    ]
    result = fit(train, c_grid=(1.0, 10.0), n_splits=3, min_df=1)

    assert result.best_c in {1.0, 10.0}
    assert set(result.cv_macro_f1) == {1.0, 10.0}
    test = [make_message(PHRASES[label], label, "t", i) for i, label in enumerate(Label)]
    assert predict(result.pipeline, test) == [label.value for label in Label]
