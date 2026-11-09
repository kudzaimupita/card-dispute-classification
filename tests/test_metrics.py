import pytest

from dispute_clf.evaluation.metrics import (
    LABELS,
    bootstrap_interval,
    macro_f1,
    paired_bootstrap_difference,
    score,
)

TRUTH = [label for label in LABELS for _ in range(20)]


def test_perfect_predictions_score_one_with_degenerate_interval() -> None:
    scores = score(TRUTH, TRUTH, n_resamples=200)
    assert scores.macro_f1.estimate == scores.macro_f1.low == scores.macro_f1.high == 1.0
    assert scores.confusion[0][0] == 20


def test_interval_contains_the_point_estimate() -> None:
    noisy = TRUTH[:]
    noisy[::7] = ["other"] * len(noisy[::7])
    interval = bootstrap_interval(TRUTH, noisy, macro_f1, n_resamples=300, seed=1)
    assert interval.low <= interval.estimate <= interval.high < 1.0


def test_paired_difference_of_identical_systems_is_zero() -> None:
    noisy = TRUTH[:]
    noisy[::5] = ["fraud"] * len(noisy[::5])
    diff = paired_bootstrap_difference(TRUTH, noisy, noisy, macro_f1, n_resamples=200)
    assert diff.estimate == diff.low == diff.high == 0.0


def test_paired_difference_favours_the_better_system() -> None:
    worse = TRUTH[:]
    worse[::3] = ["other"] * len(worse[::3])
    diff = paired_bootstrap_difference(TRUTH, TRUTH, worse, macro_f1, n_resamples=300)
    assert diff.low > 0


@pytest.mark.parametrize("confidence", [0.9, 0.99])
def test_wider_confidence_gives_wider_interval(confidence: float) -> None:
    noisy = TRUTH[:]
    noisy[::4] = ["other"] * len(noisy[::4])
    base = bootstrap_interval(TRUTH, noisy, macro_f1, n_resamples=500, confidence=0.95)
    other = bootstrap_interval(TRUTH, noisy, macro_f1, n_resamples=500, confidence=confidence)
    widths = (base.high - base.low, other.high - other.low)
    assert (widths[1] < widths[0]) if confidence < 0.95 else (widths[1] > widths[0])


def test_score_restricted_to_present_labels_ignores_absent_classes() -> None:
    truth = ["fraud", "other"] * 10
    full = score(truth, truth, n_resamples=50)
    present = score(truth, truth, labels=["fraud", "other"], n_resamples=50)
    assert present.macro_f1.estimate == 1.0
    assert full.macro_f1.estimate == 0.4  # three absent labels each count as F1 = 0
    assert set(present.per_class_f1) == {"fraud", "other"}
