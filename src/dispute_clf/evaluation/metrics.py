"""Classification metrics and bootstrap confidence intervals.

All functions take label strings so that any classifier, local or remote, can be scored the
same way. The bootstrap resamples test items; paired comparisons reuse the same resample
indices for both systems so that the interval reflects the difference, not two noises.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from dispute_clf.labels import Label

Metric = Callable[[Sequence[str], Sequence[str]], float]

LABELS: list[str] = [label.value for label in Label.ordered()]


def macro_f1(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    return float(f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0))


def accuracy(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    return float(accuracy_score(y_true, y_pred))


class Interval(BaseModel):
    model_config = ConfigDict(frozen=True)

    estimate: float
    low: float
    high: float


class Scores(BaseModel):
    model_config = ConfigDict(frozen=True)

    n: int
    accuracy: Interval
    macro_f1: Interval
    per_class_f1: dict[str, float]
    confusion: list[list[int]]
    labels: list[str]


def _resample_indices(n: int, n_resamples: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, n, size=(n_resamples, n))


def bootstrap_interval(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    metric: Metric,
    *,
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
) -> Interval:
    truth, pred = np.asarray(y_true), np.asarray(y_pred)
    samples = [
        metric(truth[idx].tolist(), pred[idx].tolist())
        for idx in _resample_indices(len(truth), n_resamples, seed)
    ]
    tail = (1 - confidence) / 2 * 100
    low, high = np.percentile(samples, [tail, 100 - tail])
    return Interval(estimate=metric(y_true, y_pred), low=float(low), high=float(high))


def paired_bootstrap_difference(
    y_true: Sequence[str],
    pred_a: Sequence[str],
    pred_b: Sequence[str],
    metric: Metric,
    *,
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
) -> Interval:
    """Interval for ``metric(a) - metric(b)`` using shared resample indices."""
    truth, a, b = np.asarray(y_true), np.asarray(pred_a), np.asarray(pred_b)
    differences = [
        metric(truth[idx].tolist(), a[idx].tolist()) - metric(truth[idx].tolist(), b[idx].tolist())
        for idx in _resample_indices(len(truth), n_resamples, seed)
    ]
    tail = (1 - confidence) / 2 * 100
    low, high = np.percentile(differences, [tail, 100 - tail])
    return Interval(
        estimate=metric(y_true, pred_a) - metric(y_true, pred_b), low=float(low), high=float(high)
    )


def score(
    y_true: Sequence[str], y_pred: Sequence[str], *, n_resamples: int = 1000, seed: int = 0
) -> Scores:
    per_class = f1_score(y_true, y_pred, labels=LABELS, average=None, zero_division=0)
    return Scores(
        n=len(y_true),
        accuracy=bootstrap_interval(y_true, y_pred, accuracy, n_resamples=n_resamples, seed=seed),
        macro_f1=bootstrap_interval(y_true, y_pred, macro_f1, n_resamples=n_resamples, seed=seed),
        per_class_f1={label: float(value) for label, value in zip(LABELS, per_class, strict=True)},
        confusion=confusion_matrix(y_true, y_pred, labels=LABELS).tolist(),
        labels=LABELS,
    )
