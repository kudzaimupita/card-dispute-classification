"""TF-IDF with logistic regression.

Word unigrams and bigrams capture vocabulary; character n-grams within word boundaries
make the model tolerant of typos and morphological variants. The regularisation strength
is selected by grouped, stratified cross-validation on the training split only, so that
messages from one generation request never sit in both a training and a validation fold.
"""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import FeatureUnion, Pipeline

from dispute_clf.records import Message

DEFAULT_C_GRID: tuple[float, ...] = (0.3, 1.0, 3.0, 10.0)


class TfidfFit(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    pipeline: Pipeline
    best_c: float
    cv_macro_f1: dict[float, float]


def build_pipeline(c: float = 1.0, *, min_df: int = 2, seed: int = 0) -> Pipeline:
    features = FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    analyzer="word", ngram_range=(1, 2), min_df=min_df, sublinear_tf=True
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb", ngram_range=(3, 5), min_df=min_df, sublinear_tf=True
                ),
            ),
        ]
    )
    classifier = LogisticRegression(C=c, class_weight="balanced", max_iter=2000, random_state=seed)
    return Pipeline([("features", features), ("classifier", classifier)])


def fit(
    train: Sequence[Message],
    *,
    c_grid: Sequence[float] = DEFAULT_C_GRID,
    n_splits: int = 5,
    min_df: int = 2,
    seed: int = 0,
) -> TfidfFit:
    texts = [message.text for message in train]
    labels = [message.label.value for message in train]
    groups = [message.group for message in train]

    search = GridSearchCV(
        build_pipeline(min_df=min_df, seed=seed),
        param_grid={"classifier__C": list(c_grid)},
        scoring="f1_macro",
        cv=StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed),
        refit=True,
    )
    search.fit(texts, labels, groups=groups)

    cv_scores = {
        float(params["classifier__C"]): float(mean)
        for params, mean in zip(
            search.cv_results_["params"], search.cv_results_["mean_test_score"], strict=True
        )
    }
    return TfidfFit(
        pipeline=search.best_estimator_,
        best_c=float(search.best_params_["classifier__C"]),
        cv_macro_f1=cv_scores,
    )


def predict(pipeline: Pipeline, messages: Sequence[Message]) -> list[str]:
    return [str(label) for label in pipeline.predict([message.text for message in messages])]
