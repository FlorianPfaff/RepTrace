from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

DEFAULT_CLASSIFIER_PARAMS = {
    "lasso": 0.005,
    "multiclass-svm": 0.5,
    "multiclass-svm-weighted": 0.5,
    "svm-binary": 0.5,
    "binary-svm": 0.5,
    "random-forest": 100,
    "gradient-boosting": 100,
    "knn": 5,
    "mostFrequentDummy": None,
    "always1Dummy": None,
    "scikit-mlp": (150, 1000),
}


@dataclass(frozen=True)
class ClassifierSpec:
    """Factory metadata for classifiers that may or may not fit in builder."""

    builder: Callable[[np.ndarray, np.ndarray, Any, int | None], Any]
    fits_in_builder: bool = False


def should_use_default_classifier_param(classifier_param: Any) -> bool:
    """Return true for legacy NaN placeholders that request default params."""

    try:
        return bool(np.all(np.isnan(classifier_param)))
    except TypeError:
        return False


def get_default_classifier_param(classifier: str) -> Any:
    """Return a defensive copy of the configured default classifier parameter."""

    if classifier in DEFAULT_CLASSIFIER_PARAMS:
        classifier_param = DEFAULT_CLASSIFIER_PARAMS[classifier]
        if isinstance(classifier_param, dict):
            return classifier_param.copy()
        return classifier_param
    raise ValueError(f"Unsupported classifier: {classifier}")


def _build_multiclass_svm(_features: np.ndarray, _labels: np.ndarray, classifier_param: Any, random_state: int | None):
    return make_pipeline(
        StandardScaler(),
        SVC(C=classifier_param, kernel="linear", random_state=random_state),
    )


def _build_multiclass_svm_weighted(_features: np.ndarray, _labels: np.ndarray, classifier_param: Any, random_state: int | None):
    return make_pipeline(
        StandardScaler(),
        SVC(
            C=classifier_param,
            kernel="linear",
            class_weight="balanced",
            random_state=random_state,
        ),
    )


def _build_random_forest(_features: np.ndarray, _labels: np.ndarray, classifier_param: Any, random_state: int | None):
    return RandomForestClassifier(
        n_estimators=int(classifier_param),
        min_samples_leaf=5,
        random_state=random_state,
    )


def _build_gradient_boosting(_features: np.ndarray, _labels: np.ndarray, classifier_param: Any, random_state: int | None):
    return GradientBoostingClassifier(
        n_estimators=int(classifier_param),
        random_state=random_state,
    )


def _build_knn(_features: np.ndarray, _labels: np.ndarray, classifier_param: Any, _random_state: int | None):
    return KNeighborsClassifier(n_neighbors=int(classifier_param))


def _build_most_frequent_dummy(_features: np.ndarray, _labels: np.ndarray, _classifier_param: Any, _random_state: int | None):
    return DummyClassifier(strategy="most_frequent")


def _build_always_one_dummy(_features: np.ndarray, _labels: np.ndarray, _classifier_param: Any, _random_state: int | None):
    return DummyClassifier(strategy="constant", constant=1)


def _build_scikit_mlp(_features: np.ndarray, _labels: np.ndarray, classifier_param: Any, random_state: int | None):
    return MLPClassifier(
        hidden_layer_sizes=int(classifier_param[0]),
        max_iter=int(classifier_param[1]),
        random_state=random_state,
    )


CLASSIFIER_REGISTRY = {
    "multiclass-svm": ClassifierSpec(_build_multiclass_svm),
    "multiclass-svm-weighted": ClassifierSpec(_build_multiclass_svm_weighted),
    "random-forest": ClassifierSpec(_build_random_forest),
    "gradient-boosting": ClassifierSpec(_build_gradient_boosting),
    "knn": ClassifierSpec(_build_knn),
    "mostFrequentDummy": ClassifierSpec(_build_most_frequent_dummy),
    "always1Dummy": ClassifierSpec(_build_always_one_dummy),
    "scikit-mlp": ClassifierSpec(_build_scikit_mlp),
}


def train_classifier(
    features: Sequence[Sequence[float]] | np.ndarray,
    labels: Sequence | np.ndarray,
    classifier: str,
    classifier_param: Any,
    random_state: int | None = None,
    *,
    registry: dict[str, ClassifierSpec] | None = None,
):
    """Build and fit a classifier from a registry entry."""

    registry = CLASSIFIER_REGISTRY if registry is None else registry
    features = np.asarray(features)
    labels = np.asarray(labels).ravel()
    try:
        classifier_spec = registry[classifier]
    except KeyError as exc:
        supported_classifiers = ", ".join(sorted(registry))
        raise ValueError(f"Unsupported classifier: {classifier}. Supported classifiers: {supported_classifiers}") from exc

    model = classifier_spec.builder(features, labels, classifier_param, random_state)
    if classifier_spec.fits_in_builder:
        return model
    model.fit(features, labels)
    return model


def train_multiclass_classifier(
    features: Sequence[Sequence[float]] | np.ndarray,
    labels: Sequence | np.ndarray,
    classifier: str,
    classifier_param: Any,
    random_state: int | None = None,
):
    """Backward-compatible name for registry-based classifier training."""

    return train_classifier(features, labels, classifier, classifier_param, random_state=random_state)


def train_gradient_boosting(
    train_features: Sequence[Sequence[float]] | np.ndarray,
    train_labels: Sequence | np.ndarray,
    classifier_param: Any,
    random_state: int | None = None,
):
    """Train the legacy binary gradient boosting helper used by one-vs-rest decoding."""

    model = GradientBoostingClassifier(
        n_estimators=int(classifier_param),
        max_leaf_nodes=21,
        learning_rate=0.1,
        random_state=random_state,
    )
    model.fit(train_features, train_labels)
    return model


def train_lasso_logistic(
    train_features: Sequence[Sequence[float]] | np.ndarray,
    train_labels: Sequence | np.ndarray,
    lambda_: float,
    random_state: int | None = None,
):
    """Train an L1-regularized logistic model for binary one-vs-rest decoding."""

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            penalty="l1",
            C=1 / lambda_,
            solver="liblinear",
            max_iter=1000,
            random_state=random_state,
        ),
    )
    model.fit(train_features, train_labels)
    return model


def train_binary_svm(
    train_features: Sequence[Sequence[float]] | np.ndarray,
    train_labels: Sequence | np.ndarray,
    box_constraint: float,
    random_state: int | None = None,
):
    """Train a linear binary SVM helper for one-vs-rest decoding."""

    model = make_pipeline(
        StandardScaler(),
        SVC(C=box_constraint, kernel="linear", random_state=random_state),
    )
    model.fit(train_features, train_labels)
    return model


def prediction_scores(model: Any, features: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
    """Return one confidence-like score per row from common classifier APIs."""

    features = np.asarray(features, dtype=float)
    if features.ndim != 2:
        raise ValueError("features must be a two-dimensional feature matrix.")
    if hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(features), dtype=float)
        if scores.ndim == 1:
            return np.abs(scores)
        return np.max(scores, axis=1)
    if hasattr(model, "predict_proba"):
        scores = np.asarray(model.predict_proba(features), dtype=float)
        return np.max(scores, axis=1)
    return np.full(features.shape[0], np.nan, dtype=float)
