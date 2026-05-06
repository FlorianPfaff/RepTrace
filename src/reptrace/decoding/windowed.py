from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.decomposition import PCA

from reptrace.decoding.classifiers import prediction_scores


@dataclass(frozen=True)
class WindowedModelBundle:
    """Fitted model plus label-independent feature transform metadata."""

    model: Any
    train_window: tuple[float, float] | None
    train_labels: np.ndarray
    pca_coeff: np.ndarray | None
    train_features_mean: np.ndarray | None
    explained_variance_percent: float
    actual_components_pca: int


@dataclass(frozen=True)
class WindowedDecodingResult:
    """Predictions and null scores for one train/validation feature window."""

    model_bundle: WindowedModelBundle
    predictions: np.ndarray
    scores: np.ndarray
    accuracy: float
    permutation_accuracy: np.ndarray
    permutation_p_value: float


FitModel = Callable[[np.ndarray, np.ndarray], Any]


def fit_window_model(
    train_features: Sequence[Sequence[float]] | np.ndarray,
    train_labels: Sequence | np.ndarray,
    *,
    fit_model: FitModel,
    components_pca: int | float = float("inf"),
    train_window: tuple[float, float] | None = None,
) -> WindowedModelBundle:
    """Fit one model for a precomputed feature window.

    Dataset-specific projects provide the windowed feature matrix and a model
    factory. RepTrace owns the reusable PCA fit/transform bookkeeping.
    """

    train_features = _feature_matrix(train_features, name="train_features")
    train_labels = _label_vector(train_labels, expected_length=train_features.shape[0], name="train_labels")
    transformed_features, pca_coeff, feature_mean, explained_variance, actual_components = _fit_pca_transform(
        train_features,
        components_pca,
    )
    model = fit_model(transformed_features, train_labels)
    return WindowedModelBundle(
        model=model,
        train_window=train_window,
        train_labels=train_labels,
        pca_coeff=pca_coeff,
        train_features_mean=feature_mean,
        explained_variance_percent=explained_variance,
        actual_components_pca=actual_components,
    )


def transform_window_features(
    model_bundle: WindowedModelBundle,
    features: Sequence[Sequence[float]] | np.ndarray,
) -> np.ndarray:
    """Apply the fitted PCA transform from a windowed model bundle."""

    features = _feature_matrix(features, name="features")
    if model_bundle.pca_coeff is None:
        return features
    if model_bundle.train_features_mean is None:
        raise ValueError("PCA coefficients require train_features_mean.")
    return (features - model_bundle.train_features_mean) @ model_bundle.pca_coeff[:, : model_bundle.actual_components_pca]


def predict_window_model(
    model_bundle: WindowedModelBundle,
    features: Sequence[Sequence[float]] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict labels and confidence-like scores for a precomputed feature window."""

    transformed_features = transform_window_features(model_bundle, features)
    predictions = np.asarray(model_bundle.model.predict(transformed_features))
    scores = prediction_scores(model_bundle.model, transformed_features)
    return predictions, scores


def score_windowed_decoding(
    train_features: Sequence[Sequence[float]] | np.ndarray,
    train_labels: Sequence | np.ndarray,
    validation_features: Sequence[Sequence[float]] | np.ndarray,
    validation_labels: Sequence | np.ndarray,
    *,
    fit_model: FitModel,
    components_pca: int | float = float("inf"),
    train_window: tuple[float, float] | None = None,
    n_permutations: int = 0,
    permutation_rng: np.random.Generator | None = None,
) -> WindowedDecodingResult:
    """Fit, predict, score accuracy, and optionally compute shuffled-label null scores."""

    train_features = _feature_matrix(train_features, name="train_features")
    train_labels = _label_vector(train_labels, expected_length=train_features.shape[0], name="train_labels")
    validation_features = _feature_matrix(validation_features, name="validation_features")
    validation_labels = _label_vector(
        validation_labels,
        expected_length=validation_features.shape[0],
        name="validation_labels",
    )
    model_bundle = fit_window_model(
        train_features,
        train_labels,
        fit_model=fit_model,
        components_pca=components_pca,
        train_window=train_window,
    )
    predictions, scores = predict_window_model(model_bundle, validation_features)
    accuracy = float(np.mean(predictions == validation_labels)) if len(validation_labels) else np.nan

    permutation_accuracy = np.array([], dtype=float)
    permutation_p_value = np.nan
    if n_permutations > 0:
        transformed_train = transform_window_features(model_bundle, train_features)
        transformed_validation = transform_window_features(model_bundle, validation_features)
        permutation_accuracy = permutation_accuracy_curve(
            transformed_train,
            validation_features=transformed_validation,
            validation_labels=validation_labels,
            train_labels=train_labels,
            fit_model=fit_model,
            n_permutations=n_permutations,
            permutation_rng=permutation_rng,
        )
        permutation_p_value = permutation_p_from_accuracy(
            accuracy,
            permutation_accuracy,
        )

    return WindowedDecodingResult(
        model_bundle=model_bundle,
        predictions=predictions,
        scores=scores,
        accuracy=accuracy,
        permutation_accuracy=permutation_accuracy,
        permutation_p_value=permutation_p_value,
    )


def permutation_accuracy_curve(
    train_features: Sequence[Sequence[float]] | np.ndarray,
    *,
    validation_features: Sequence[Sequence[float]] | np.ndarray,
    validation_labels: Sequence | np.ndarray,
    train_labels: Sequence | np.ndarray,
    fit_model: FitModel,
    n_permutations: int,
    permutation_rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Train shuffled-label models and return validation accuracies."""

    if n_permutations < 0:
        raise ValueError("n_permutations must be non-negative.")
    train_features = _feature_matrix(train_features, name="train_features")
    validation_features = _feature_matrix(validation_features, name="validation_features")
    train_labels = _label_vector(train_labels, expected_length=train_features.shape[0], name="train_labels")
    validation_labels = _label_vector(
        validation_labels,
        expected_length=validation_features.shape[0],
        name="validation_labels",
    )
    if permutation_rng is None:
        permutation_rng = np.random.default_rng()

    permuted_scores = []
    for _ in range(int(n_permutations)):
        permuted_train_labels = np.array(train_labels, copy=True)
        permutation_rng.shuffle(permuted_train_labels)
        model = fit_model(train_features, permuted_train_labels)
        predictions = np.asarray(model.predict(validation_features))
        permuted_scores.append(float(np.mean(predictions == validation_labels)))
    return np.asarray(permuted_scores, dtype=float)


def permutation_p_from_accuracy(accuracy: float, permutation_accuracy: Sequence[float] | np.ndarray) -> float:
    """Return the one-sided permutation p-value with plus-one correction."""

    permutation_accuracy = np.asarray(permutation_accuracy, dtype=float)
    if permutation_accuracy.size == 0 or not np.isfinite(accuracy):
        return np.nan
    return float((np.sum(permutation_accuracy >= accuracy) + 1.0) / (permutation_accuracy.size + 1.0))


def _fit_pca_transform(
    features: np.ndarray,
    components_pca: int | float,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, float, int]:
    actual_components = _actual_pca_components(components_pca, features)
    if components_pca == float("inf"):
        return features, None, None, np.nan, actual_components

    feature_mean = np.mean(features, axis=0)
    centered = features - feature_mean
    pca = PCA(n_components=actual_components)
    transformed = pca.fit_transform(centered)
    explained_variance = float(np.sum(pca.explained_variance_ratio_) * 100.0)
    return transformed, pca.components_.T, feature_mean, explained_variance, actual_components


def _actual_pca_components(components_pca: int | float, features: np.ndarray) -> int:
    if components_pca == float("inf"):
        return int(features.shape[1])
    return min(int(components_pca), int(features.shape[0]), int(features.shape[1]))


def _feature_matrix(features: Sequence[Sequence[float]] | np.ndarray, *, name: str) -> np.ndarray:
    matrix = np.asarray(features, dtype=float)
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional feature matrix.")
    if matrix.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one row.")
    return matrix


def _label_vector(labels: Sequence | np.ndarray, *, expected_length: int, name: str) -> np.ndarray:
    vector = np.asarray(labels).ravel()
    if vector.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    if len(vector) != expected_length:
        raise ValueError(f"{name} length must match feature rows: {len(vector)} != {expected_length}.")
    return vector
