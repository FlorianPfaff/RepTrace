from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from reptrace.decoding.classifiers import (
    positive_class_score,
    train_binary_svm,
    train_classifier,
    train_gradient_boosting,
    train_lasso_logistic,
)
from reptrace.decoding.windowed import (
    WindowedDecodingResult,
    fit_window_model,
    predict_window_model,
    score_windowed_decoding,
    transform_window_features,
)

BINARY_ONE_VS_REST_CLASSIFIERS = ("gradient-boosting", "lasso", "svm-binary", "binary-svm")


@dataclass(frozen=True)
class CrossValidationResult:
    """Predictions and accuracy from feature-level cross-validation."""

    accuracy: float
    predictions: np.ndarray
    fold_ids: np.ndarray


def sequential_fold_ids(n_trials: int, n_folds: int) -> np.ndarray:
    """Return legacy contiguous fold ids from 1 to ``n_folds``."""

    if n_trials < 1:
        raise ValueError("n_trials must be at least 1.")
    if n_folds < 1:
        raise ValueError("n_folds must be at least 1.")
    return np.ceil(np.arange(1, n_trials + 1) / (n_trials / n_folds)).astype(int)


def append_null_class_features(
    stimulus_features: Sequence[Sequence[float]] | np.ndarray,
    labels: Sequence | np.ndarray,
    null_features: Sequence[Sequence[float]] | np.ndarray | None = None,
    *,
    null_label: int | float = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Append baseline/null feature rows with a constant null-class label."""

    stimulus_features = _feature_matrix(stimulus_features, name="stimulus_features")
    labels = _label_vector(labels, expected_length=stimulus_features.shape[0], name="labels")
    if null_features is None:
        return stimulus_features, labels

    null_features = _feature_matrix(null_features, name="null_features")
    null_labels = np.full(null_features.shape[0], null_label, dtype=labels.dtype)
    return np.vstack([stimulus_features, null_features]), np.concatenate([labels, null_labels])


def replace_null_class_predictions(
    predictions: Sequence | np.ndarray,
    *,
    null_label: int | float = 0,
    fallback_label: int | float = 1,
) -> np.ndarray:
    """Replace predicted null labels with a non-null class label."""

    predictions = np.asarray(predictions).copy()
    null_mask = predictions == null_label
    if not np.any(null_mask):
        return predictions
    non_null = predictions[~null_mask]
    if len(non_null) == 0:
        predictions[null_mask] = fallback_label
        return predictions
    nonzero_labels, counts = np.unique(non_null, return_counts=True)
    predictions[null_mask] = nonzero_labels[np.argmin(counts)]
    return predictions


# pylint: disable-next=too-many-arguments,too-many-positional-arguments,too-many-locals
def cross_validate_feature_decoding(
    stimulus_features: Sequence[Sequence[float]] | np.ndarray,
    labels: Sequence | np.ndarray,
    *,
    null_features: Sequence[Sequence[float]] | np.ndarray | None = None,
    n_folds: int = 10,
    classifier: str = "multiclass-svm",
    classifier_param: Any = 0.5,
    components_pca: int | float = float("inf"),
    random_state: int | None = None,
    fit_model: Callable[[np.ndarray, np.ndarray], Any] | None = None,
    null_label: int | float = 0,
) -> CrossValidationResult:
    """Run contiguous-fold decoding on precomputed stimulus/null feature matrices."""

    stimulus_features = _feature_matrix(stimulus_features, name="stimulus_features")
    labels = _label_vector(labels, expected_length=stimulus_features.shape[0], name="labels")
    n_trials = len(labels)
    fold_ids = sequential_fold_ids(n_trials, n_folds)
    features, augmented_labels = append_null_class_features(
        stimulus_features,
        labels,
        null_features,
        null_label=null_label,
    )
    augmented_folds = fold_ids
    if null_features is not None:
        null_features = _feature_matrix(null_features, name="null_features")
        if null_features.shape[0] != n_trials:
            raise ValueError("null_features must contain one row per stimulus trial for fold augmentation.")
        augmented_folds = np.concatenate([fold_ids, fold_ids])

    predictions = np.full(n_trials, np.nan)
    class_labels = np.asarray(sorted(np.unique(labels)))
    for fold in range(1, n_folds + 1):
        train_mask = augmented_folds != fold
        test_mask = (augmented_folds == fold) & (augmented_labels != null_label)
        train_features = features[train_mask]
        train_labels = augmented_labels[train_mask]
        test_features = features[test_mask]

        if classifier in BINARY_ONE_VS_REST_CLASSIFIERS:
            fold_predictions = _one_vs_rest_predictions(
                train_features,
                train_labels,
                test_features,
                class_labels,
                classifier=classifier,
                classifier_param=classifier_param,
                components_pca=components_pca,
                random_state=random_state,
            )
        else:
            model_bundle = fit_window_model(
                train_features,
                train_labels,
                fit_model=_fit_model(classifier, classifier_param, random_state, fit_model),
                components_pca=components_pca,
            )
            fold_predictions, _ = predict_window_model(model_bundle, test_features)
        predictions[fold_ids == fold] = fold_predictions

    predictions = replace_null_class_predictions(predictions, null_label=null_label)
    accuracy = float(np.mean(labels == predictions)) if len(labels) else np.nan
    return CrossValidationResult(accuracy=accuracy, predictions=predictions, fold_ids=fold_ids)


def evaluate_feature_transfer(
    train_features: Sequence[Sequence[float]] | np.ndarray,
    train_labels: Sequence | np.ndarray,
    validation_features: Sequence[Sequence[float]] | np.ndarray,
    validation_labels: Sequence | np.ndarray,
    *,
    train_null_features: Sequence[Sequence[float]] | np.ndarray | None = None,
    classifier: str = "multiclass-svm",
    classifier_param: Any = 0.5,
    components_pca: int | float = float("inf"),
    random_state: int | None = None,
    fit_model: Callable[[np.ndarray, np.ndarray], Any] | None = None,
    null_label: int | float = 0,
) -> WindowedDecodingResult:
    """Train on one feature matrix and score transfer to a validation matrix."""

    train_features, train_labels = append_null_class_features(
        train_features,
        train_labels,
        train_null_features,
        null_label=null_label,
    )
    return score_windowed_decoding(
        train_features,
        train_labels,
        validation_features,
        validation_labels,
        fit_model=_fit_model(classifier, classifier_param, random_state, fit_model),
        components_pca=components_pca,
    )


def _fit_model(classifier: str, classifier_param: Any, random_state: int | None, fit_model: Callable[[np.ndarray, np.ndarray], Any] | None):
    if fit_model is not None:
        return fit_model
    return lambda features, labels: train_classifier(
        features,
        labels,
        classifier,
        classifier_param,
        random_state=random_state,
    )


def _one_vs_rest_predictions(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    test_features: np.ndarray,
    class_labels: np.ndarray,
    *,
    classifier: str,
    classifier_param: Any,
    components_pca: int | float,
    random_state: int | None,
) -> np.ndarray:
    all_scores = np.zeros((test_features.shape[0], len(class_labels)))
    for class_index, class_label in enumerate(class_labels):
        binary_bundle = fit_window_model(
            train_features,
            train_labels == class_label,
            fit_model=lambda features, labels, class_label=class_label: _fit_binary_model(
                features,
                labels,
                classifier=classifier,
                classifier_param=classifier_param,
                random_state=random_state,
            ),
            components_pca=components_pca,
        )
        transformed_test = transform_window_features(binary_bundle, test_features)
        if classifier in ("lasso", "svm-binary", "binary-svm"):
            all_scores[:, class_index] = positive_class_score(binary_bundle.model, transformed_test)
        else:
            all_scores[:, class_index] = binary_bundle.model.predict(transformed_test)
    return class_labels[np.argmax(all_scores, axis=1)]


def _fit_binary_model(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    *,
    classifier: str,
    classifier_param: Any,
    random_state: int | None,
):
    if classifier == "gradient-boosting":
        return train_gradient_boosting(
            train_features,
            train_labels,
            classifier_param,
            random_state=random_state,
        )
    if classifier == "lasso":
        return train_lasso_logistic(
            train_features,
            train_labels,
            classifier_param,
            random_state=random_state,
        )
    if classifier in ("svm-binary", "binary-svm"):
        return train_binary_svm(
            train_features,
            train_labels,
            classifier_param,
            random_state=random_state,
        )
    raise ValueError(f"Unsupported one-vs-rest classifier: {classifier}")


def _feature_matrix(features: Sequence[Sequence[float]] | np.ndarray, *, name: str) -> np.ndarray:
    matrix = np.asarray(features, dtype=float)
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional feature matrix.")
    if matrix.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one row.")
    return matrix


def _label_vector(labels: Sequence | np.ndarray, *, expected_length: int, name: str) -> np.ndarray:
    vector = np.asarray(labels).ravel()
    if len(vector) != expected_length:
        raise ValueError(f"{name} length must match feature rows: {len(vector)} != {expected_length}.")
    return vector
