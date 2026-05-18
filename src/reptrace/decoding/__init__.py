from __future__ import annotations

import ast
import json
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from reptrace.decoding.classifiers import (
    CLASSIFIER_REGISTRY,
    get_default_classifier_param,
    train_multiclass_classifier,
)
from reptrace.decoding.sampling import (
    CLASS_LIMIT_SELECTION_MODES as CLASS_LIMIT_SELECTION_MODES,
    DEFAULT_CLASS_LIMIT_SEED as DEFAULT_CLASS_LIMIT_SEED,
    DEFAULT_CLASS_LIMIT_SELECTION as DEFAULT_CLASS_LIMIT_SELECTION,
    normalize_class_limit_seed as normalize_class_limit_seed,
    normalize_class_limit_selection as normalize_class_limit_selection,
    select_class_limited_indices as select_class_limited_indices,
)

STANDARD_DECODER_CHOICES = ("logistic", "lda", "linear_svm")
REGISTRY_DECODER_CHOICES = tuple(sorted(CLASSIFIER_REGISTRY))
DECODER_CHOICES = (*STANDARD_DECODER_CHOICES, *REGISTRY_DECODER_CHOICES)
EMISSION_MODE_CHOICES = ("calibrated", "uncalibrated")
FEATURE_PREPROCESSOR_CHOICES = ("none", "pca", "pca_whiten")
_CALIBRATED_REGISTRY_DECODERS = (
    "correlation-prototype",
    "multiclass-svm",
    "multiclass-svm-weighted",
)


class RegistryClassifier(ClassifierMixin, BaseEstimator):
    """Sklearn-compatible adapter around the shared classifier registry."""

    def __init__(
        self,
        classifier: str,
        classifier_param: Any = None,
        random_state: int | None = None,
    ):
        self.classifier = classifier
        self.classifier_param = classifier_param
        self.random_state = random_state

    def fit(self, features, labels):
        normalized = normalize_decoder_name(self.classifier)
        if normalized not in REGISTRY_DECODER_CHOICES:
            raise ValueError(f"RegistryClassifier only supports registry decoders, got '{self.classifier}'.")
        classifier_param = resolve_decoder_param(normalized, self.classifier_param)
        self.model_ = train_multiclass_classifier(
            features,
            labels,
            normalized,
            classifier_param,
            random_state=self.random_state,
        )
        self.classes_ = np.asarray(self.model_.classes_)
        return self

    def predict(self, features):
        return self.model_.predict(features)

    def predict_proba(self, features):
        if not hasattr(self.model_, "predict_proba"):
            raise AttributeError(f"{self.model_.__class__.__name__!r} object has no attribute 'predict_proba'")
        return np.asarray(self.model_.predict_proba(features), dtype=float)

    def decision_function(self, features):
        if hasattr(self.model_, "decision_function"):
            scores = np.asarray(self.model_.decision_function(features), dtype=float)
            if scores.ndim == 2 and scores.shape[1] == 2:
                return scores[:, 1]
            return scores
        if hasattr(self.model_, "predict_proba"):
            probabilities = np.asarray(self.model_.predict_proba(features), dtype=float)
            if probabilities.ndim == 2 and probabilities.shape[1] == 2:
                return probabilities[:, 1]
            return probabilities
        predictions = np.asarray(self.model_.predict(features), dtype=int)
        scores = np.zeros((predictions.shape[0], self.classes_.shape[0]), dtype=float)
        for row_index, encoded_label in enumerate(predictions):
            class_positions = np.where(self.classes_ == encoded_label)[0]
            if class_positions.size:
                scores[row_index, int(class_positions[0])] = 1.0
        return scores


def make_logistic_decoder(
    max_iter: int = 1000,
    *,
    feature_preprocessor: str = "none",
    pca_components: int | float | str | None = None,
):
    """Create the default calibrated-probability baseline decoder."""
    return make_decoder(
        "logistic",
        max_iter=max_iter,
        feature_preprocessor=feature_preprocessor,
        pca_components=pca_components,
    )


def make_decoder(
    name: str = "logistic",
    *,
    max_iter: int = 1000,
    emission_mode: str = "calibrated",
    feature_preprocessor: str = "none",
    pca_components: int | float | str | None = None,
    decoder_param: Any = None,
    random_state: int | None = 13,
):
    """Create a standard probability-producing decoder by name.

    Optional feature preprocessing is inserted after fold-local standardization
    and before the classifier. This keeps low-rank transforms such as PCA inside
    each cross-validation fold and prevents train/test leakage.
    """
    normalized = normalize_decoder_name(name)
    emission_mode = normalize_emission_mode(emission_mode)
    feature_steps = _feature_preprocessor_steps(feature_preprocessor, pca_components)

    if normalized == "logistic":
        return make_pipeline(
            StandardScaler(),
            *feature_steps,
            LogisticRegression(
                class_weight="balanced",
                max_iter=max_iter,
                solver="lbfgs",
                random_state=random_state,
            ),
        )
    if normalized == "lda":
        return make_pipeline(
            StandardScaler(),
            *feature_steps,
            LinearDiscriminantAnalysis(solver="svd"),
        )

    if normalized in REGISTRY_DECODER_CHOICES:
        registry_decoder = make_pipeline(
            StandardScaler(),
            *feature_steps,
            RegistryClassifier(
                normalized,
                classifier_param=decoder_param,
                random_state=random_state,
            ),
        )
        if emission_mode == "calibrated" and normalized in _CALIBRATED_REGISTRY_DECODERS:
            return _make_calibrated_classifier(registry_decoder)
        return registry_decoder

    linear_svm = make_pipeline(
        StandardScaler(),
        *feature_steps,
        LinearSVC(
            class_weight="balanced",
            max_iter=max_iter,
            random_state=random_state,
        ),
    )
    if emission_mode == "uncalibrated":
        return linear_svm
    return _make_calibrated_classifier(linear_svm)


def _make_calibrated_classifier(estimator):
    try:
        return CalibratedClassifierCV(
            estimator=estimator,
            method="sigmoid",
            cv=3,
        )
    except TypeError:
        return CalibratedClassifierCV(
            base_estimator=estimator,
            method="sigmoid",
            cv=3,
        )


def normalize_decoder_name(name: str) -> str:
    """Normalize decoder aliases to the names used in result tables."""
    normalized = name.lower().replace("-", "_")
    if normalized == "svm":
        return "linear_svm"
    if normalized in STANDARD_DECODER_CHOICES:
        return normalized
    for registry_name in REGISTRY_DECODER_CHOICES:
        if normalized == registry_name.lower().replace("-", "_"):
            return registry_name
    raise ValueError(f"Unknown decoder '{name}'. Available decoders: {', '.join(DECODER_CHOICES)}.")


def normalize_decoder_param(value: Any) -> Any:
    """Parse optional decoder/classifier parameters from CLI or manifest values."""
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if stripped == "" or stripped.lower() in {"none", "default"}:
            return None
        if stripped.lower() == "null":
            return None
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            pass
        try:
            return ast.literal_eval(stripped)
        except (ValueError, SyntaxError):
            return stripped
    return value


def resolve_decoder_param(decoder: str, value: Any) -> Any:
    """Return the effective classifier parameter used by a decoder."""
    normalized = normalize_decoder_name(decoder)
    parsed = normalize_decoder_param(value)
    if normalized in REGISTRY_DECODER_CHOICES and parsed is None:
        return get_default_classifier_param(normalized)
    return parsed


def normalize_emission_mode(mode: str) -> str:
    """Normalize calibrated/uncalibrated emission mode names."""
    normalized = mode.lower().replace("-", "_")
    if normalized not in EMISSION_MODE_CHOICES:
        raise ValueError(f"Unknown emission mode '{mode}'. Available modes: {', '.join(EMISSION_MODE_CHOICES)}.")
    return normalized


def normalize_feature_preprocessor(name: str | None) -> str:
    """Normalize feature-preprocessor aliases to canonical result-table names."""
    normalized = "none" if name is None else name.lower().replace("-", "_")
    if normalized in {"identity", "standard", "standardize", "scaler", "standard_scaler"}:
        return "none"
    if normalized in {"pca_whitened", "whitened_pca", "whiten_pca"}:
        return "pca_whiten"
    if normalized not in FEATURE_PREPROCESSOR_CHOICES:
        raise ValueError(
            f"Unknown feature preprocessor '{name}'. Available preprocessors: {', '.join(FEATURE_PREPROCESSOR_CHOICES)}."
        )
    return normalized


def normalize_pca_components(n_components: int | float | str | None) -> int | float | None:
    """Normalize PCA component specifications for sklearn.

    Integers select an explicit component count. Floats in ``(0, 1)`` select an
    explained-variance fraction. ``None``, ``auto``, or an empty string keep
    sklearn's default ``PCA(n_components=None)`` behavior.
    """
    if n_components is None:
        return None
    if isinstance(n_components, str):
        stripped = n_components.strip()
        if stripped == "" or stripped.lower() in {"none", "auto", "default"}:
            return None
        try:
            parsed: int | float = float(stripped) if any(marker in stripped for marker in (".", "e", "E")) else int(stripped)
        except ValueError as exc:
            raise ValueError("pca_components must be an integer count, a variance fraction in (0, 1), or None.") from exc
        return normalize_pca_components(parsed)
    if isinstance(n_components, (np.integer,)):
        n_components = int(n_components)
    if isinstance(n_components, (np.floating,)):
        n_components = float(n_components)
    if isinstance(n_components, bool):
        raise ValueError("pca_components must be numeric, not boolean.")
    if isinstance(n_components, int):
        if n_components < 1:
            raise ValueError("Integer pca_components must be at least 1.")
        return n_components
    if isinstance(n_components, float):
        if not np.isfinite(n_components) or n_components <= 0.0:
            raise ValueError("Float pca_components must be finite and positive.")
        if n_components < 1.0:
            return float(n_components)
        if n_components.is_integer():
            return int(n_components)
    raise ValueError("pca_components must be an integer count, a variance fraction in (0, 1), or None.")


def _feature_preprocessor_steps(feature_preprocessor: str | None, pca_components: int | float | str | None) -> list[PCA]:
    normalized = normalize_feature_preprocessor(feature_preprocessor)
    if normalized == "none":
        if pca_components is not None:
            raise ValueError("pca_components can only be set when feature_preprocessor is 'pca' or 'pca_whiten'.")
        return []
    return [
        PCA(
            n_components=normalize_pca_components(pca_components),
            whiten=normalized == "pca_whiten",
            svd_solver="full",
        )
    ]


def score_to_probabilities(scores: np.ndarray) -> np.ndarray:
    """Convert uncalibrated decision scores into pseudo-probability emissions."""
    scores = np.asarray(scores, dtype=float)
    if scores.ndim == 1:
        clipped = np.clip(scores, -50.0, 50.0)
        positive = 1.0 / (1.0 + np.exp(-clipped))
        return np.column_stack([1.0 - positive, positive])
    if scores.ndim != 2:
        raise ValueError("Decision scores must be one- or two-dimensional.")
    shifted = scores - scores.max(axis=1, keepdims=True)
    exp_scores = np.exp(np.clip(shifted, -50.0, 50.0))
    return exp_scores / exp_scores.sum(axis=1, keepdims=True)


def predict_emission_probabilities(model, features: np.ndarray, *, emission_mode: str = "calibrated") -> np.ndarray:
    """Predict calibrated probabilities or uncalibrated score-derived emissions."""
    emission_mode = normalize_emission_mode(emission_mode)
    if emission_mode == "uncalibrated" and hasattr(model, "decision_function"):
        return score_to_probabilities(model.decision_function(features))
    if hasattr(model, "predict_proba"):
        try:
            return np.asarray(model.predict_proba(features), dtype=float)
        except AttributeError:
            pass
    if hasattr(model, "decision_function"):
        return score_to_probabilities(model.decision_function(features))
    raise ValueError("Decoder does not provide predict_proba or decision_function.")


def make_cross_validator(labels: np.ndarray, groups: np.ndarray | None, n_splits: int):
    """Create stratified CV splits, optionally preserving group boundaries."""
    _, class_counts = np.unique(labels, return_counts=True)
    if len(class_counts) < 2:
        raise ValueError("Need at least two classes for decoding.")
    if np.min(class_counts) < n_splits:
        raise ValueError(
            f"Need at least {n_splits} examples per class; smallest class has {np.min(class_counts)}."
        )
    if groups is not None:
        unique_groups = np.unique(groups)
        if len(unique_groups) < n_splits:
            raise ValueError(
                f"Need at least {n_splits} groups for grouped CV, found {len(unique_groups)}."
            )
        return StratifiedGroupKFold(n_splits=n_splits).split(
            np.zeros_like(labels),
            labels,
            groups,
        )
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=13).split(
        np.zeros_like(labels),
        labels,
    )


def time_windows(times: np.ndarray, window_ms: float, step_ms: float) -> list[tuple[int, int, float]]:
    """Return sample index windows and their center times for time-resolved decoding."""
    if times.ndim != 1:
        raise ValueError("times must be one-dimensional")
    if len(times) < 2:
        raise ValueError("times must contain at least two samples")
    if window_ms <= 0 or step_ms <= 0:
        raise ValueError("window_ms and step_ms must be positive")

    sfreq = 1000.0 / np.median(np.diff(times * 1000.0))
    window_samples = max(1, int(round((window_ms / 1000.0) * sfreq)))
    step_samples = max(1, int(round((step_ms / 1000.0) * sfreq)))
    windows = []
    for start in range(0, len(times) - window_samples + 1, step_samples):
        stop = start + window_samples
        center = float(np.mean(times[start:stop]))
        windows.append((start, stop, center))
    return windows
