from __future__ import annotations

import ast
import inspect
import json
from collections.abc import Sequence
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from reptrace.decoding.classifiers import CLASSIFIER_REGISTRY, get_default_classifier_param, train_multiclass_classifier
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
TUNING_SCORING_CHOICES = ("accuracy", "balanced_accuracy", "neg_log_loss")
DEFAULT_TUNING_C_GRID = (0.01, 0.1, 1.0, 10.0, 100.0)
_CALIBRATED_REGISTRY_DECODERS = ("correlation-prototype", "multiclass-svm", "multiclass-svm-weighted")


class RegistryClassifier(ClassifierMixin, BaseEstimator):
    """Sklearn-compatible adapter around the shared classifier registry."""

    def __init__(self, classifier: str, classifier_param: Any = None, random_state: int | None = None):
        self.classifier = classifier
        self.classifier_param = classifier_param
        self.random_state = random_state

    def fit(self, features, labels):
        normalized = normalize_decoder_name(self.classifier)
        if normalized not in REGISTRY_DECODER_CHOICES:
            raise ValueError(f"RegistryClassifier only supports registry decoders, got '{self.classifier}'.")
        self.model_ = train_multiclass_classifier(
            features,
            labels,
            normalized,
            resolve_decoder_param(normalized, self.classifier_param),
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
            return scores[:, 1] if scores.ndim == 2 and scores.shape[1] == 2 else scores
        if hasattr(self.model_, "predict_proba"):
            probabilities = np.asarray(self.model_.predict_proba(features), dtype=float)
            return probabilities[:, 1] if probabilities.ndim == 2 and probabilities.shape[1] == 2 else probabilities
        predictions = np.asarray(self.model_.predict(features), dtype=int)
        scores = np.zeros((predictions.shape[0], self.classes_.shape[0]), dtype=float)
        for row_index, encoded_label in enumerate(predictions):
            class_positions = np.where(self.classes_ == encoded_label)[0]
            if class_positions.size:
                scores[row_index, int(class_positions[0])] = 1.0
        return scores


def make_logistic_decoder(max_iter: int = 1000, *, feature_preprocessor: str = "none", pca_components: int | float | str | None = None):
    return make_decoder("logistic", max_iter=max_iter, feature_preprocessor=feature_preprocessor, pca_components=pca_components)


def make_decoder(
    name: str = "logistic",
    *,
    max_iter: int = 1000,
    emission_mode: str = "calibrated",
    feature_preprocessor: str = "none",
    pca_components: int | float | str | None = None,
    decoder_param: Any = None,
    random_state: int | None = 13,
    tune_hyperparameters: bool = False,
    tuning_cv: int | Sequence[tuple[np.ndarray, np.ndarray]] = 3,
    tuning_scoring: str = "accuracy",
    tuning_c_grid: Sequence[float] | str | None = None,
):
    normalized = normalize_decoder_name(name)
    emission_mode = normalize_emission_mode(emission_mode)
    feature_steps = _feature_preprocessor_steps(feature_preprocessor, pca_components)
    if tune_hyperparameters:
        return make_tuned_decoder(
            normalized,
            max_iter=max_iter,
            emission_mode=emission_mode,
            feature_preprocessor=feature_preprocessor,
            pca_components=pca_components,
            cv=tuning_cv,
            scoring=tuning_scoring,
            c_grid=tuning_c_grid,
            random_state=random_state,
        )
    if normalized == "logistic":
        return make_pipeline(
            StandardScaler(),
            *feature_steps,
            LogisticRegression(class_weight="balanced", max_iter=max_iter, solver="lbfgs", random_state=random_state),
        )
    if normalized == "lda":
        return make_pipeline(StandardScaler(), *feature_steps, LinearDiscriminantAnalysis(solver="svd"))
    if normalized in REGISTRY_DECODER_CHOICES:
        registry_decoder = make_pipeline(
            StandardScaler(),
            *feature_steps,
            RegistryClassifier(normalized, classifier_param=decoder_param, random_state=random_state),
        )
        if emission_mode == "calibrated" and normalized in _CALIBRATED_REGISTRY_DECODERS:
            return _make_calibrated_classifier(registry_decoder, method="sigmoid", cv=3)
        return registry_decoder
    linear_svm = make_pipeline(
        StandardScaler(),
        *feature_steps,
        LinearSVC(class_weight="balanced", max_iter=max_iter, random_state=random_state),
    )
    if emission_mode == "uncalibrated":
        return linear_svm
    return _make_calibrated_classifier(linear_svm, method="sigmoid", cv=3)


def make_tuned_decoder(
    name: str = "logistic",
    *,
    max_iter: int = 1000,
    emission_mode: str = "calibrated",
    feature_preprocessor: str = "none",
    pca_components: int | float | str | None = None,
    cv: int | Sequence[tuple[np.ndarray, np.ndarray]] = 3,
    scoring: str = "accuracy",
    c_grid: Sequence[float] | str | None = None,
    random_state: int | None = 13,
):
    normalized = normalize_decoder_name(name)
    emission_mode = normalize_emission_mode(emission_mode)
    scoring = normalize_tuning_scoring(scoring)
    c_grid = parse_c_grid(c_grid)
    feature_steps = _feature_preprocessor_steps(feature_preprocessor, pca_components)
    if normalized in REGISTRY_DECODER_CHOICES:
        raise ValueError("Hyperparameter tuning is currently supported only for logistic, lda, and linear_svm decoders.")
    if normalized == "logistic":
        estimator = make_pipeline(
            StandardScaler(),
            *feature_steps,
            LogisticRegression(class_weight="balanced", max_iter=max_iter, solver="lbfgs", random_state=random_state),
        )
        param_grid = {"logisticregression__C": c_grid}
    elif normalized == "lda":
        estimator = make_pipeline(StandardScaler(), *feature_steps, LinearDiscriminantAnalysis())
        param_grid = [
            {"lineardiscriminantanalysis__solver": ["svd"], "lineardiscriminantanalysis__shrinkage": [None]},
            {"lineardiscriminantanalysis__solver": ["lsqr"], "lineardiscriminantanalysis__shrinkage": ["auto"]},
        ]
    elif normalized == "linear_svm":
        if emission_mode == "uncalibrated" and scoring == "neg_log_loss":
            raise ValueError("neg_log_loss tuning requires probability estimates; use calibrated emissions for linear_svm.")
        linear_svm = make_pipeline(
            StandardScaler(),
            *feature_steps,
            LinearSVC(class_weight="balanced", max_iter=max_iter, random_state=random_state),
        )
        if emission_mode == "uncalibrated":
            estimator = linear_svm
            param_grid = {"linearsvc__C": c_grid}
        else:
            estimator = _make_calibrated_classifier(linear_svm, method="sigmoid", cv=3)
            param_grid = {_calibrated_estimator_param(estimator, "linearsvc__C"): c_grid}
    else:
        raise ValueError(f"Unsupported tuned decoder '{name}'.")
    return GridSearchCV(estimator=estimator, param_grid=param_grid, scoring=scoring, cv=cv, refit=True)


def _make_calibrated_classifier(estimator, *, method: str, cv: int):
    kwargs = {"method": method, "cv": cv}
    if "estimator" in inspect.signature(CalibratedClassifierCV).parameters:
        kwargs["estimator"] = estimator
    else:
        kwargs["base_estimator"] = estimator
    return CalibratedClassifierCV(**kwargs)


def _calibrated_estimator_param(estimator, nested_parameter: str) -> str:
    params = estimator.get_params()
    for prefix in ("estimator", "base_estimator"):
        candidate = f"{prefix}__{nested_parameter}"
        if candidate in params:
            return candidate
    raise ValueError(f"Could not find calibrated-estimator parameter for '{nested_parameter}'.")


def parse_c_grid(values: Sequence[float] | str | None) -> tuple[float, ...]:
    if values is None:
        return DEFAULT_TUNING_C_GRID
    if isinstance(values, str):
        values = [value.strip() for value in values.split(",") if value.strip()]
    grid = tuple(float(value) for value in values)
    if not grid:
        raise ValueError("At least one C value is required for hyperparameter tuning.")
    if any(value <= 0 for value in grid):
        raise ValueError("All C values must be positive.")
    return grid


def normalize_tuning_scoring(scoring: str) -> str:
    normalized = scoring.lower().replace("-", "_")
    if normalized not in TUNING_SCORING_CHOICES:
        raise ValueError(f"Unknown tuning scoring '{scoring}'. Available values: {', '.join(TUNING_SCORING_CHOICES)}.")
    return normalized


def normalize_decoder_name(name: str) -> str:
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
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if stripped == "" or stripped.lower() in {"none", "default", "null"}:
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
    normalized = normalize_decoder_name(decoder)
    parsed = normalize_decoder_param(value)
    if normalized in REGISTRY_DECODER_CHOICES and parsed is None:
        return get_default_classifier_param(normalized)
    return parsed


def normalize_emission_mode(mode: str) -> str:
    normalized = mode.lower().replace("-", "_")
    if normalized not in EMISSION_MODE_CHOICES:
        raise ValueError(f"Unknown emission mode '{mode}'. Available modes: {', '.join(EMISSION_MODE_CHOICES)}.")
    return normalized


def normalize_feature_preprocessor(name: str | None) -> str:
    normalized = "none" if name is None else name.lower().replace("-", "_")
    if normalized in {"identity", "standard", "standardize", "scaler", "standard_scaler"}:
        return "none"
    if normalized in {"pca_whitened", "whitened_pca", "whiten_pca"}:
        return "pca_whiten"
    if normalized not in FEATURE_PREPROCESSOR_CHOICES:
        raise ValueError(f"Unknown feature preprocessor '{name}'. Available preprocessors: {', '.join(FEATURE_PREPROCESSOR_CHOICES)}.")
    return normalized


def normalize_pca_components(n_components: int | float | str | None) -> int | float | None:
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
    return [PCA(n_components=normalize_pca_components(pca_components), whiten=normalized == "pca_whiten", svd_solver="full")]


def score_to_probabilities(scores: np.ndarray) -> np.ndarray:
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
    _, class_counts = np.unique(labels, return_counts=True)
    if len(class_counts) < 2:
        raise ValueError("Need at least two classes for decoding.")
    if np.min(class_counts) < n_splits:
        raise ValueError(f"Need at least {n_splits} examples per class; smallest class has {np.min(class_counts)}.")
    if groups is not None:
        unique_groups = np.unique(groups)
        if len(unique_groups) < n_splits:
            raise ValueError(f"Need at least {n_splits} groups for grouped CV, found {len(unique_groups)}.")
        return StratifiedGroupKFold(n_splits=n_splits).split(np.zeros_like(labels), labels, groups)
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=13).split(np.zeros_like(labels), labels)


def make_tuning_cross_validator(labels: np.ndarray, groups: np.ndarray | None, n_splits: int):
    _, class_counts = np.unique(labels, return_counts=True)
    if len(class_counts) < 2:
        raise ValueError("Need at least two classes for decoder hyperparameter tuning.")
    feasible_splits = min(int(n_splits), int(np.min(class_counts)))
    if groups is not None:
        feasible_splits = min(feasible_splits, len(np.unique(groups)))
    if feasible_splits < 2:
        raise ValueError("Need at least two examples per class and two groups when grouped to tune decoder hyperparameters.")
    return list(make_cross_validator(labels, groups, feasible_splits))


def time_windows(times: np.ndarray, window_ms: float, step_ms: float) -> list[tuple[int, int, float]]:
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
        windows.append((start, stop, float(np.mean(times[start:stop]))))
    return windows
