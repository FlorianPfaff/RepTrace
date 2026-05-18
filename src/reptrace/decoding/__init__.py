from __future__ import annotations

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from reptrace.decoding.sampling import (
    CLASS_LIMIT_SELECTION_MODES as CLASS_LIMIT_SELECTION_MODES,
    DEFAULT_CLASS_LIMIT_SEED as DEFAULT_CLASS_LIMIT_SEED,
    DEFAULT_CLASS_LIMIT_SELECTION as DEFAULT_CLASS_LIMIT_SELECTION,
    normalize_class_limit_seed as normalize_class_limit_seed,
    normalize_class_limit_selection as normalize_class_limit_selection,
    select_class_limited_indices as select_class_limited_indices,
)

DECODER_CHOICES = ("logistic", "lda", "linear_svm")
EMISSION_MODE_CHOICES = ("calibrated", "uncalibrated")
FEATURE_PREPROCESSOR_CHOICES = ("none", "pca", "pca_whiten")
CALIBRATION_METHOD_CHOICES = ("sigmoid", "isotonic")


def make_logistic_decoder(
    max_iter: int = 1000,
    *,
    feature_preprocessor: str = "none",
    pca_components: int | float | str | None = None,
    regularization_c: int | float | str = 1.0,
):
    """Create the default calibrated-probability baseline decoder."""
    return make_decoder(
        "logistic",
        max_iter=max_iter,
        feature_preprocessor=feature_preprocessor,
        pca_components=pca_components,
        regularization_c=regularization_c,
    )


def make_decoder(
    name: str = "logistic",
    *,
    max_iter: int = 1000,
    emission_mode: str = "calibrated",
    feature_preprocessor: str = "none",
    pca_components: int | float | str | None = None,
    regularization_c: int | float | str = 1.0,
    lda_shrinkage: str | float | None = None,
    calibration_method: str = "sigmoid",
):
    """Create a standard probability-producing decoder by name.

    Optional feature preprocessing is inserted after fold-local standardization
    and before the classifier. This keeps low-rank transforms such as PCA inside
    each cross-validation fold and prevents train/test leakage.
    """
    normalized = normalize_decoder_name(name)
    emission_mode = normalize_emission_mode(emission_mode)
    regularization_c = normalize_regularization_c(regularization_c)
    lda_shrinkage = normalize_lda_shrinkage(lda_shrinkage)
    calibration_method = normalize_calibration_method(calibration_method)
    feature_steps = _feature_preprocessor_steps(feature_preprocessor, pca_components)

    if normalized == "logistic":
        return make_pipeline(
            StandardScaler(),
            *feature_steps,
            LogisticRegression(
                C=regularization_c,
                class_weight="balanced",
                max_iter=max_iter,
                solver="lbfgs",
            ),
        )
    if normalized == "lda":
        if lda_shrinkage is None:
            classifier = LinearDiscriminantAnalysis(solver="svd")
        else:
            classifier = LinearDiscriminantAnalysis(solver="lsqr", shrinkage=lda_shrinkage)
        return make_pipeline(
            StandardScaler(),
            *feature_steps,
            classifier,
        )

    linear_svm = make_pipeline(
        StandardScaler(),
        *feature_steps,
        LinearSVC(
            C=regularization_c,
            class_weight="balanced",
            max_iter=max_iter,
        ),
    )
    if emission_mode == "uncalibrated":
        return linear_svm
    return _make_calibrated_classifier(linear_svm, method=calibration_method)


def _make_calibrated_classifier(estimator, *, method: str = "sigmoid"):
    method = normalize_calibration_method(method)
    try:
        return CalibratedClassifierCV(
            estimator=estimator,
            method=method,
            cv=3,
        )
    except TypeError:
        return CalibratedClassifierCV(
            base_estimator=estimator,
            method=method,
            cv=3,
        )


def normalize_decoder_name(name: str) -> str:
    """Normalize decoder aliases to the names used in result tables."""
    normalized = name.lower().replace("-", "_")
    if normalized == "svm":
        return "linear_svm"
    if normalized not in DECODER_CHOICES:
        raise ValueError(f"Unknown decoder '{name}'. Available decoders: {', '.join(DECODER_CHOICES)}.")
    return normalized


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


def normalize_calibration_method(method: str) -> str:
    """Normalize probability-calibration method names for sklearn."""
    normalized = method.lower().replace("-", "_")
    if normalized not in CALIBRATION_METHOD_CHOICES:
        raise ValueError(f"Unknown calibration method '{method}'. Available methods: {', '.join(CALIBRATION_METHOD_CHOICES)}.")
    return normalized


def normalize_regularization_c(value: int | float | str) -> float:
    """Normalize inverse regularization strength for linear classifiers."""
    if isinstance(value, str):
        try:
            value = float(value.strip())
        except ValueError as exc:
            raise ValueError("regularization_c must be a positive finite number.") from exc
    if isinstance(value, bool):
        raise ValueError("regularization_c must be numeric, not boolean.")
    if isinstance(value, (np.integer,)):
        value = int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0.0:
        raise ValueError("regularization_c must be a positive finite number.")
    return float(value)


def normalize_lda_shrinkage(value: str | float | None) -> str | float | None:
    """Normalize LDA shrinkage settings.

    ``None``/``none`` keeps sklearn's SVD LDA baseline. ``auto`` and floats in
    ``[0, 1]`` select the shrinkage-capable LSQR solver.
    """
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip().lower()
        if stripped == "" or stripped == "none":
            return None
        if stripped == "auto":
            return "auto"
        try:
            value = float(stripped)
        except ValueError as exc:
            raise ValueError("lda_shrinkage must be None, 'auto', or a float in [0, 1].") from exc
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, bool) or not isinstance(value, float) or not np.isfinite(value) or value < 0.0 or value > 1.0:
        raise ValueError("lda_shrinkage must be None, 'auto', or a float in [0, 1].")
    return float(value)


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
        return np.asarray(model.predict_proba(features), dtype=float)
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
