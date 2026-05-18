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
DEFAULT_SVM_CALIBRATION_SPLITS = 3


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
    calibration_cv=None,
):
    """Create a standard probability-producing decoder by name.

    Optional feature preprocessing is inserted after fold-local standardization
    and before the classifier. This keeps low-rank transforms such as PCA inside
    each cross-validation fold and prevents train/test leakage.

    ``calibration_cv`` is only used for calibrated linear SVMs. Passing explicit
    split indices lets grouped workflows calibrate the SVM without mixing
    acquisition/session groups inside the inner calibration folds.
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
            ),
        )
    if normalized == "lda":
        return make_pipeline(
            StandardScaler(),
            *feature_steps,
            LinearDiscriminantAnalysis(solver="svd"),
        )

    linear_svm = make_pipeline(
        StandardScaler(),
        *feature_steps,
        LinearSVC(
            class_weight="balanced",
            max_iter=max_iter,
        ),
    )
    if emission_mode == "uncalibrated":
        return linear_svm
    return _make_calibrated_classifier(
        linear_svm,
        cv=DEFAULT_SVM_CALIBRATION_SPLITS if calibration_cv is None else calibration_cv,
    )


def _make_calibrated_classifier(estimator, *, cv=DEFAULT_SVM_CALIBRATION_SPLITS):
    try:
        return CalibratedClassifierCV(
            estimator=estimator,
            method="sigmoid",
            cv=cv,
        )
    except TypeError:
        return CalibratedClassifierCV(
            base_estimator=estimator,
            method="sigmoid",
            cv=cv,
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


def make_grouped_calibration_cv(
    labels: np.ndarray,
    groups: np.ndarray,
    n_splits: int = DEFAULT_SVM_CALIBRATION_SPLITS,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create inner SVM-calibration folds that keep groups disjoint.

    The returned split indices are relative to the outer training set. They can
    be passed directly as ``cv`` to ``CalibratedClassifierCV``.
    """
    labels = np.asarray(labels)
    groups = np.asarray(groups)
    if labels.ndim != 1 or groups.ndim != 1:
        raise ValueError("labels and groups must be one-dimensional for grouped calibration CV.")
    if len(labels) != len(groups):
        raise ValueError("labels and groups must contain the same number of samples for grouped calibration CV.")
    if n_splits < 2:
        raise ValueError("Need at least two calibration splits for grouped SVM calibration.")

    classes, class_counts = np.unique(labels, return_counts=True)
    if len(classes) < 2:
        raise ValueError("Need at least two classes for grouped SVM calibration.")

    unique_groups = np.unique(groups)
    n_inner_splits = min(int(n_splits), int(np.min(class_counts)), len(unique_groups))
    if n_inner_splits < 2:
        raise ValueError(
            "Grouped SVM calibration needs at least two groups and two examples per class in the outer training fold."
        )

    splitter = StratifiedGroupKFold(n_splits=n_inner_splits, shuffle=True, random_state=13)
    splits = [
        (np.asarray(inner_train_idx, dtype=int), np.asarray(calibration_idx, dtype=int))
        for inner_train_idx, calibration_idx in splitter.split(np.zeros(len(labels)), labels, groups)
    ]
    expected_classes = set(classes.tolist())
    for inner_train_idx, calibration_idx in splits:
        train_groups = set(groups[inner_train_idx].tolist())
        calibration_groups = set(groups[calibration_idx].tolist())
        if not train_groups.isdisjoint(calibration_groups):
            raise RuntimeError("Grouped calibration split leaked a group across inner train and calibration folds.")
        if set(labels[inner_train_idx].tolist()) != expected_classes or set(labels[calibration_idx].tolist()) != expected_classes:
            raise ValueError(
                "Grouped SVM calibration requires every inner training and calibration fold to contain all classes; "
                "provide more trials/groups or use --emission-mode uncalibrated."
            )
    return splits


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
