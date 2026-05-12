from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from reptrace.observations import ProbabilityObservationTable

DECODER_CHOICES = ("logistic", "lda", "linear_svm")
EMISSION_MODE_CHOICES = ("calibrated", "uncalibrated")


@dataclass(frozen=True)
class DecoderFitResult:
    """Held-out predictions emitted by a decoder backend for one split/window."""

    probabilities: np.ndarray
    predictions: np.ndarray
    test_labels: np.ndarray
    backend: str
    decoder: str
    emission_mode: str


class DecoderBackend(Protocol):
    """Common interface for decoder families that emit probability observations."""

    name: str
    decoder: str
    emission_mode: str

    def fit_predict(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        train_idx: np.ndarray,
        test_idx: np.ndarray,
    ) -> DecoderFitResult:
        """Fit on ``train_idx`` and return held-out probabilities for ``test_idx``."""
        ...

    def fit_predict_observation_table(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        train_idx: np.ndarray,
        test_idx: np.ndarray,
        *,
        class_names: Sequence[object],
        fold: int,
        split_id: str,
        seed: int | None,
        train_time: float | None,
        test_time: float,
        original_indices: Sequence[int] | np.ndarray | None = None,
        subject: str | None = None,
        session_values: Sequence[object] | np.ndarray | None = None,
        group_values: Sequence[object] | np.ndarray | None = None,
        window_start: float | None = None,
        window_stop: float | None = None,
        calibration_fold: str | int | None = None,
        preprocessing_hash: str | None = None,
        model_hash: str | None = None,
    ) -> ProbabilityObservationTable:
        """Fit and return a canonical held-out probability-observation table."""
        ...


@dataclass(frozen=True)
class SklearnDecoderBackend:
    """Scikit-learn decoder backend for RepTrace probability observations."""

    decoder: str = "logistic"
    max_iter: int = 1000
    emission_mode: str = "calibrated"
    name: str = "sklearn"

    def __post_init__(self) -> None:
        object.__setattr__(self, "decoder", normalize_decoder_name(self.decoder))
        object.__setattr__(self, "emission_mode", normalize_emission_mode(self.emission_mode))

    def fit_predict(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        train_idx: np.ndarray,
        test_idx: np.ndarray,
    ) -> DecoderFitResult:
        """Fit the configured sklearn decoder and return held-out probabilities."""
        model = make_decoder(self.decoder, max_iter=self.max_iter, emission_mode=self.emission_mode)
        model.fit(features[train_idx], labels[train_idx])
        probabilities = predict_emission_probabilities(model, features[test_idx], emission_mode=self.emission_mode)
        return DecoderFitResult(
            probabilities=probabilities,
            predictions=probabilities.argmax(axis=1),
            test_labels=labels[test_idx],
            backend=self.name,
            decoder=self.decoder,
            emission_mode=self.emission_mode,
        )

    def fit_predict_observation_table(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        train_idx: np.ndarray,
        test_idx: np.ndarray,
        *,
        class_names: Sequence[object],
        fold: int,
        split_id: str,
        seed: int | None,
        train_time: float | None,
        test_time: float,
        original_indices: Sequence[int] | np.ndarray | None = None,
        subject: str | None = None,
        session_values: Sequence[object] | np.ndarray | None = None,
        group_values: Sequence[object] | np.ndarray | None = None,
        window_start: float | None = None,
        window_stop: float | None = None,
        calibration_fold: str | int | None = None,
        preprocessing_hash: str | None = None,
        model_hash: str | None = None,
    ) -> ProbabilityObservationTable:
        """Fit and return a canonical held-out probability-observation table."""
        result = self.fit_predict(features, labels, train_idx, test_idx)
        return ProbabilityObservationTable.from_decoded_fold(
            probabilities=result.probabilities,
            test_labels=result.test_labels,
            predictions=result.predictions,
            class_names=class_names,
            test_indices=test_idx,
            original_indices=original_indices,
            subject=subject,
            session_values=session_values,
            group_values=group_values,
            fold=fold,
            split_id=split_id,
            seed=seed,
            decoder=result.decoder,
            backend=result.backend,
            emission_mode=result.emission_mode,
            train_time=train_time,
            test_time=test_time,
            window_start=window_start,
            window_stop=window_stop,
            calibration_fold=calibration_fold,
            preprocessing_hash=preprocessing_hash,
            model_hash=model_hash,
        )


def make_decoder_backend(name: str = "logistic", *, max_iter: int = 1000, emission_mode: str = "calibrated") -> DecoderBackend:
    """Create the default decoder backend for a decoder name.

    This keeps the public factory backend-oriented without changing existing
    decoder names. Future backends can be selected by expanding this factory or
    by accepting explicit backend objects in workflow functions.
    """
    return SklearnDecoderBackend(decoder=name, max_iter=max_iter, emission_mode=emission_mode)


def make_logistic_decoder(max_iter: int = 1000):
    """Create the default calibrated-probability baseline decoder."""
    return make_decoder("logistic", max_iter=max_iter)


def make_decoder(name: str = "logistic", *, max_iter: int = 1000, emission_mode: str = "calibrated"):
    """Create a standard probability-producing decoder by name."""
    normalized = normalize_decoder_name(name)
    emission_mode = normalize_emission_mode(emission_mode)

    if normalized == "logistic":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(
                class_weight="balanced",
                max_iter=max_iter,
                solver="lbfgs",
            ),
        )
    if normalized == "lda":
        return make_pipeline(
            StandardScaler(),
            LinearDiscriminantAnalysis(solver="svd"),
        )

    linear_svm = make_pipeline(
        StandardScaler(),
        LinearSVC(
            class_weight="balanced",
            max_iter=max_iter,
        ),
    )
    if emission_mode == "uncalibrated":
        return linear_svm
    return CalibratedClassifierCV(
        estimator=linear_svm,
        method="sigmoid",
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


def make_cross_validator(labels: np.ndarray, groups: np.ndarray | None, n_splits: int, *, random_state: int | None = 13):
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
        return StratifiedGroupKFold(n_splits=n_splits, shuffle=random_state is not None, random_state=random_state).split(
            np.zeros_like(labels),
            labels,
            groups,
        )
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state).split(
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
