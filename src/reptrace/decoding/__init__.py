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

from reptrace.observations import ProbabilityObservationTable, stable_hash

DECODER_CHOICES = ("logistic", "lda", "linear_svm")
EMISSION_MODE_CHOICES = ("calibrated", "uncalibrated")


@dataclass(frozen=True)
class DecoderPredictionResult:
    """Held-out probabilities emitted by a decoder backend."""

    probabilities: np.ndarray
    predictions: np.ndarray
    test_labels: np.ndarray
    model: object
    backend: str
    decoder: str
    emission_mode: str


class DecoderBackend(Protocol):
    """Feature-matrix decoder backend that can emit canonical observations."""

    name: str
    decoder: str
    emission_mode: str

    def fit(self, features: np.ndarray, labels: np.ndarray) -> object:
        """Fit the backend on a feature matrix and integer labels."""
        ...

    def predict_probabilities(self, model: object, features: np.ndarray) -> np.ndarray:
        """Return class probabilities or probability-like emissions."""
        ...

    def fit_predict(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        train_idx: np.ndarray,
        test_idx: np.ndarray,
    ) -> DecoderPredictionResult:
        """Fit on ``train_idx`` and predict held-out probabilities for ``test_idx``."""
        ...

    def build_observation_table(
        self,
        result: DecoderPredictionResult,
        *,
        class_names: Sequence[object],
        test_indices: Sequence[int] | np.ndarray,
        fold: int | str,
        time: float,
        original_indices: Sequence[int] | np.ndarray | None = None,
        subject: str | None = None,
        session_values: Sequence[object] | np.ndarray | None = None,
        group_values: Sequence[object] | np.ndarray | None = None,
        split_id: str = "",
        seed: int | str | None = None,
        train_time: float | None = None,
        window_start: float | None = None,
        window_stop: float | None = None,
        calibration_fold: str | int | None = None,
        preprocessing_hash: str = "",
        model_hash: str | None = None,
    ) -> ProbabilityObservationTable:
        """Build canonical observation rows from held-out predictions."""
        ...


@dataclass(frozen=True)
class SklearnDecoderBackend:
    """Scikit-learn backend for RepTrace feature-matrix decoders."""

    decoder: str = "logistic"
    max_iter: int = 1000
    emission_mode: str = "calibrated"
    name: str = "sklearn"

    def __post_init__(self) -> None:
        object.__setattr__(self, "decoder", normalize_decoder_name(self.decoder))
        object.__setattr__(self, "emission_mode", normalize_emission_mode(self.emission_mode))

    def model_hash(self) -> str:
        """Return a stable hash for this backend/model configuration."""
        return stable_hash(
            {
                "backend": self.name,
                "decoder": self.decoder,
                "emission_mode": self.emission_mode,
                "max_iter": self.max_iter,
            }
        )

    def fit(self, features: np.ndarray, labels: np.ndarray) -> object:
        """Fit the configured scikit-learn decoder."""
        model = make_decoder(self.decoder, max_iter=self.max_iter, emission_mode=self.emission_mode)
        model.fit(features, labels)
        return model

    def predict_probabilities(self, model: object, features: np.ndarray) -> np.ndarray:
        """Predict calibrated probabilities or score-derived emissions."""
        return predict_emission_probabilities(model, features, emission_mode=self.emission_mode)

    def fit_predict(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        train_idx: np.ndarray,
        test_idx: np.ndarray,
    ) -> DecoderPredictionResult:
        """Fit on ``train_idx`` and predict held-out probabilities for ``test_idx``."""
        model = self.fit(features[train_idx], labels[train_idx])
        probabilities = self.predict_probabilities(model, features[test_idx])
        return DecoderPredictionResult(
            probabilities=probabilities,
            predictions=probabilities.argmax(axis=1),
            test_labels=labels[test_idx],
            model=model,
            backend=self.name,
            decoder=self.decoder,
            emission_mode=self.emission_mode,
        )

    def build_observation_table(
        self,
        result: DecoderPredictionResult,
        *,
        class_names: Sequence[object],
        test_indices: Sequence[int] | np.ndarray,
        fold: int | str,
        time: float,
        original_indices: Sequence[int] | np.ndarray | None = None,
        subject: str | None = None,
        session_values: Sequence[object] | np.ndarray | None = None,
        group_values: Sequence[object] | np.ndarray | None = None,
        split_id: str = "",
        seed: int | str | None = None,
        train_time: float | None = None,
        window_start: float | None = None,
        window_stop: float | None = None,
        calibration_fold: str | int | None = None,
        preprocessing_hash: str = "",
        model_hash: str | None = None,
    ) -> ProbabilityObservationTable:
        """Build canonical observation rows from a backend prediction result."""
        return ProbabilityObservationTable.from_decoded_fold(
            probabilities=result.probabilities,
            test_labels=result.test_labels,
            predictions=result.predictions,
            class_names=class_names,
            test_indices=test_indices,
            fold=fold,
            decoder=result.decoder,
            backend=result.backend,
            emission_mode=result.emission_mode,
            time=time,
            original_indices=original_indices,
            subject=subject,
            session_values=session_values,
            group_values=group_values,
            split_id=split_id,
            seed=seed,
            train_time=train_time,
            window_start=window_start,
            window_stop=window_stop,
            calibration_fold=calibration_fold,
            preprocessing_hash=preprocessing_hash,
            model_hash=self.model_hash() if model_hash is None else model_hash,
        )

    def fit_predict_observation_table(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        train_idx: np.ndarray,
        test_idx: np.ndarray,
        **observation_kwargs: object,
    ) -> ProbabilityObservationTable:
        """Fit, predict, and return canonical held-out observation rows."""
        result = self.fit_predict(features, labels, train_idx, test_idx)
        return self.build_observation_table(result, test_indices=test_idx, **observation_kwargs)


def make_decoder_backend(
    decoder: str = "logistic",
    *,
    backend: str = "sklearn",
    max_iter: int = 1000,
    emission_mode: str = "calibrated",
) -> DecoderBackend:
    """Create a feature-matrix decoder backend.

    The backend API is intentionally independent of MNE objects so it can be
    used by epoch workflows, continuous-scan workflows, and future deep-model
    backends that produce the same canonical probability-observation table.
    """
    normalized_backend = backend.lower().replace("-", "_")
    if normalized_backend in {"sklearn", "scikit_learn"}:
        return SklearnDecoderBackend(decoder=decoder, max_iter=max_iter, emission_mode=emission_mode)
    raise ValueError(f"Unknown decoder backend {backend!r}. Available backends: sklearn.")


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
