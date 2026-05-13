from __future__ import annotations

from typing import Literal

import numpy as np

NormalizationMode = Literal["none", "subject_z", "subject_trial_z", "subject_baseline_z", "subject_baseline_whiten"]


def subject_zscore(features: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    """Z-score each feature column using rows from one subject/group."""
    features = _feature_matrix(features, name="features")
    mean = np.mean(features, axis=0, keepdims=True)
    std = nonzero_scale(np.std(features, axis=0, keepdims=True), eps=eps)
    return (features - mean) / std


def trial_zscore(features: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    """Z-score each row independently."""
    features = _feature_matrix(features, name="features")
    mean = np.mean(features, axis=1, keepdims=True)
    std = nonzero_scale(np.std(features, axis=1, keepdims=True), eps=eps)
    return (features - mean) / std


def baseline_feature_statistics(baseline_features: np.ndarray, *, eps: float = 1e-12) -> tuple[np.ndarray, np.ndarray]:
    """Return row-shaped mean and non-zero std vectors from baseline features."""
    baseline_features = _feature_matrix(baseline_features, name="baseline_features")
    mean = np.mean(baseline_features, axis=0, keepdims=True)
    std = nonzero_scale(np.std(baseline_features, axis=0, keepdims=True), eps=eps)
    return mean, std


def baseline_zscore(features: np.ndarray, baseline_mean: np.ndarray, baseline_std: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    """Z-score features using externally supplied baseline statistics."""
    features = _feature_matrix(features, name="features")
    mean = _row_vector(baseline_mean, name="baseline_mean")
    std = nonzero_scale(_row_vector(baseline_std, name="baseline_std"), eps=eps)
    _require_feature_width(features, mean, name="baseline_mean")
    _require_feature_width(features, std, name="baseline_std")
    return (features - mean) / std


def covariance_matrix(features: np.ndarray) -> np.ndarray:
    """Return a symmetric covariance matrix, with identity fallback for one row."""
    features = _feature_matrix(features, name="features")
    n_features = int(features.shape[1])
    if features.shape[0] < 2:
        return np.eye(n_features, dtype=float)
    covariance = np.asarray(np.cov(features, rowvar=False), dtype=float)
    if covariance.ndim == 0:
        covariance = covariance.reshape(1, 1)
    return 0.5 * (covariance + covariance.T)


def shrink_covariance(covariance: np.ndarray, *, shrinkage: float = 0.1) -> np.ndarray:
    """Shrink covariance toward its diagonal."""
    covariance = _square_matrix(covariance, name="covariance")
    if not 0.0 <= float(shrinkage) <= 1.0:
        raise ValueError("shrinkage must be in [0, 1].")
    diagonal = np.diag(np.diag(covariance))
    return (1.0 - float(shrinkage)) * covariance + float(shrinkage) * diagonal


def whitening_matrix(covariance: np.ndarray, *, eigenvalue_floor: float = 1e-6) -> np.ndarray:
    """Return a symmetric inverse-square-root whitening matrix."""
    covariance = _square_matrix(covariance, name="covariance")
    if float(eigenvalue_floor) < 0.0:
        raise ValueError("eigenvalue_floor must be non-negative.")
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    floor = max(float(np.max(eigenvalues)) * float(eigenvalue_floor), 1e-12)
    inverse_sqrt = 1.0 / np.sqrt(np.maximum(eigenvalues, floor))
    whitening = (eigenvectors * inverse_sqrt) @ eigenvectors.T
    return 0.5 * (whitening + whitening.T)


def baseline_whitening_matrix(baseline_features: np.ndarray, *, shrinkage: float = 0.1, eigenvalue_floor: float = 1e-6) -> np.ndarray:
    """Estimate a shrinkage covariance whitening matrix from baseline rows."""
    covariance = covariance_matrix(baseline_features)
    covariance = shrink_covariance(covariance, shrinkage=shrinkage)
    return whitening_matrix(covariance, eigenvalue_floor=eigenvalue_floor)


def baseline_whiten(
    features: np.ndarray,
    baseline_mean: np.ndarray,
    whitening: np.ndarray,
    *,
    n_channels: int | None = None,
) -> np.ndarray:
    """Center features by baseline mean and apply a whitening matrix.

    If ``n_channels`` is provided and the feature width is a multiple of it, the
    whitening matrix is applied independently to every flattened time step.
    """
    features = _feature_matrix(features, name="features")
    mean = _row_vector(baseline_mean, name="baseline_mean")
    whitening = _square_matrix(whitening, name="whitening")
    _require_feature_width(features, mean, name="baseline_mean")
    centered = features - mean
    if n_channels is None:
        _require_feature_width(centered, whitening, name="whitening")
        return centered @ whitening.T

    n_channels = int(n_channels)
    if n_channels <= 0:
        raise ValueError("n_channels must be positive.")
    if whitening.shape != (n_channels, n_channels):
        raise ValueError("whitening shape must be (n_channels, n_channels).")
    if centered.shape[1] % n_channels:
        raise ValueError("feature width must be a multiple of n_channels.")
    n_timepoints = int(centered.shape[1] // n_channels)
    matrices = centered.reshape(centered.shape[0], n_timepoints, n_channels)
    whitened = matrices @ whitening.T
    return whitened.reshape(centered.shape[0], -1)


def normalize_features(
    features: np.ndarray,
    *,
    mode: NormalizationMode = "none",
    baseline_mean: np.ndarray | None = None,
    baseline_std: np.ndarray | None = None,
    whitening: np.ndarray | None = None,
    n_channels: int | None = None,
    eps: float = 1e-12,
) -> np.ndarray:
    """Apply a named dataset-independent normalization mode."""
    features = _feature_matrix(features, name="features")
    if mode == "none":
        return features.copy()
    if mode == "subject_z":
        return subject_zscore(features, eps=eps)
    if mode == "subject_trial_z":
        return trial_zscore(features, eps=eps)
    if mode == "subject_baseline_z":
        if baseline_mean is None or baseline_std is None:
            raise ValueError("subject_baseline_z requires baseline_mean and baseline_std.")
        return baseline_zscore(features, baseline_mean, baseline_std, eps=eps)
    if mode == "subject_baseline_whiten":
        if baseline_mean is None or whitening is None:
            raise ValueError("subject_baseline_whiten requires baseline_mean and whitening.")
        return baseline_whiten(features, baseline_mean, whitening, n_channels=n_channels)
    raise ValueError(f"Unsupported normalization mode: {mode}")


def nonzero_scale(values: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    """Replace near-zero scale values by one to avoid division blow-ups."""
    values = np.asarray(values, dtype=float)
    return np.where(np.abs(values) < float(eps), 1.0, values)


def _feature_matrix(features: np.ndarray, *, name: str) -> np.ndarray:
    matrix = np.asarray(features, dtype=float)
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional feature matrix.")
    if matrix.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one row.")
    return matrix


def _row_vector(values: np.ndarray, *, name: str) -> np.ndarray:
    vector = np.asarray(values, dtype=float)
    if vector.ndim == 1:
        vector = vector[None, :]
    if vector.ndim != 2 or vector.shape[0] != 1:
        raise ValueError(f"{name} must be a row vector.")
    return vector


def _square_matrix(values: np.ndarray, *, name: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{name} must be a square matrix.")
    return matrix


def _require_feature_width(features: np.ndarray, values: np.ndarray, *, name: str) -> None:
    if values.shape[1] != features.shape[1]:
        raise ValueError(f"{name} width must match feature width.")
