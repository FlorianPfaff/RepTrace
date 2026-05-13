from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ProcrustesTransform:
    """Orthogonal channel-space transform from one subject/group to a template."""

    source_center: np.ndarray
    target_center: np.ndarray
    rotation: np.ndarray


@dataclass(frozen=True)
class ClassPatternAlignmentResult:
    """Aligned feature matrices and metadata from class-pattern Procrustes alignment."""

    aligned_features: tuple[np.ndarray, ...]
    transforms: tuple[ProcrustesTransform, ...]
    common_classes: tuple[Any, ...]
    template: np.ndarray


def class_pattern_procrustes_alignment(
    feature_matrices: Sequence[np.ndarray],
    label_vectors: Sequence[Sequence[Any] | np.ndarray],
    *,
    n_channels: int | None = None,
    common_classes: Sequence[Any] | None = None,
    n_template_iterations: int = 3,
) -> ClassPatternAlignmentResult:
    """Align feature matrices by orthogonal Procrustes on class-average channel patterns.

    ``feature_matrices`` are trial-by-feature arrays. If ``n_channels`` is
    provided and feature width is larger than ``n_channels``, features are
    interpreted as flattened time-by-channel matrices and alignment is applied
    independently at every time step.
    """
    feature_matrices = tuple(feature_matrices)
    label_vectors = tuple(label_vectors)
    if len(feature_matrices) != len(label_vectors):
        raise ValueError("feature_matrices and label_vectors must have the same length.")
    features = tuple(_feature_matrix(matrix, name="feature_matrices") for matrix in feature_matrices)
    labels = tuple(_label_vector(vector, expected_length=matrix.shape[0]) for vector, matrix in zip(label_vectors, features))
    if len(features) < 1:
        raise ValueError("At least one feature matrix is required.")
    n_channels = _resolve_n_channels(features, n_channels)
    common_classes = _common_classes(labels) if common_classes is None else tuple(common_classes)
    if len(common_classes) < 2:
        identity = np.eye(n_channels, dtype=float)
        transforms = tuple(ProcrustesTransform(np.zeros(n_channels), np.zeros(n_channels), identity) for _ in features)
        return ClassPatternAlignmentResult(features, transforms, tuple(common_classes), np.empty((0, n_channels), dtype=float))

    class_patterns = tuple(_class_channel_patterns(matrix, vector, common_classes, n_channels=n_channels) for matrix, vector in zip(features, labels))
    transforms, template = fit_procrustes_transforms(class_patterns, n_template_iterations=n_template_iterations)
    aligned = tuple(apply_procrustes_transform(matrix, transform, n_channels=n_channels) for matrix, transform in zip(features, transforms))
    return ClassPatternAlignmentResult(aligned, transforms, tuple(common_classes), template)


def fit_procrustes_transforms(
    class_patterns: Sequence[np.ndarray],
    *,
    n_template_iterations: int = 3,
) -> tuple[tuple[ProcrustesTransform, ...], np.ndarray]:
    """Fit transforms that align class-pattern matrices to a shared template."""
    patterns = tuple(_pattern_matrix(pattern) for pattern in class_patterns)
    if not patterns:
        raise ValueError("At least one class-pattern matrix is required.")
    shapes = {pattern.shape for pattern in patterns}
    if len(shapes) != 1:
        raise ValueError("All class-pattern matrices must have the same shape.")
    template = np.mean(np.stack(patterns, axis=0), axis=0)
    for _ in range(int(n_template_iterations)):
        transforms = tuple(procrustes_transform(pattern, template) for pattern in patterns)
        aligned_patterns = tuple(apply_procrustes_transform_to_patterns(pattern, transform) for pattern, transform in zip(patterns, transforms))
        template = np.mean(np.stack(aligned_patterns, axis=0), axis=0)
    transforms = tuple(procrustes_transform(pattern, template) for pattern in patterns)
    return transforms, template


def procrustes_transform(source: np.ndarray, target: np.ndarray) -> ProcrustesTransform:
    """Fit an orthogonal Procrustes transform from source patterns to target patterns."""
    source = _pattern_matrix(source)
    target = _pattern_matrix(target)
    if source.shape != target.shape:
        raise ValueError("source and target patterns must have the same shape.")
    source_center = np.mean(source, axis=0)
    target_center = np.mean(target, axis=0)
    source_centered = source - source_center
    target_centered = target - target_center
    cross_covariance = source_centered.T @ target_centered
    left, _singular_values, right_t = np.linalg.svd(cross_covariance, full_matrices=False)
    rotation = left @ right_t
    return ProcrustesTransform(source_center=source_center, target_center=target_center, rotation=rotation)


def apply_procrustes_transform(features: np.ndarray, transform: ProcrustesTransform, *, n_channels: int | None = None) -> np.ndarray:
    """Apply a channel-space Procrustes transform to trial features."""
    features = _feature_matrix(features, name="features")
    n_channels = _resolve_n_channels((features,), n_channels)
    feature_tensor = features_as_trial_channel_matrix(features, n_channels=n_channels)
    aligned = (feature_tensor - transform.source_center) @ transform.rotation + transform.target_center
    if feature_tensor.shape[1] == 1:
        return aligned[:, 0, :]
    return aligned.reshape(features.shape[0], -1)


def apply_procrustes_transform_to_patterns(patterns: np.ndarray, transform: ProcrustesTransform) -> np.ndarray:
    """Apply a Procrustes transform to class-pattern rows."""
    return (np.asarray(patterns, dtype=float) - transform.source_center) @ transform.rotation + transform.target_center


def features_as_trial_channel_matrix(features: np.ndarray, *, n_channels: int) -> np.ndarray:
    """View features as trial-by-time-by-channel for channel-space operations."""
    features = _feature_matrix(features, name="features")
    n_channels = int(n_channels)
    if n_channels <= 0:
        raise ValueError("n_channels must be positive.")
    if features.shape[1] == n_channels:
        return features[:, None, :]
    if features.shape[1] % n_channels:
        raise ValueError("feature width must equal or be a multiple of n_channels.")
    n_timepoints = int(features.shape[1] // n_channels)
    return features.reshape(features.shape[0], n_timepoints, n_channels)


def _class_channel_patterns(features: np.ndarray, labels: np.ndarray, common_classes: Sequence[Any], *, n_channels: int) -> np.ndarray:
    channel_features = features_as_trial_channel_matrix(features, n_channels=n_channels)
    patterns = []
    for class_label in common_classes:
        class_features = channel_features[labels == class_label]
        if class_features.size == 0:
            raise ValueError(f"Missing class {class_label!r} while fitting Procrustes alignment.")
        patterns.append(np.mean(class_features, axis=(0, 1)))
    return np.vstack(patterns)


def _common_classes(label_vectors: Sequence[np.ndarray]) -> tuple[Any, ...]:
    label_sets = [set(labels.tolist()) for labels in label_vectors]
    if not label_sets:
        return tuple()
    return tuple(sorted(set.intersection(*label_sets), key=lambda value: (str(type(value)), str(value))))


def _resolve_n_channels(feature_matrices: Sequence[np.ndarray], n_channels: int | None) -> int:
    widths = {int(matrix.shape[1]) for matrix in feature_matrices}
    if len(widths) != 1:
        raise ValueError("All feature matrices must have the same width.")
    if n_channels is None:
        return next(iter(widths))
    n_channels = int(n_channels)
    if n_channels <= 0:
        raise ValueError("n_channels must be positive.")
    for width in widths:
        if width != n_channels and width % n_channels:
            raise ValueError("Feature width must equal or be a multiple of n_channels.")
    return n_channels


def _feature_matrix(features: np.ndarray, *, name: str) -> np.ndarray:
    matrix = np.asarray(features, dtype=float)
    if matrix.ndim != 2:
        raise ValueError(f"{name} must contain two-dimensional feature matrices.")
    if matrix.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one row.")
    return matrix


def _label_vector(labels: Sequence[Any] | np.ndarray, *, expected_length: int) -> np.ndarray:
    vector = np.asarray(labels).ravel()
    if vector.ndim != 1:
        raise ValueError("label vectors must be one-dimensional.")
    if vector.shape[0] != expected_length:
        raise ValueError("label vector length must match feature rows.")
    return vector


def _pattern_matrix(patterns: np.ndarray) -> np.ndarray:
    matrix = np.asarray(patterns, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("class-pattern matrices must be two-dimensional.")
    if matrix.shape[0] < 2:
        raise ValueError("class-pattern matrices must contain at least two class rows.")
    return matrix
