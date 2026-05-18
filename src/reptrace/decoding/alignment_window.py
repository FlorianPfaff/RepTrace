"""Feature-window adaptation helpers for cross-window alignment projections.

These utilities support decoding workflows that fit an alignment projection on
one feature window, then apply that projection to features extracted from a
possibly different decoding window. When the feature widths differ, the
projection and centering vector can be collapsed to channel space and reused
across the decoding window samples.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class WindowedFeatureSet(Protocol):
    """Minimal feature-set interface needed for alignment-window adaptation."""

    features: np.ndarray
    labels: np.ndarray
    n_channels: int
    n_window_samples: int


@dataclass(frozen=True)
class AlignmentWindow:
    """Resolved alignment-window parameters."""

    center: float
    size: float

    @property
    def start(self) -> float:
        """Window start time, using center-size convention."""

        return self.center - self.size / 2.0

    @property
    def stop(self) -> float:
        """Window stop time, using center-size convention."""

        return self.center + self.size / 2.0


def resolved_alignment_window(config) -> AlignmentWindow:
    """Return explicit alignment-window values, defaulting to the decoding window.

    The ``config`` object is expected to expose ``window_center`` and
    ``window_size`` attributes. Optional ``alignment_window_center`` and
    ``alignment_window_size`` attributes override the decoding window when they
    are not ``None``.
    """

    center = config.window_center if getattr(config, "alignment_window_center", None) is None else config.alignment_window_center
    size = config.window_size if getattr(config, "alignment_window_size", None) is None else config.alignment_window_size
    return AlignmentWindow(center=float(center), size=float(size))


def uses_separate_alignment_window(config) -> bool:
    """Return whether alignment and decoding windows differ."""

    alignment_window = resolved_alignment_window(config)
    return not (np.isclose(alignment_window.center, float(config.window_center)) and np.isclose(alignment_window.size, float(config.window_size)))


def validate_paired_feature_sets(decode_set: WindowedFeatureSet, alignment_set: WindowedFeatureSet, *, participant: int | None = None) -> None:
    """Validate that two feature sets refer to the same trial rows.

    The decoding and alignment feature matrices may have different column counts
    because they can represent different windows. They must, however, have the
    same row count, labels, and number of channels.
    """

    if decode_set.features.shape[0] != alignment_set.features.shape[0]:
        context = "" if participant is None else f" for participant {participant}"
        raise ValueError(f"Decoding and alignment feature rows differ{context}.")
    if not np.array_equal(np.asarray(decode_set.labels), np.asarray(alignment_set.labels)):
        context = "" if participant is None else f" for participant {participant}"
        raise ValueError(f"Decoding and alignment labels differ{context}.")
    if int(decode_set.n_channels) != int(alignment_set.n_channels):
        context = "" if participant is None else f" for participant {participant}"
        raise ValueError(f"Decoding and alignment channel counts differ{context}.")


def transform_with_alignment_projection(
    features: np.ndarray,
    *,
    decode_feature_set: WindowedFeatureSet,
    projection: np.ndarray,
    projection_feature_mean: np.ndarray,
    projection_feature_set: WindowedFeatureSet,
    feature_mean: np.ndarray | None = None,
    feature_mean_set: WindowedFeatureSet | None = None,
) -> np.ndarray:
    """Apply an alignment projection to features from a possibly different window.

    When feature widths match, this is the standard centered linear projection.
    When widths differ, the projection and centering vector are collapsed to
    channel space by averaging across the alignment-window samples, then applied
    independently to each decoding-window sample.
    """

    matrix = _feature_matrix(features, name="features")
    projection = _feature_matrix(projection, name="projection")
    projection_mean = np.asarray(projection_feature_mean, dtype=float).ravel()
    mean = projection_mean if feature_mean is None else np.asarray(feature_mean, dtype=float).ravel()
    mean_set = projection_feature_set if feature_mean is None else (feature_mean_set or decode_feature_set)

    if matrix.shape[1] == projection.shape[0]:
        if mean.shape[0] != matrix.shape[1]:
            raise ValueError(f"feature_mean length must match features columns: {mean.shape[0]} != {matrix.shape[1]}.")
        return (matrix - mean) @ projection

    channel_projection = _projection_to_channel_space(projection, projection_feature_set)
    channel_mean = _feature_mean_to_channel_space(mean, mean_set)
    return _apply_channel_projection(matrix, decode_feature_set, channel_projection, channel_mean)


def _feature_matrix(value: np.ndarray, *, name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=float)
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be a 2D matrix.")
    return matrix


def _projection_to_channel_space(projection: np.ndarray, feature_set: WindowedFeatureSet) -> np.ndarray:
    n_channels = int(feature_set.n_channels)
    if projection.shape[0] == n_channels:
        return projection
    expected = int(feature_set.n_window_samples) * n_channels
    if projection.shape[0] != expected:
        raise ValueError(f"Projection rows are incompatible with the alignment feature shape: {projection.shape[0]} != {expected}.")
    return projection.reshape(int(feature_set.n_window_samples), n_channels, projection.shape[1]).mean(axis=0)


def _feature_mean_to_channel_space(mean: np.ndarray, feature_set: WindowedFeatureSet) -> np.ndarray:
    n_channels = int(feature_set.n_channels)
    if mean.shape[0] == n_channels:
        return mean
    expected = int(feature_set.n_window_samples) * n_channels
    if mean.shape[0] != expected:
        raise ValueError(f"Feature mean is incompatible with the feature shape: {mean.shape[0]} != {expected}.")
    return mean.reshape(int(feature_set.n_window_samples), n_channels).mean(axis=0)


def _apply_channel_projection(matrix: np.ndarray, feature_set: WindowedFeatureSet, channel_projection: np.ndarray, channel_mean: np.ndarray) -> np.ndarray:
    n_channels = int(feature_set.n_channels)
    if matrix.shape[1] == n_channels:
        return (matrix - channel_mean) @ channel_projection
    expected = int(feature_set.n_window_samples) * n_channels
    if matrix.shape[1] != expected:
        raise ValueError(f"Feature columns are incompatible with the decoding feature shape: {matrix.shape[1]} != {expected}.")
    trial_channel = matrix.reshape(matrix.shape[0], int(feature_set.n_window_samples), n_channels)
    transformed = (trial_channel - channel_mean[None, None, :]) @ channel_projection
    return transformed.reshape(matrix.shape[0], -1)
