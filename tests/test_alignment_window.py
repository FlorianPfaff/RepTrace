from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from reptrace.decoding.alignment_window import (
    resolved_alignment_window,
    transform_with_alignment_projection,
    uses_separate_alignment_window,
    validate_paired_feature_sets,
)


@dataclass(frozen=True)
class DummyConfig:
    window_center: float = 0.3
    window_size: float = 0.1
    alignment_window_center: float | None = None
    alignment_window_size: float | None = None


@dataclass(frozen=True)
class DummyFeatureSet:
    features: np.ndarray
    labels: np.ndarray
    n_channels: int
    n_window_samples: int
    feature_order: str = "channel_time"


def test_resolved_alignment_window_defaults_to_decoding_window() -> None:
    window = resolved_alignment_window(DummyConfig())

    assert window.center == pytest.approx(0.3)
    assert window.size == pytest.approx(0.1)
    assert window.start == pytest.approx(0.25)
    assert window.stop == pytest.approx(0.35)
    assert not uses_separate_alignment_window(DummyConfig())


def test_resolved_alignment_window_uses_explicit_alignment_values() -> None:
    config = DummyConfig(alignment_window_center=0.5, alignment_window_size=0.2)
    window = resolved_alignment_window(config)

    assert window.center == pytest.approx(0.5)
    assert window.size == pytest.approx(0.2)
    assert window.start == pytest.approx(0.4)
    assert window.stop == pytest.approx(0.6)
    assert uses_separate_alignment_window(config)


def test_validate_paired_feature_sets_allows_different_window_widths() -> None:
    decode = DummyFeatureSet(np.zeros((3, 4)), np.array([1, 2, 3]), n_channels=2, n_window_samples=2)
    alignment = DummyFeatureSet(np.zeros((3, 6)), np.array([1, 2, 3]), n_channels=2, n_window_samples=3)

    validate_paired_feature_sets(decode, alignment)


def test_validate_paired_feature_sets_rejects_label_mismatch_with_context() -> None:
    decode = DummyFeatureSet(np.zeros((3, 4)), np.array([1, 2, 3]), n_channels=2, n_window_samples=2)
    alignment = DummyFeatureSet(np.zeros((3, 6)), np.array([1, 9, 3]), n_channels=2, n_window_samples=3)

    with pytest.raises(ValueError, match="labels differ for participant 7"):
        validate_paired_feature_sets(decode, alignment, participant=7)


def test_transform_with_alignment_projection_uses_direct_projection_when_widths_match() -> None:
    feature_set = DummyFeatureSet(np.zeros((2, 3)), np.array([1, 2]), n_channels=3, n_window_samples=1)
    features = np.array([[2.0, 3.0, 4.0], [3.0, 5.0, 7.0]])
    projection = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    mean = np.array([1.0, 1.0, 1.0])

    transformed = transform_with_alignment_projection(
        features,
        decode_feature_set=feature_set,
        projection=projection,
        projection_feature_mean=mean,
        projection_feature_set=feature_set,
    )

    np.testing.assert_allclose(transformed, (features - mean) @ projection)


def test_transform_with_alignment_projection_defaults_to_mne_channel_time_order() -> None:
    decode = DummyFeatureSet(np.zeros((1, 4)), np.array([1]), n_channels=2, n_window_samples=2)
    alignment = DummyFeatureSet(np.zeros((1, 6)), np.array([1]), n_channels=2, n_window_samples=3)
    # MNE-style flattening of (channels, time): [c0t0, c0t1, c1t0, c1t1].
    features = np.array([[4.0, 5.0, 7.0, 8.0]])
    projection = np.array(
        [
            [1.0, 0.0],
            [3.0, 0.0],
            [5.0, 0.0],
            [0.0, 2.0],
            [0.0, 4.0],
            [0.0, 6.0],
        ]
    )
    projection_mean = np.array([1.0, 3.0, 5.0, 2.0, 4.0, 6.0])

    transformed = transform_with_alignment_projection(
        features,
        decode_feature_set=decode,
        projection=projection,
        projection_feature_mean=projection_mean,
        projection_feature_set=alignment,
    )

    np.testing.assert_allclose(transformed, np.array([[3.0, 12.0, 6.0, 16.0]]))


def test_transform_with_alignment_projection_supports_explicit_time_channel_order() -> None:
    decode = DummyFeatureSet(
        np.zeros((1, 4)),
        np.array([1]),
        n_channels=2,
        n_window_samples=2,
        feature_order="time_channel",
    )
    alignment = DummyFeatureSet(
        np.zeros((1, 6)),
        np.array([1]),
        n_channels=2,
        n_window_samples=3,
        feature_order="time_channel",
    )
    # Legacy flattening of (time, channels): [t0c0, t0c1, t1c0, t1c1].
    features = np.array([[4.0, 7.0, 5.0, 8.0]])
    projection = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [3.0, 0.0],
            [0.0, 3.0],
            [5.0, 0.0],
            [0.0, 5.0],
        ]
    )
    projection_mean = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

    transformed = transform_with_alignment_projection(
        features,
        decode_feature_set=decode,
        projection=projection,
        projection_feature_mean=projection_mean,
        projection_feature_set=alignment,
    )

    np.testing.assert_allclose(transformed, np.array([[3.0, 9.0, 6.0, 12.0]]))


def test_transform_with_alignment_projection_rejects_invalid_feature_order() -> None:
    decode = DummyFeatureSet(
        np.zeros((1, 4)),
        np.array([1]),
        n_channels=2,
        n_window_samples=2,
        feature_order="channels-first",
    )
    alignment = DummyFeatureSet(np.zeros((1, 6)), np.array([1]), n_channels=2, n_window_samples=3)

    with pytest.raises(ValueError, match="feature_order"):
        transform_with_alignment_projection(
            np.zeros((1, 4)),
            decode_feature_set=decode,
            projection=np.zeros((6, 2)),
            projection_feature_mean=np.zeros(6),
            projection_feature_set=alignment,
        )


def test_transform_with_alignment_projection_rejects_incompatible_projection_width() -> None:
    decode = DummyFeatureSet(np.zeros((1, 4)), np.array([1]), n_channels=2, n_window_samples=2)
    alignment = DummyFeatureSet(np.zeros((1, 6)), np.array([1]), n_channels=2, n_window_samples=3)

    with pytest.raises(ValueError, match="Projection rows are incompatible"):
        transform_with_alignment_projection(
            np.zeros((1, 4)),
            decode_feature_set=decode,
            projection=np.zeros((5, 2)),
            projection_feature_mean=np.zeros(6),
            projection_feature_set=alignment,
        )
