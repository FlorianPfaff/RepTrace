import numpy as np
import pytest

from reptrace.decoding.alignment import (
    apply_procrustes_transform,
    class_pattern_procrustes_alignment,
    features_as_trial_channel_matrix,
    procrustes_transform,
)
from reptrace.decoding.normalization import (
    baseline_feature_statistics,
    baseline_whiten,
    baseline_whitening_matrix,
    baseline_zscore,
    covariance_matrix,
    normalize_features,
    shrink_covariance,
    subject_zscore,
    trial_zscore,
    whitening_matrix,
)


def test_subject_zscore_normalizes_columns_and_keeps_constant_columns_zero():
    features = np.array([[1.0, 2.0, 5.0], [2.0, 4.0, 5.0], [3.0, 6.0, 5.0]])

    normalized = subject_zscore(features)

    np.testing.assert_allclose(normalized.mean(axis=0), np.zeros(3), atol=1e-12)
    np.testing.assert_allclose(normalized.std(axis=0)[:2], np.ones(2), atol=1e-12)
    np.testing.assert_allclose(normalized[:, 2], np.zeros(3), atol=1e-12)


def test_trial_zscore_normalizes_rows():
    features = np.array([[1.0, 2.0, 3.0], [4.0, 4.0, 4.0]])

    normalized = trial_zscore(features)

    assert normalized[0].mean() == pytest.approx(0.0)
    assert normalized[0].std() == pytest.approx(1.0)
    np.testing.assert_allclose(normalized[1], np.zeros(3), atol=1e-12)


def test_baseline_statistics_and_zscore_use_supplied_reference():
    baseline = np.array([[1.0, 2.0], [3.0, 6.0]])
    features = np.array([[2.0, 4.0], [4.0, 8.0]])
    mean, std = baseline_feature_statistics(baseline)

    normalized = baseline_zscore(features, mean, std)

    np.testing.assert_allclose(mean, np.array([[2.0, 4.0]]))
    np.testing.assert_allclose(std, np.array([[1.0, 2.0]]))
    np.testing.assert_allclose(normalized, np.array([[0.0, 0.0], [2.0, 2.0]]))


def test_covariance_shrinkage_and_whitening_are_symmetric():
    baseline = np.array([[1.0, 0.0], [2.0, 0.5], [3.0, 1.0]])

    covariance = covariance_matrix(baseline)
    shrunk = shrink_covariance(covariance, shrinkage=0.25)
    whitening = whitening_matrix(shrunk)
    estimated = baseline_whitening_matrix(baseline, shrinkage=0.25)

    np.testing.assert_allclose(covariance, covariance.T)
    np.testing.assert_allclose(shrunk, shrunk.T)
    np.testing.assert_allclose(whitening, whitening.T)
    np.testing.assert_allclose(estimated, whitening)


def test_baseline_whiten_supports_direct_and_flattened_channel_features():
    whitening = np.array([[2.0, 0.0], [0.0, 0.5]])
    direct_features = np.array([[2.0, 6.0]])
    direct_mean = np.array([[1.0, 2.0]])
    flat_features = np.array([[2.0, 6.0, 3.0, 8.0]])
    flat_mean = np.array([[1.0, 2.0, 1.0, 2.0]])

    direct = baseline_whiten(direct_features, direct_mean, whitening)
    flattened = baseline_whiten(flat_features, flat_mean, whitening, n_channels=2)

    np.testing.assert_allclose(direct, np.array([[2.0, 2.0]]))
    np.testing.assert_allclose(flattened, np.array([[2.0, 2.0, 4.0, 3.0]]))


def test_normalize_features_dispatches_modes_and_requires_baseline_inputs():
    features = np.array([[1.0, 2.0], [3.0, 4.0]])

    np.testing.assert_allclose(normalize_features(features, mode="none"), features)
    assert normalize_features(features, mode="subject_trial_z").shape == features.shape
    with pytest.raises(ValueError, match="baseline_mean"):
        normalize_features(features, mode="subject_baseline_z")


def test_procrustes_transform_maps_rotated_patterns_to_target():
    target = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    rotation = np.array([[0.0, -1.0], [1.0, 0.0]])
    source = target @ rotation + np.array([[2.0, -1.0]])

    transform = procrustes_transform(source, target)
    aligned = (source - transform.source_center) @ transform.rotation + transform.target_center

    np.testing.assert_allclose(aligned, target, atol=1e-12)


def test_class_pattern_procrustes_alignment_reduces_cross_subject_pattern_distance():
    labels = np.array([0, 0, 1, 1, 2, 2])
    base_patterns = np.array([[2.0, 0.0], [0.0, 2.0], [-2.0, 0.0]])
    features_a = np.repeat(base_patterns, 2, axis=0)
    rotation = np.array([[0.0, -1.0], [1.0, 0.0]])
    features_b = features_a @ rotation + np.array([[3.0, -2.0]])

    before = _class_patterns(features_a, labels) - _class_patterns(features_b, labels)
    result = class_pattern_procrustes_alignment((features_a, features_b), (labels, labels), n_channels=2)
    after = _class_patterns(result.aligned_features[0], labels) - _class_patterns(result.aligned_features[1], labels)

    assert result.common_classes == (0, 1, 2)
    assert len(result.transforms) == 2
    assert np.linalg.norm(after) < np.linalg.norm(before) * 0.1


def test_class_pattern_procrustes_alignment_supports_flattened_time_channel_features():
    labels = np.array([0, 0, 1, 1])
    features = np.array(
        [
            [1.0, 2.0, 3.0, 4.0],
            [1.5, 2.5, 3.5, 4.5],
            [-1.0, -2.0, -3.0, -4.0],
            [-1.5, -2.5, -3.5, -4.5],
        ]
    )

    tensor = features_as_trial_channel_matrix(features, n_channels=2)
    result = class_pattern_procrustes_alignment((features, features), (labels, labels), n_channels=2)
    transformed = apply_procrustes_transform(features, result.transforms[0], n_channels=2)

    assert tensor.shape == (4, 2, 2)
    assert result.aligned_features[0].shape == features.shape
    assert transformed.shape == features.shape


def _class_patterns(features, labels):
    return np.vstack([features[labels == label].mean(axis=0) for label in sorted(set(labels.tolist()))])
