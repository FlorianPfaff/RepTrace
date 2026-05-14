import numpy as np
import pytest

from reptrace.decoding.hyperalignment import (
    class_alignment_matrices,
    fit_class_hyperalignment,
    fit_hyperalignment,
    fit_projection_to_hyperalignment,
    transform_with_projection,
)


def _random_semi_orthogonal(rng, n_features, n_components):
    basis, _ = np.linalg.qr(rng.normal(size=(n_features, n_components)))
    return basis[:, :n_components]


def _mean_pairwise_correlation(matrices):
    flattened = [matrix.ravel() for matrix in matrices]
    values = []
    for i in range(len(flattened)):
        for j in range(i + 1, len(flattened)):
            values.append(np.corrcoef(flattened[i], flattened[j])[0, 1])
    return float(np.mean(values))


def test_fit_hyperalignment_aligns_synthetic_rotated_subjects():
    rng = np.random.default_rng(13)
    n_rows = 80
    n_features = 25
    n_components = 5
    template = rng.normal(size=(n_rows, n_components))
    aligned = {}
    for subject in range(4):
        projection = _random_semi_orthogonal(rng, n_features, n_components)
        offset = rng.normal(size=(1, n_features))
        aligned[subject] = template @ projection.T + offset + 0.01 * rng.normal(size=(n_rows, n_features))

    model = fit_hyperalignment(aligned, n_components=n_components, n_iterations=8)
    transformed = [model.transform(subject, aligned[subject]) for subject in aligned]

    assert model.n_components == n_components
    assert model.group_projection.shape == (n_features, n_components)
    assert _mean_pairwise_correlation(transformed) > 0.95


def test_class_alignment_supports_class_mean_and_class_repetition():
    labels = np.array([1, 1, 2, 2, 3, 3])
    features_a = np.column_stack([np.arange(6), np.arange(6) + 10])
    features_b = features_a + 100
    features = {"a": features_a, "b": features_b}
    label_map = {"a": labels, "b": labels}

    means = class_alignment_matrices(features, label_map, sample_mode="class_mean")
    repetitions = class_alignment_matrices(features, label_map, sample_mode="class_repetition", max_repetitions_per_class=1)

    assert means.aligned_by_subject["a"].shape == (3, 2)
    assert means.aligned_by_subject["a"][0, 0] == pytest.approx(0.5)
    assert repetitions.aligned_by_subject["a"].shape == (3, 2)
    assert repetitions.n_repetitions_per_class == 1
    assert repetitions.aligned_by_subject["a"][:, 0].tolist() == [0, 2, 4]


def test_fit_class_hyperalignment_rejects_missing_classes():
    with pytest.raises(ValueError, match="missing classes"):
        fit_class_hyperalignment(
            {"a": np.ones((3, 2)), "b": np.ones((2, 2))},
            {"a": np.array([1, 2, 3]), "b": np.array([1, 2])},
            sample_mode="class_mean",
        )


def test_fit_projection_to_existing_template_for_target_subject():
    rng = np.random.default_rng(19)
    n_rows = 50
    n_features = 20
    n_components = 4
    template = rng.normal(size=(n_rows, n_components))
    train = {}
    for subject in ("s1", "s2", "s3"):
        projection = _random_semi_orthogonal(rng, n_features, n_components)
        train[subject] = template @ projection.T + 0.01 * rng.normal(size=(n_rows, n_features))
    target_projection = _random_semi_orthogonal(rng, n_features, n_components)
    target = template @ target_projection.T + 0.01 * rng.normal(size=(n_rows, n_features))

    model = fit_hyperalignment(train, n_components=n_components, n_iterations=6)
    projection = fit_projection_to_hyperalignment(model, target, subject_id="target")
    transformed = transform_with_projection(projection, target)

    assert transformed.shape == (n_rows, n_components)
    assert np.corrcoef(transformed.ravel(), model.template.ravel())[0, 1] > 0.9
