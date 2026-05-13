import numpy as np
import pytest

from reptrace.decoding.mcca import class_alignment_matrices, fit_class_mcca, fit_mcca


def test_fit_mcca_recovers_correlated_shared_components():
    rng = np.random.default_rng(42)
    shared = rng.normal(size=(80, 3))
    aligned = {}
    for subject in ("s1", "s2", "s3"):
        mixing = rng.normal(size=(7, 3))
        aligned[subject] = shared @ mixing.T + 0.03 * rng.normal(size=(80, 7))

    model = fit_mcca(aligned, n_components=3, regularization=1e-5)
    transformed_1 = model.transform("s1", aligned["s1"])
    transformed_2 = model.transform("s2", aligned["s2"])

    assert model.n_components == 3
    assert transformed_1.shape == (80, 3)
    assert model.group_projection is not None
    assert model.transform_group(aligned["s1"]).shape == (80, 3)
    assert abs(np.corrcoef(transformed_1[:, 0], transformed_2[:, 0])[0, 1]) > 0.85


def test_class_mean_alignment_uses_common_classes_in_order():
    features = {
        1: np.array([[0.0, 0.0], [2.0, 2.0], [4.0, 4.0], [6.0, 6.0]]),
        2: np.array([[1.0, 1.0], [3.0, 3.0], [5.0, 5.0], [7.0, 7.0]]),
    }
    labels = {1: np.array([1, 1, 2, 2]), 2: np.array([1, 1, 2, 2])}

    alignment = class_alignment_matrices(features, labels, sample_mode="class_mean")

    assert alignment.classes.tolist() == [1, 2]
    assert alignment.n_repetitions_per_class is None
    assert alignment.aligned_by_subject[1].tolist() == [[1.0, 1.0], [5.0, 5.0]]
    assert alignment.aligned_by_subject[2].tolist() == [[2.0, 2.0], [6.0, 6.0]]


def test_class_repetition_alignment_caps_repetitions():
    features = {
        "a": np.arange(12, dtype=float).reshape(6, 2),
        "b": np.arange(100, 112, dtype=float).reshape(6, 2),
    }
    labels = {"a": np.array([1, 1, 1, 2, 2, 2]), "b": np.array([1, 1, 2, 2, 2, 2])}

    alignment = class_alignment_matrices(
        features,
        labels,
        sample_mode="class_repetition",
        max_repetitions_per_class=2,
    )

    assert alignment.n_repetitions_per_class == 2
    assert alignment.aligned_by_subject["a"].shape == (4, 2)
    assert alignment.aligned_by_subject["b"].shape == (4, 2)
    assert alignment.aligned_by_subject["a"].tolist() == [[0.0, 1.0], [2.0, 3.0], [6.0, 7.0], [8.0, 9.0]]


def test_fit_class_mcca_rejects_missing_common_class():
    features = {1: np.ones((2, 2)), 2: np.ones((2, 2))}
    labels = {1: np.array([1, 2]), 2: np.array([1, 1])}

    with pytest.raises(ValueError, match="missing requested classes"):
        fit_class_mcca(features, labels, classes=[1, 2])
