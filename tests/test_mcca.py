import numpy as np
import pytest

from reptrace.decoding.mcca import class_alignment_matrices, fit_class_mcca, fit_mcca


def _synthetic_subjects(seed=13):
    rng = np.random.default_rng(seed)
    latent = rng.normal(size=(24, 3))
    subjects = {}
    for subject in range(4):
        mixing = rng.normal(size=(3, 8))
        subjects[subject] = latent @ mixing + 0.05 * rng.normal(size=(24, 8))
    return subjects


def test_fit_mcca_recovers_shared_rows():
    subjects = _synthetic_subjects()
    model = fit_mcca(subjects, n_components=3, regularization=1e-5)

    transformed = [model.transform(subject, features) for subject, features in subjects.items()]
    pairwise = []
    for left in range(len(transformed)):
        for right in range(left + 1, len(transformed)):
            score = np.corrcoef(transformed[left][:, 0], transformed[right][:, 0])[0, 1]
            pairwise.append(abs(score))

    assert model.n_components == 3
    assert np.mean(pairwise) > 0.8
    assert model.group_projection is not None
    assert model.transform_group(subjects[0]).shape == (24, 3)


def test_class_alignment_matrices_class_mean():
    features = {
        "a": np.array([[1.0, 0.0], [3.0, 0.0], [0.0, 2.0], [0.0, 4.0]]),
        "b": np.array([[2.0, 1.0], [4.0, 1.0], [1.0, 3.0], [1.0, 5.0]]),
    }
    labels = {"a": np.array([1, 1, 2, 2]), "b": np.array([1, 1, 2, 2])}

    alignment = class_alignment_matrices(features, labels, sample_mode="class_mean")

    assert alignment.classes.tolist() == [1, 2]
    assert alignment.n_repetitions_per_class is None
    assert np.allclose(alignment.aligned_by_subject["a"], [[2.0, 0.0], [0.0, 3.0]])


def test_class_alignment_matrices_class_repetition():
    features = {
        "a": np.array([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]]),
        "b": np.array([[11.0], [12.0], [13.0], [14.0], [15.0], [16.0]]),
    }
    labels = {"a": np.array([1, 2, 1, 2, 1, 2]), "b": np.array([1, 2, 1, 2, 1, 2])}

    alignment = class_alignment_matrices(features, labels, sample_mode="class_repetition", n_repetitions_per_class=2)

    assert alignment.n_repetitions_per_class == 2
    assert alignment.aligned_by_subject["a"].ravel().tolist() == [1.0, 3.0, 2.0, 4.0]


def test_fit_class_mcca_rejects_missing_class():
    features = {"a": np.ones((4, 2)), "b": np.ones((4, 2))}
    labels = {"a": np.array([0, 0, 1, 1]), "b": np.array([0, 0, 0, 0])}

    with pytest.raises(ValueError, match="expected"):
        fit_class_mcca(features, labels)
