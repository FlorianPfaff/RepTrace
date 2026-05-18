import numpy as np
import pytest

from reptrace.decoding.mcca import class_alignment_matrices, fit_class_mcca, fit_mcca
from reptrace.decoding.mcca_target import class_alignment_matrix, fit_target_mcca_projection


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


def test_class_alignment_matrix_uses_explicit_class_order():
    features = np.array([[1.0, 0.0], [3.0, 0.0], [0.0, 2.0], [0.0, 4.0]])
    labels = np.array([1, 1, 2, 2])

    aligned = class_alignment_matrix(features, labels, classes=np.array([2, 1]), sample_mode="class_mean")

    assert np.allclose(aligned, [[0.0, 3.0], [2.0, 0.0]])
    with pytest.raises(ValueError, match="absent"):
        class_alignment_matrix(features, labels, classes=np.array([1, 3]), sample_mode="class_mean")


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


def test_fit_target_mcca_projection_projects_held_out_subject_to_template():
    subjects = _synthetic_subjects()
    training_subjects = {subject: features for subject, features in subjects.items() if subject != 3}
    model = fit_mcca(training_subjects, n_components=3, regularization=1e-5)

    projection = fit_target_mcca_projection(subjects[3], model, regularization=1e-5)
    transformed = projection.transform(subjects[3])

    correlations = [
        abs(np.corrcoef(transformed[:, component], model.component_scores[:, component])[0, 1])
        for component in range(model.n_components)
    ]
    assert projection.projection.shape == (8, 3)
    assert transformed.shape == model.component_scores.shape
    assert np.mean(correlations) > 0.8


def test_fit_target_mcca_projection_rejects_row_mismatch():
    with pytest.raises(ValueError, match="template rows"):
        fit_target_mcca_projection(np.ones((3, 2)), np.ones((4, 2)), regularization=1e-6)
