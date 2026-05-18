import numpy as np
import pytest

from reptrace.decoding.hyperalignment_initialization import (
    HYPERALIGNMENT_INITIALIZATION_MODES,
    fit_class_hyperalignment,
    fit_hyperalignment,
)


def _aligned_subjects():
    rng = np.random.default_rng(0)
    return {
        "s1": rng.normal(size=(6, 4)),
        "s2": rng.normal(size=(6, 4)) + 0.1,
        "s3": rng.normal(size=(6, 4)) - 0.1,
    }


def test_mean_initialized_hyperalignment_fits_common_space():
    aligned = _aligned_subjects()

    model = fit_hyperalignment(aligned, n_components=3, n_iterations=2, initialization="mean")

    assert HYPERALIGNMENT_INITIALIZATION_MODES == ("pca", "mean")
    assert model.n_components == 3
    assert model.template.shape == (6, 3)
    assert model.group_feature_mean.shape == (4,)
    assert model.group_projection.shape == (4, 3)
    assert model.transform("s1", aligned["s1"]).shape == (6, 3)


def test_class_hyperalignment_accepts_mean_initialization():
    aligned = _aligned_subjects()
    features = {subject: np.vstack([matrix, matrix + 0.01]) for subject, matrix in aligned.items()}
    labels = {subject: np.array([1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3]) for subject in aligned}

    model, alignment = fit_class_hyperalignment(
        features,
        labels,
        sample_mode="class_repetition",
        n_repetitions_per_class=2,
        n_components=3,
        n_iterations=2,
        initialization="mean",
    )

    assert alignment.sample_mode == "class_repetition"
    assert alignment.n_repetitions_per_class == 2
    assert alignment.repetition_selection == "random"
    assert alignment.repetition_seed == 0
    assert model.template.shape == (6, 3)


def test_pca_initialization_still_allows_different_feature_dimensions():
    rng = np.random.default_rng(1)

    model = fit_hyperalignment({"s1": rng.normal(size=(6, 4)), "s2": rng.normal(size=(6, 5))}, n_components=3, n_iterations=2)

    assert model.n_components == 3
    assert model.group_feature_mean is None
    assert model.group_projection is None


def test_mean_initialization_requires_matching_feature_dimensions():
    rng = np.random.default_rng(2)

    with pytest.raises(ValueError, match="same feature dimension"):
        fit_hyperalignment({"s1": rng.normal(size=(6, 4)), "s2": rng.normal(size=(6, 5))}, n_components=3, n_iterations=2, initialization="mean")


def test_unknown_hyperalignment_initialization_rejected():
    with pytest.raises(ValueError, match="Unsupported hyperalignment initialization"):
        fit_hyperalignment(_aligned_subjects(), initialization="unsupported")
