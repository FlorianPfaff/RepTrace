import numpy as np
import pytest

from reptrace.decoding import (
    DECODER_CHOICES,
    make_cross_validator,
    make_decoder,
    normalize_calibration_method,
    normalize_decoder_name,
    normalize_feature_preprocessor,
    normalize_lda_shrinkage,
    normalize_pca_components,
    normalize_regularization_c,
    predict_emission_probabilities,
    score_to_probabilities,
    time_windows,
)


def test_time_windows_returns_expected_centers():
    times = np.array([0.00, 0.01, 0.02, 0.03, 0.04])

    windows = time_windows(times, window_ms=20.0, step_ms=10.0)

    assert windows == [(0, 2, 0.005), (1, 3, 0.015), (2, 4, 0.025), (3, 5, 0.035)]


def test_make_cross_validator_supports_grouped_splits():
    labels = np.array([0, 0, 1, 1, 0, 0, 1, 1])
    groups = np.array([0, 0, 0, 0, 1, 1, 1, 1])

    splits = list(make_cross_validator(labels, groups, n_splits=2))

    assert len(splits) == 2
    for train_idx, test_idx in splits:
        assert set(groups[train_idx]).isdisjoint(set(groups[test_idx]))


def test_make_decoder_produces_probabilities_for_standard_decoders():
    rng = np.random.default_rng(13)
    features = rng.normal(size=(30, 4))
    labels = np.array([0, 1] * 15)

    for decoder in DECODER_CHOICES:
        model = make_decoder(decoder, max_iter=2000)
        model.fit(features, labels)
        probabilities = model.predict_proba(features[:3])
        assert probabilities.shape == (3, 2)


def test_make_decoder_exposes_regularization_and_shrinkage_hyperparameters():
    rng = np.random.default_rng(13)
    features = rng.normal(size=(40, 6))
    labels = np.array([0, 1] * 20)

    logistic = make_decoder("logistic", max_iter=2000, regularization_c=0.1)
    logistic.fit(features, labels)
    assert logistic.named_steps["logisticregression"].C == 0.1

    svm = make_decoder("linear_svm", max_iter=2000, emission_mode="uncalibrated", regularization_c="10")
    svm.fit(features, labels)
    assert svm.named_steps["linearsvc"].C == 10.0

    lda = make_decoder("lda", lda_shrinkage="auto")
    lda.fit(features, labels)
    assert lda.named_steps["lineardiscriminantanalysis"].solver == "lsqr"
    assert lda.named_steps["lineardiscriminantanalysis"].shrinkage == "auto"


def test_make_decoder_fits_pca_inside_probability_pipeline():
    rng = np.random.default_rng(13)
    features = rng.normal(size=(40, 8))
    labels = np.array([0, 1] * 20)

    model = make_decoder("logistic", max_iter=2000, feature_preprocessor="pca", pca_components=3)
    model.fit(features, labels)
    probabilities = model.predict_proba(features[:5])

    assert model.named_steps["pca"].n_components == 3
    assert probabilities.shape == (5, 2)
    assert probabilities.sum(axis=1).round(6).tolist() == [1.0] * 5


def test_make_decoder_accepts_pca_whiten_alias_and_fractional_components():
    rng = np.random.default_rng(13)
    features = rng.normal(size=(40, 8))
    labels = np.array([0, 1] * 20)

    model = make_decoder(
        "linear_svm",
        max_iter=2000,
        emission_mode="uncalibrated",
        feature_preprocessor="pca-whiten",
        pca_components="0.95",
    )
    model.fit(features, labels)
    probabilities = predict_emission_probabilities(model, features[:5], emission_mode="uncalibrated")

    assert model.named_steps["pca"].whiten is True
    assert probabilities.shape == (5, 2)
    assert probabilities.sum(axis=1).round(6).tolist() == [1.0] * 5


def test_pca_components_are_only_allowed_with_pca_preprocessing():
    with pytest.raises(ValueError, match="pca_components"):
        make_decoder("logistic", feature_preprocessor="none", pca_components=3)


def test_normalize_feature_preprocessor_and_components():
    assert normalize_feature_preprocessor("pca-whiten") == "pca_whiten"
    assert normalize_feature_preprocessor("identity") == "none"
    assert normalize_pca_components("3") == 3
    assert normalize_pca_components("0.95") == 0.95
    assert normalize_pca_components("auto") is None


def test_normalize_model_selection_hyperparameters():
    assert normalize_regularization_c("0.1") == 0.1
    assert normalize_lda_shrinkage("none") is None
    assert normalize_lda_shrinkage("auto") == "auto"
    assert normalize_lda_shrinkage("0.5") == 0.5
    assert normalize_calibration_method("sigmoid") == "sigmoid"
    with pytest.raises(ValueError, match="regularization_c"):
        normalize_regularization_c(0)
    with pytest.raises(ValueError, match="lda_shrinkage"):
        normalize_lda_shrinkage("2")


def test_uncalibrated_linear_svm_uses_score_derived_emissions():
    rng = np.random.default_rng(13)
    features = rng.normal(size=(30, 4))
    labels = np.array([0, 1] * 15)

    model = make_decoder("linear_svm", max_iter=2000, emission_mode="uncalibrated")
    model.fit(features, labels)
    probabilities = predict_emission_probabilities(model, features[:3], emission_mode="uncalibrated")

    assert probabilities.shape == (3, 2)
    assert probabilities.sum(axis=1).round(6).tolist() == [1.0, 1.0, 1.0]


def test_score_to_probabilities_handles_binary_scores():
    probabilities = score_to_probabilities(np.array([-1.0, 0.0, 1.0]))

    assert probabilities.shape == (3, 2)
    assert probabilities.sum(axis=1).round(6).tolist() == [1.0, 1.0, 1.0]


def test_normalize_decoder_name_accepts_svm_alias():
    assert normalize_decoder_name("svm") == "linear_svm"
