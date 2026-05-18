import numpy as np
import pytest

from reptrace.decoding import (
    DECODER_CHOICES,
    REGISTRY_DECODER_CHOICES,
    STANDARD_DECODER_CHOICES,
    make_cross_validator,
    make_decoder,
    normalize_decoder_name,
    normalize_decoder_param,
    normalize_feature_preprocessor,
    normalize_pca_components,
    predict_emission_probabilities,
    resolve_decoder_param,
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

    for decoder in STANDARD_DECODER_CHOICES:
        model = make_decoder(decoder, max_iter=2000)
        model.fit(features, labels)
        probabilities = model.predict_proba(features[:3])
        assert probabilities.shape == (3, 2)


def test_decoder_choices_include_fast_shared_registry_decoders():
    assert "shrinkage-lda" in DECODER_CHOICES
    assert "correlation-prototype" in DECODER_CHOICES
    assert "multinomial-logistic" in DECODER_CHOICES
    assert "shrinkage-lda" in REGISTRY_DECODER_CHOICES
    assert normalize_decoder_name("shrinkage_lda") == "shrinkage-lda"
    assert normalize_decoder_name("mostFrequentDummy") == "mostFrequentDummy"


def test_registry_decoder_produces_probabilities():
    rng = np.random.default_rng(13)
    features = rng.normal(size=(30, 4))
    labels = np.array([0, 1] * 15)

    model = make_decoder("shrinkage-lda")
    model.fit(features, labels)
    probabilities = predict_emission_probabilities(model, features[:5])

    assert probabilities.shape == (5, 2)
    assert probabilities.sum(axis=1).round(6).tolist() == [1.0] * 5


def test_registry_decoder_supports_score_derived_emissions():
    features = np.array(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.1, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.1],
            [0.0, 0.0, 1.0],
            [0.1, 0.0, 1.0],
        ]
    )
    labels = np.array([0, 0, 1, 1, 2, 2])

    model = make_decoder("correlation-prototype", emission_mode="uncalibrated")
    model.fit(features, labels)
    probabilities = predict_emission_probabilities(model, features[:3], emission_mode="uncalibrated")

    assert probabilities.shape == (3, 3)
    assert probabilities.sum(axis=1).round(6).tolist() == [1.0] * 3


def test_calibrated_registry_decoder_can_score_svm_without_native_probabilities():
    rng = np.random.default_rng(13)
    features = rng.normal(size=(36, 4))
    labels = np.array([0, 1] * 18)

    model = make_decoder("multiclass-svm-weighted", decoder_param=0.5, emission_mode="calibrated")
    model.fit(features, labels)
    probabilities = predict_emission_probabilities(model, features[:4], emission_mode="calibrated")

    assert probabilities.shape == (4, 2)
    assert probabilities.sum(axis=1).round(6).tolist() == [1.0] * 4


def test_decoder_param_normalization_and_defaults():
    assert normalize_decoder_param("0.5") == 0.5
    assert normalize_decoder_param("5") == 5
    assert normalize_decoder_param("[5, 50]") == [5, 50]
    assert normalize_decoder_param('{"hidden_dim": 4}') == {"hidden_dim": 4}
    assert normalize_decoder_param("default") is None
    assert resolve_decoder_param("multinomial-logistic", None) == 1.0
    assert resolve_decoder_param("shrinkage-lda", "0.25") == 0.25


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
