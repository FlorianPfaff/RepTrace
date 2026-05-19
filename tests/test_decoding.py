import numpy as np
import pytest

from reptrace.decoding import (
    BUILTIN_DECODER_CHOICES,
    DECODER_CHOICES,
    make_cross_validator,
    make_decoder,
    make_tuning_cross_validator,
    normalize_decoder_name,
    normalize_feature_preprocessor,
    normalize_pca_components,
    parse_c_grid,
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


def test_make_tuning_cross_validator_caps_to_feasible_grouped_splits():
    labels = np.array([0, 0, 1, 1, 0, 0, 1, 1])
    groups = np.array([0, 0, 0, 0, 1, 1, 1, 1])

    splits = make_tuning_cross_validator(labels, groups, n_splits=5)

    assert len(splits) == 2
    for train_idx, test_idx in splits:
        assert set(groups[train_idx]).isdisjoint(set(groups[test_idx]))


def test_make_decoder_produces_probabilities_for_standard_decoders():
    rng = np.random.default_rng(13)
    features = rng.normal(size=(30, 4))
    labels = np.array([0, 1] * 15)

    for decoder in BUILTIN_DECODER_CHOICES:
        model = make_decoder(decoder, max_iter=2000)
        model.fit(features, labels)
        probabilities = model.predict_proba(features[:3])
        assert probabilities.shape == (3, 2)


def test_decoder_choices_expose_classifier_registry_entries():
    assert "correlation-prototype" in DECODER_CHOICES
    assert "multinomial-logistic" in DECODER_CHOICES
    assert "random-forest" in DECODER_CHOICES
    assert normalize_decoder_name("correlation_prototype") == "correlation-prototype"
    assert normalize_decoder_name("multiclass-svm-weighted") == "multiclass-svm-weighted"


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


def test_gaussian_nb_produces_probabilities():
    rng = np.random.default_rng(23)
    features = rng.normal(size=(30, 8))
    labels = np.array([0, 1] * 15)

    model = make_decoder("naive-bayes")
    model.fit(features, labels)
    probabilities = model.predict_proba(features[:3])

    assert model.named_steps["gaussiannb"].var_smoothing == 1e-9
    assert probabilities.shape == (3, 2)
    assert probabilities.sum(axis=1).round(6).tolist() == [1.0, 1.0, 1.0]


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


def test_make_decoder_can_tune_regularization_with_inner_cv():
    rng = np.random.default_rng(13)
    features = rng.normal(size=(24, 4))
    labels = np.array([0, 1] * 12)

    model = make_decoder(
        "logistic",
        max_iter=2000,
        tune_hyperparameters=True,
        tuning_cv=2,
        tuning_c_grid=(0.1, 1.0),
    )
    model.fit(features, labels)
    probabilities = model.predict_proba(features[:3])

    assert probabilities.shape == (3, 2)
    assert model.best_params_["logisticregression__C"] in {0.1, 1.0}


def test_tuned_lda_compares_svd_and_shrinkage_variants():
    rng = np.random.default_rng(13)
    features = rng.normal(size=(24, 4))
    labels = np.array([0, 1] * 12)

    model = make_decoder("lda", tune_hyperparameters=True, tuning_cv=2)
    model.fit(features, labels)

    assert "lineardiscriminantanalysis__solver" in model.best_params_
    assert model.predict_proba(features[:3]).shape == (3, 2)


def test_shrinkage_lda_uses_lsqr_auto_shrinkage():
    rng = np.random.default_rng(17)
    features = rng.normal(size=(30, 12))
    labels = np.array([0, 1] * 15)

    model = make_decoder("shrinkage-lda")
    model.fit(features, labels)

    lda = model.named_steps["lineardiscriminantanalysis"]
    assert lda.solver == "lsqr"
    assert lda.shrinkage == "auto"
    assert model.predict_proba(features[:3]).shape == (3, 2)


def test_tuned_shrinkage_lda_selects_shrinkage_strength():
    rng = np.random.default_rng(19)
    features = rng.normal(size=(30, 8))
    labels = np.array([0, 1] * 15)

    model = make_decoder("lda-shrinkage", tune_hyperparameters=True, tuning_cv=2)
    model.fit(features, labels)

    assert "lineardiscriminantanalysis__shrinkage" in model.best_params_
    assert model.predict_proba(features[:3]).shape == (3, 2)


def test_tuned_gaussian_nb_selects_var_smoothing():
    rng = np.random.default_rng(29)
    features = rng.normal(size=(30, 8))
    labels = np.array([0, 1] * 15)

    model = make_decoder("gaussian-nb", tune_hyperparameters=True, tuning_cv=2)
    model.fit(features, labels)

    assert "gaussiannb__var_smoothing" in model.best_params_
    assert model.predict_proba(features[:3]).shape == (3, 2)


def test_parse_c_grid_accepts_comma_separated_values():
    assert parse_c_grid("0.1,1,10") == (0.1, 1.0, 10.0)


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
    assert normalize_decoder_name("naive-bayes") == "gaussian_nb"
    assert normalize_decoder_name("svm") == "linear_svm"
    assert normalize_decoder_name("lda-shrinkage") == "shrinkage_lda"
