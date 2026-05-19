import numpy as np
import pytest

from reptrace.decoding import (
    DECODER_CHOICES,
    make_cross_validator,
    make_decoder,
    make_tuning_cross_validator,
    normalize_anova_select_percentile,
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

    for decoder in DECODER_CHOICES:
        model = make_decoder(decoder, max_iter=2000)
        model.fit(features, labels)
        probabilities = model.predict_proba(features[:3])
        assert probabilities.shape == (3, 2)


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


def test_sparse_logistic_uses_l1_saga_regularization():
    rng = np.random.default_rng(13)
    features = rng.normal(size=(30, 6))
    labels = np.array([0, 1] * 15)

    model = make_decoder("l1-logistic", max_iter=2000)
    model.fit(features, labels)
    probabilities = model.predict_proba(features[:3])

    classifier = model.named_steps["logisticregression"]
    assert classifier.l1_ratio == 1.0
    assert classifier.solver == "saga"
    assert classifier.class_weight == "balanced"
    assert probabilities.shape == (3, 2)


def test_elastic_net_logistic_uses_saga_with_default_l1_ratio():
    rng = np.random.default_rng(13)
    features = rng.normal(size=(30, 6))
    labels = np.array([0, 1] * 15)

    model = make_decoder("elasticnet-logistic", max_iter=2000)
    model.fit(features, labels)
    probabilities = model.predict_proba(features[:3])

    classifier = model.named_steps["logisticregression"]
    assert classifier.solver == "saga"
    assert classifier.l1_ratio == 0.5
    assert classifier.class_weight == "balanced"
    assert probabilities.shape == (3, 2)


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


def test_make_decoder_fits_anova_selection_inside_probability_pipeline():
    rng = np.random.default_rng(13)
    features = rng.normal(size=(40, 20))
    labels = np.array([0, 1] * 20)

    model = make_decoder("logistic", max_iter=2000, feature_preprocessor="anova-select", pca_components=25)
    model.fit(features, labels)
    probabilities = model.predict_proba(features[:5])

    selector = model.named_steps["selectpercentile"]
    assert selector.percentile == 25
    assert probabilities.shape == (5, 2)
    assert probabilities.sum(axis=1).round(6).tolist() == [1.0] * 5


def test_pca_components_are_only_allowed_with_pca_preprocessing():
    with pytest.raises(ValueError, match="pca_components"):
        make_decoder("logistic", feature_preprocessor="none", pca_components=3)


def test_normalize_feature_preprocessor_and_components():
    assert normalize_feature_preprocessor("pca-whiten") == "pca_whiten"
    assert normalize_feature_preprocessor("select-percentile") == "anova_select"
    assert normalize_feature_preprocessor("identity") == "none"
    assert normalize_pca_components("3") == 3
    assert normalize_pca_components("0.95") == 0.95
    assert normalize_pca_components("auto") is None
    assert normalize_anova_select_percentile(None) == 20
    assert normalize_anova_select_percentile("25") == 25


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


def test_tuned_anova_select_searches_percentile_inside_inner_cv():
    rng = np.random.default_rng(17)
    features = rng.normal(size=(24, 20))
    labels = np.array([0, 1] * 12)

    model = make_decoder(
        "logistic",
        max_iter=2000,
        feature_preprocessor="anova-select",
        pca_components=20,
        tune_hyperparameters=True,
        tuning_cv=2,
        tuning_c_grid=(0.1, 1.0),
    )
    model.fit(features, labels)

    assert model.predict_proba(features[:3]).shape == (3, 2)
    assert model.best_params_["logisticregression__C"] in {0.1, 1.0}
    assert model.best_params_["selectpercentile__percentile"] in {10, 20, 40, 60}


def test_tuned_lda_compares_svd_and_shrinkage_variants():
    rng = np.random.default_rng(13)
    features = rng.normal(size=(24, 4))
    labels = np.array([0, 1] * 12)

    model = make_decoder("lda", tune_hyperparameters=True, tuning_cv=2)
    model.fit(features, labels)

    assert "lineardiscriminantanalysis__solver" in model.best_params_
    assert model.predict_proba(features[:3]).shape == (3, 2)


def test_tuned_sparse_logistic_tunes_c_with_inner_cv():
    rng = np.random.default_rng(13)
    features = rng.normal(size=(24, 6))
    labels = np.array([0, 1] * 12)

    model = make_decoder(
        "sparse-logreg",
        max_iter=2000,
        tune_hyperparameters=True,
        tuning_cv=2,
        tuning_c_grid=(0.1, 1.0),
    )
    model.fit(features, labels)

    assert model.predict_proba(features[:3]).shape == (3, 2)
    assert model.best_params_["logisticregression__C"] in {0.1, 1.0}


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


def test_tuned_elastic_net_logistic_searches_c_and_l1_ratio():
    rng = np.random.default_rng(23)
    features = rng.normal(size=(24, 6))
    labels = np.array([0, 1] * 12)

    model = make_decoder(
        "logistic-elastic-net",
        max_iter=2000,
        tune_hyperparameters=True,
        tuning_cv=2,
        tuning_c_grid=(0.1, 1.0),
    )
    model.fit(features, labels)

    assert model.predict_proba(features[:3]).shape == (3, 2)
    assert model.best_params_["logisticregression__C"] in {0.1, 1.0}
    assert model.best_params_["logisticregression__l1_ratio"] in {0.15, 0.5, 0.85}


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
    assert normalize_decoder_name("svm") == "linear_svm"
    assert normalize_decoder_name("l1-logistic") == "sparse_logistic"
    assert normalize_decoder_name("elasticnet-logistic") == "elastic_net_logistic"
    assert normalize_decoder_name("lda-shrinkage") == "shrinkage_lda"
