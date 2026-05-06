import sys
import warnings

import numpy as np
import pytest
from sklearn.exceptions import ConvergenceWarning

from reptrace.decoding.classifiers import (
    CLASSIFIER_REGISTRY,
    get_default_classifier_param,
    prediction_scores,
    should_use_default_classifier_param,
    train_binary_svm,
    train_classifier,
    train_gradient_boosting,
    train_lasso_logistic,
)


@pytest.fixture
def multiclass_data():
    features = np.array(
        [
            [0.0, 0.0],
            [0.0, 0.2],
            [1.0, 1.0],
            [1.0, 1.2],
            [2.0, 2.0],
            [2.0, 2.2],
        ]
    )
    labels = np.array([0, 0, 1, 1, 2, 2])
    return features, labels


def test_registry_contains_shared_sklearn_classifiers():
    assert set(CLASSIFIER_REGISTRY) == {
        "always1Dummy",
        "gradient-boosting",
        "knn",
        "mostFrequentDummy",
        "multiclass-svm",
        "multiclass-svm-weighted",
        "random-forest",
        "scikit-mlp",
    }


def test_registry_trains_fast_sklearn_classifiers(multiclass_data):
    features, labels = multiclass_data
    classifier_params = {
        "always1Dummy": None,
        "gradient-boosting": 5,
        "knn": 1,
        "mostFrequentDummy": None,
        "multiclass-svm": 1.0,
        "multiclass-svm-weighted": 1.0,
        "random-forest": 5,
        "scikit-mlp": (5, 50),
    }

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        for classifier, classifier_param in classifier_params.items():
            model = train_classifier(features, labels, classifier, classifier_param, random_state=13)
            assert len(model.predict(features)) == len(labels)


def test_random_state_reproduces_stochastic_classifier_predictions(multiclass_data):
    features, labels = multiclass_data

    model_a = train_classifier(features, labels, "random-forest", 5, random_state=7)
    model_b = train_classifier(features, labels, "random-forest", 5, random_state=7)

    np.testing.assert_array_equal(model_a.predict(features), model_b.predict(features))


def test_default_classifier_params_include_legacy_binary_helpers():
    assert get_default_classifier_param("multiclass-svm") == 0.5
    assert get_default_classifier_param("svm-binary") == 0.5
    assert get_default_classifier_param("binary-svm") == 0.5
    assert get_default_classifier_param("lasso") == 0.005
    assert should_use_default_classifier_param(np.nan)
    assert not should_use_default_classifier_param(None)
    assert not should_use_default_classifier_param({"hidden_dim": 10})


def test_binary_helper_trainers_fit_binary_labels():
    features = np.array([[-2.0, 0.0], [-1.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    labels = np.array([False, False, True, True])

    helpers = (
        lambda: train_binary_svm(features, labels, 1.0, random_state=13),
        lambda: train_lasso_logistic(features, labels, 0.1, random_state=13),
        lambda: train_gradient_boosting(features, labels, 5, random_state=13),
    )

    for build_model in helpers:
        model = build_model()
        assert len(model.predict(features)) == len(labels)


def test_prediction_scores_supports_decision_function_and_probabilities(multiclass_data):
    features, labels = multiclass_data
    svm = train_classifier(features, labels, "multiclass-svm", 1.0)
    knn = train_classifier(features, labels, "knn", 1)

    assert prediction_scores(svm, features).shape == (len(labels),)
    assert prediction_scores(knn, features).shape == (len(labels),)


def test_prediction_scores_returns_nan_without_score_api():
    class LabelOnlyModel:
        def predict(self, features):
            return np.zeros(len(features), dtype=int)

    scores = prediction_scores(LabelOnlyModel(), np.zeros((3, 2)))

    assert np.isnan(scores).tolist() == [True, True, True]


def test_optional_non_sklearn_dependencies_are_not_imported(multiclass_data):
    features, labels = multiclass_data

    train_classifier(features, labels, "multiclass-svm", 1.0)

    assert "xgboost" not in sys.modules
    assert "torch" not in sys.modules
    assert "pytorch_lightning" not in sys.modules


def test_unsupported_classifier_error_lists_supported_names(multiclass_data):
    features, labels = multiclass_data

    with pytest.raises(ValueError, match="Supported classifiers"):
        train_classifier(features, labels, "unknown", None)
