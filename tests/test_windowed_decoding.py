import numpy as np
import pytest

from reptrace.decoding.windowed import (
    fit_window_model,
    permutation_p_from_accuracy,
    predict_window_model,
    score_windowed_decoding,
)


class _SignClassifier:
    def __init__(self):
        self.negative_label = None
        self.positive_label = None

    def fit(self, features, labels):
        labels = np.asarray(labels)
        features = np.asarray(features)
        self.negative_label = _majority_label(labels[features[:, 0] < 0])
        self.positive_label = _majority_label(labels[features[:, 0] >= 0])
        return self

    def predict(self, features):
        features = np.asarray(features)
        return np.where(features[:, 0] < 0, self.negative_label, self.positive_label)

    def decision_function(self, features):
        return np.asarray(features)[:, 0]


def _majority_label(labels):
    values, counts = np.unique(labels, return_counts=True)
    return values[np.argmax(counts)]


def _fit_sign_classifier(features, labels):
    return _SignClassifier().fit(features, labels)


def test_fit_window_model_applies_pca_and_predicts_validation_features():
    train_features = np.array(
        [
            [-2.0, -2.0],
            [-1.0, -1.0],
            [1.0, 1.0],
            [2.0, 2.0],
        ]
    )
    train_labels = np.array([0, 0, 1, 1])
    validation_features = np.array([[-1.5, -1.5], [1.5, 1.5]])

    model_bundle = fit_window_model(
        train_features,
        train_labels,
        fit_model=_fit_sign_classifier,
        components_pca=1,
        train_window=(0.125, 0.225),
    )
    predictions, scores = predict_window_model(model_bundle, validation_features)

    assert predictions.tolist() == [0, 1]
    assert scores.shape == (2,)
    assert model_bundle.train_window == (0.125, 0.225)
    assert model_bundle.actual_components_pca == 1
    assert model_bundle.pca_coeff.shape == (2, 1)
    assert model_bundle.explained_variance_percent == pytest.approx(100.0)


def test_score_windowed_decoding_returns_accuracy_predictions_and_permutation_p():
    result = score_windowed_decoding(
        train_features=np.array([[-2.0], [-1.0], [1.0], [2.0]]),
        train_labels=np.array([0, 0, 1, 1]),
        validation_features=np.array([[-1.5], [1.5]]),
        validation_labels=np.array([0, 1]),
        fit_model=_fit_sign_classifier,
        components_pca=float("inf"),
        n_permutations=4,
        permutation_rng=np.random.default_rng(13),
    )

    assert result.accuracy == 1.0
    assert result.predictions.tolist() == [0, 1]
    assert result.permutation_accuracy.shape == (4,)
    assert result.permutation_p_value == permutation_p_from_accuracy(1.0, result.permutation_accuracy)


def test_score_windowed_decoding_rejects_mismatched_validation_labels():
    with pytest.raises(ValueError, match="validation_labels length must match feature rows"):
        score_windowed_decoding(
            train_features=np.array([[-1.0], [1.0]]),
            train_labels=np.array([0, 1]),
            validation_features=np.array([[-1.0], [1.0]]),
            validation_labels=np.array([0]),
            fit_model=_fit_sign_classifier,
        )


def test_permutation_p_from_accuracy_uses_plus_one_correction():
    assert permutation_p_from_accuracy(0.75, np.array([0.1, 0.5, 0.8])) == pytest.approx(0.5)
