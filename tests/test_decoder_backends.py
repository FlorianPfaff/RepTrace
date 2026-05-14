from __future__ import annotations

import numpy as np

from reptrace.decoding import DecoderPredictionResult, SklearnDecoderBackend, make_decoder_backend


def _feature_matrix() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(13)
    labels = np.array([0, 1] * 12)
    features = rng.normal(size=(24, 4))
    features[labels == 1, 0] += 2.0
    return features, labels


def test_make_decoder_backend_returns_sklearn_backend() -> None:
    backend = make_decoder_backend("logistic", max_iter=2000)

    assert isinstance(backend, SklearnDecoderBackend)
    assert backend.name == "sklearn"
    assert backend.decoder == "logistic"
    assert backend.emission_mode == "calibrated"


def test_sklearn_backend_fit_predict_returns_probabilities() -> None:
    features, labels = _feature_matrix()
    train_idx = np.arange(16)
    test_idx = np.arange(16, 24)
    backend = make_decoder_backend("logistic", max_iter=2000)

    result = backend.fit_predict(features, labels, train_idx, test_idx)

    assert isinstance(result, DecoderPredictionResult)
    assert result.probabilities.shape == (8, 2)
    assert result.predictions.shape == (8,)
    assert result.test_labels.tolist() == labels[test_idx].tolist()
    assert result.probabilities.sum(axis=1).round(6).tolist() == [1.0] * 8


def test_sklearn_backend_builds_canonical_observation_table() -> None:
    features, labels = _feature_matrix()
    train_idx = np.arange(16)
    test_idx = np.arange(16, 24)
    backend = make_decoder_backend("logistic", max_iter=2000)

    table = backend.fit_predict_observation_table(
        features,
        labels,
        train_idx,
        test_idx,
        class_names=["negative", "positive"],
        fold=0,
        time=0.125,
        split_id="unit-test-split",
        seed=13,
    )

    assert table.validate().is_valid
    assert table.frame["backend"].unique().tolist() == ["sklearn"]
    assert table.frame["split_id"].unique().tolist() == ["unit-test-split"]
    assert table.frame["model_hash"].astype(str).str.len().min() > 0
