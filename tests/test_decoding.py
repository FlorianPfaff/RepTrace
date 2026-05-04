import numpy as np

from reptrace.decoding import DECODER_CHOICES, make_cross_validator, make_decoder, normalize_decoder_name, time_windows


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


def test_normalize_decoder_name_accepts_svm_alias():
    assert normalize_decoder_name("svm") == "linear_svm"
