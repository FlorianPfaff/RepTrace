import numpy as np

from reptrace.decoding import make_decoder, make_grouped_calibration_cv


def test_grouped_svm_calibration_cv_preserves_group_boundaries():
    labels = np.tile([0, 1], 6)
    groups = np.repeat(np.arange(6), 2)

    splits = make_grouped_calibration_cv(labels, groups, n_splits=3)

    assert len(splits) == 3
    for inner_train_idx, calibration_idx in splits:
        assert set(groups[inner_train_idx]).isdisjoint(set(groups[calibration_idx]))
        assert set(labels[inner_train_idx]) == {0, 1}
        assert set(labels[calibration_idx]) == {0, 1}


def test_calibrated_linear_svm_accepts_grouped_calibration_cv():
    rng = np.random.default_rng(13)
    labels = np.tile([0, 1], 6)
    groups = np.repeat(np.arange(6), 2)
    features = rng.normal(size=(len(labels), 4))
    features[labels == 1, 0] += 1.0
    calibration_cv = make_grouped_calibration_cv(labels, groups, n_splits=3)

    model = make_decoder("linear_svm", max_iter=2000, calibration_cv=calibration_cv)
    model.fit(features, labels)
    probabilities = model.predict_proba(features[:3])

    assert probabilities.shape == (3, 2)
    assert probabilities.sum(axis=1).round(6).tolist() == [1.0, 1.0, 1.0]
