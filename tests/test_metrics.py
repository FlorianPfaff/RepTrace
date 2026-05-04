import numpy as np

from reptrace.metrics import (
    brier_score_multiclass,
    expected_calibration_error,
    reliability_bins,
)


def test_expected_calibration_error_perfect_predictions_are_zero():
    probabilities = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    labels = np.array([0, 1, 0])

    assert expected_calibration_error(probabilities, labels) == 0.0


def test_brier_score_multiclass_perfect_predictions_are_zero():
    probabilities = np.array([[1.0, 0.0], [0.0, 1.0]])
    labels = np.array([0, 1])

    assert brier_score_multiclass(probabilities, labels) == 0.0


def test_reliability_bins_reports_confidence_accuracy_gap():
    probabilities = np.array([[0.8, 0.2], [0.7, 0.3], [0.4, 0.6]])
    labels = np.array([0, 1, 1])

    bins = reliability_bins(probabilities, labels, n_bins=2)

    assert bins[0]["n_samples"] == 0
    assert bins[1]["n_samples"] == 3
    assert round(float(bins[1]["accuracy"]), 3) == 0.667
    assert round(float(bins[1]["confidence"]), 3) == 0.700
    assert round(float(bins[1]["gap"]), 3) == -0.033
