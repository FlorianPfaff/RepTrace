import numpy as np

from reptrace.metrics import (
    brier_score_multiclass,
    expected_calibration_error,
)


def test_expected_calibration_error_perfect_predictions_are_zero():
    probabilities = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    labels = np.array([0, 1, 0])

    assert expected_calibration_error(probabilities, labels) == 0.0


def test_brier_score_multiclass_perfect_predictions_are_zero():
    probabilities = np.array([[1.0, 0.0], [0.0, 1.0]])
    labels = np.array([0, 1])

    assert brier_score_multiclass(probabilities, labels) == 0.0
