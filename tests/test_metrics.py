import numpy as np
import pandas as pd

from reptrace.metrics import (
    brier_score_multiclass,
    compare_prepost_windows,
    confusion_counts,
    expected_calibration_error,
    per_class_accuracy,
    reliability_bins,
    summarize_window_metric,
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


def test_summarize_window_metric_groups_inclusive_time_window():
    frame = pd.DataFrame(
        {
            "decoder": ["logistic", "logistic", "logistic", "svm"],
            "time": [-0.2, 0.0, 0.2, 0.0],
            "accuracy": [0.4, 0.6, 0.8, 0.7],
        }
    )

    summary = summarize_window_metric(frame, "accuracy", (-0.2, 0.0), group_columns=("decoder",))

    logistic = summary.loc[summary["decoder"] == "logistic"].iloc[0]
    assert logistic["n_rows"] == 2
    assert logistic["accuracy_mean"] == 0.5
    assert logistic["window_start"] == -0.2
    assert logistic["window_stop"] == 0.0


def test_compare_prepost_windows_reports_grouped_delta():
    frame = pd.DataFrame(
        {
            "decoder": ["logistic", "logistic", "logistic", "logistic", "svm", "svm"],
            "time": [-0.2, 0.0, 0.2, 0.4, -0.2, 0.2],
            "accuracy": [0.4, 0.6, 0.8, 0.9, 0.5, 0.75],
        }
    )

    comparison = compare_prepost_windows(frame, "accuracy", (-0.2, 0.0), (0.2, 0.4), group_columns=("decoder",))

    logistic = comparison.loc[comparison["decoder"] == "logistic"].iloc[0]
    assert logistic["n_pre_rows"] == 2
    assert logistic["n_post_rows"] == 2
    assert round(float(logistic["accuracy_post_minus_pre"]), 3) == 0.35


def test_confusion_counts_accepts_pymegdec_style_columns():
    predictions = pd.DataFrame(
        {
            "participant": ["p1", "p1", "p1", "p2"],
            "window": [0.1, 0.1, 0.1, 0.1],
            "true_stimulus": ["cat", "cat", "dog", "cat"],
            "predicted_stimulus": ["cat", "dog", "dog", "cat"],
        }
    )

    counts = confusion_counts(
        predictions,
        true_column="true_stimulus",
        predicted_column="predicted_stimulus",
        group_columns=("window",),
    )

    cat_as_cat = counts[(counts["true_label"] == "cat") & (counts["predicted_label"] == "cat")].iloc[0]
    assert cat_as_cat["count"] == 2


def test_per_class_accuracy_counts_participants():
    predictions = pd.DataFrame(
        {
            "participant": ["p1", "p1", "p1", "p2"],
            "task": ["animate", "animate", "animate", "animate"],
            "true_label": ["animal", "animal", "device", "animal"],
            "predicted_label": ["animal", "device", "device", "animal"],
        }
    )

    summary = per_class_accuracy(predictions, participant_column="participant", group_columns=("task",))

    animal = summary.loc[summary["true_label"] == "animal"].iloc[0]
    assert animal["n_trials"] == 3
    assert animal["n_correct"] == 2
    assert round(float(animal["accuracy"]), 3) == 0.667
    assert animal["n_participants"] == 2
