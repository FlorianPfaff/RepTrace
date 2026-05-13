import numpy as np
import pandas as pd

from reptrace.metrics import (
    brier_score_multiclass,
    category_confusion_enrichment,
    category_confusion_matrix,
    compare_prepost_windows,
    confusion_counts,
    confusion_pair_summary,
    exact_one_sided_sign_p_value,
    expected_calibration_error,
    most_confused_class_pairs,
    one_sided_signflip_p_value,
    per_class_accuracy,
    ranked_accuracy_metrics,
    reliability_bins,
    subject_level_signflip_summary,
    summarize_window_metric,
    true_label_ranks,
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


def test_confusion_pair_summary_reports_bidirectional_pair_structure():
    predictions = pd.DataFrame(
        {
            "participant": [1, 2, 1, 2, 1, 2],
            "true_label": [1, 1, 2, 1, 2, 3],
            "predicted_label": [2, 2, 1, 1, 2, 2],
            "decoder": ["logistic"] * 6,
        }
    )
    metadata = pd.DataFrame(
        [
            {"stimulus": "1", "name": "apple", "category": "food"},
            {"stimulus": "2", "name": "pear", "category": "food"},
            {"stimulus": "3", "name": "hammer", "category": "tool"},
        ]
    )

    pairs = confusion_pair_summary(
        predictions,
        participant_column="participant",
        group_columns=("decoder",),
        metadata=metadata,
        category_columns=("category",),
    )
    top_pair = most_confused_class_pairs(
        predictions,
        participant_column="participant",
        metadata=metadata,
        category_columns=("category",),
        top_n=1,
    )

    first = pairs.iloc[0]
    assert first["decoder"] == "logistic"
    assert first["label_a"] == 1
    assert first["label_b"] == 2
    assert first["a_to_b_count"] == 2
    assert first["b_to_a_count"] == 1
    assert first["total_confusions"] == 3
    assert first["symmetric_confusion_count"] == 1
    assert first["n_confused_participants"] == 2
    assert first["a_to_b_participants"] == 2
    assert first["b_to_a_participants"] == 1
    assert first["a_to_b_rate"] == 2 / 3
    assert first["b_to_a_rate"] == 1 / 2
    assert first["expected_a_to_b_count"] == 1.5
    assert first["expected_b_to_a_count"] == 0.25
    assert first["pair_confusion_lift"] == 3 / 1.75
    assert first["total_confusion_excess"] == 1.25
    assert first["pair_standardized_residual"] == 1.25 / np.sqrt(1.75)
    assert first["label_a_category"] == "food"
    assert first["label_b_category"] == "food"
    assert bool(first["same_category"])
    assert top_pair.iloc[0]["total_confusions"] == 3


def test_category_confusion_summaries_report_enrichment_and_category_matrix():
    predictions = pd.DataFrame(
        {
            "participant": [1, 2, 1, 2, 3, 3],
            "true_label": [1, 1, 2, 3, 4, 4],
            "predicted_label": [2, 2, 1, 2, 3, 4],
            "decoder": ["logistic"] * 6,
        }
    )
    metadata = pd.DataFrame(
        [
            {"stimulus": "1", "name": "apple", "category": "food"},
            {"stimulus": "2", "name": "pear", "category": "food"},
            {"stimulus": "3", "name": "hammer", "category": "tool"},
            {"stimulus": "4", "name": "saw", "category": "tool"},
        ]
    )

    enrichment = category_confusion_enrichment(
        predictions,
        metadata=metadata,
        category_columns=("category",),
        participant_column="participant",
        group_columns=("decoder",),
        n_permutations=128,
        random_state=0,
    )
    matrix = category_confusion_matrix(
        predictions,
        metadata=metadata,
        category_columns=("category",),
        participant_column="participant",
    )

    row = enrichment.iloc[0]
    assert row["decoder"] == "logistic"
    assert row["category_column"] == "category"
    assert row["n_errors_with_category"] == 5
    assert row["same_category_errors"] == 4
    assert row["expected_same_category_errors"] == 14 / 5
    assert row["same_category_lift"] == 4 / (14 / 5)
    assert row["n_participants_with_category_errors"] == 3
    assert row["n_participants_with_same_category_errors"] == 3
    assert 0.0 <= row["same_category_permutation_p_value"] <= 1.0

    food_to_food = matrix[(matrix["true_category"] == "food") & (matrix["predicted_category"] == "food")].iloc[0]
    assert bool(food_to_food["same_category"])
    assert food_to_food["count"] == 3
    assert food_to_food["expected_count"] == 12 / 5
    assert food_to_food["category_confusion_lift"] == 3 / (12 / 5)


def test_true_label_ranks_supports_generic_labels_and_stable_ties():
    ranks = true_label_ranks(
        ["cat", "dog", "bird", "missing"],
        np.array(
            [
                [0.1, 0.9, 0.0],
                [0.8, 0.8, 0.1],
                [0.2, 0.3, 0.7],
                [0.2, 0.3, 0.7],
            ]
        ),
        ["cat", "dog", "bird"],
    )

    np.testing.assert_equal(ranks, np.array([2.0, 2.0, 1.0, np.nan]))


def test_ranked_accuracy_metrics_reports_top_k_and_rank_summaries():
    metrics = ranked_accuracy_metrics(
        ["cat", "dog", "bird", "cat"],
        np.array(
            [
                [0.1, 0.9, 0.0],
                [0.8, 0.7, 0.1],
                [0.2, 0.3, 0.7],
                [0.2, 0.3, 0.7],
            ]
        ),
        ["cat", "dog", "bird"],
    )

    np.testing.assert_equal(metrics["true_label_ranks"], np.array([2.0, 2.0, 1.0, 3.0]))
    assert metrics["top2_accuracy"] == 0.75
    assert metrics["top3_accuracy"] == 1.0
    assert metrics["mean_true_label_rank"] == 2.0
    assert metrics["median_true_label_rank"] == 2.0


def test_exact_one_sided_sign_p_value_counts_positive_subject_effects():
    assert exact_one_sided_sign_p_value([0.1, 0.2, 0.3]) == 1 / 8
    assert exact_one_sided_sign_p_value([0.1] * 23) == 1 / (2**23)
    assert np.isnan(exact_one_sided_sign_p_value([np.nan]))


def test_one_sided_signflip_p_value_uses_exact_enumeration_for_small_samples():
    p_value = one_sided_signflip_p_value([0.1, 0.2], n_permutations=10_000)

    assert p_value == 0.25
    assert one_sided_signflip_p_value([-0.2, 0.1]) == 1.0


def test_subject_level_signflip_summary_accepts_scalar_and_per_subject_chance():
    summary = subject_level_signflip_summary(
        [0.7, 0.6, 0.4, np.nan],
        chance=[0.5, 0.5, 0.5, 0.5],
        n_permutations=128,
        random_state=3,
    )

    assert summary["n_subjects"] == 3
    assert summary["n_above_chance"] == 2
    assert round(float(summary["value_mean"]), 3) == 0.567
    assert round(float(summary["effect_mean"]), 3) == 0.067
    assert summary["one_sided_exact_sign_p_value"] == 0.5
    assert 0.0 <= float(summary["one_sided_signflip_p_value"]) <= 1.0
