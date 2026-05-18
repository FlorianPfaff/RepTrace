from pathlib import Path

import pandas as pd

from reptrace.results import aggregate_time_decode_csvs, aggregate_time_decode_results, peak_metric_rows, summarize_metric_table


def _result_frame(subject: str, offset: float = 0.0) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "subject": [subject, subject, subject, subject],
            "fold": [0, 1, 0, 1],
            "time": [0.1, 0.1, 0.2, 0.2],
            "accuracy": [0.6 + offset, 0.8 + offset, 0.7 + offset, 0.9 + offset],
            "log_loss": [0.5, 0.4, 0.45, 0.35],
            "brier": [0.3, 0.2, 0.25, 0.15],
            "ece": [0.1, 0.2, 0.15, 0.25],
        }
    )


def test_aggregate_time_decode_results_averages_folds_then_subjects():
    results = pd.concat([_result_frame("s1"), _result_frame("s2", offset=0.1)], ignore_index=True)

    aggregated = aggregate_time_decode_results(results)

    assert aggregated["n_subjects"].tolist() == [2, 2]
    assert aggregated["accuracy_mean"].round(3).tolist() == [0.75, 0.85]


def test_aggregate_time_decode_results_weights_folds_by_test_size():
    results = pd.DataFrame(
        {
            "subject": ["s1", "s1", "s2", "s2"],
            "fold": [0, 1, 0, 1],
            "time": [0.1, 0.1, 0.1, 0.1],
            "accuracy": [1.0, 0.0, 0.2, 0.4],
            "log_loss": [1.0, 3.0, 2.0, 4.0],
            "brier": [0.1, 0.3, 0.2, 0.4],
            "ece": [0.05, 0.15, 0.1, 0.3],
            "n_test": [9, 1, 1, 3],
        }
    )

    aggregated = aggregate_time_decode_results(results)

    assert aggregated["n_subjects"].tolist() == [2]
    assert aggregated["accuracy_mean"].round(3).tolist() == [0.625]
    assert aggregated["log_loss_mean"].round(3).tolist() == [2.35]
    assert aggregated["brier_mean"].round(3).tolist() == [0.235]
    assert aggregated["ece_mean"].round(3).tolist() == [0.155]


def test_aggregate_time_decode_csvs_uses_filename_as_subject(tmp_path: Path):
    first = tmp_path / "sub-01.csv"
    second = tmp_path / "sub-02.csv"
    _result_frame("ignored").drop(columns="subject").to_csv(first, index=False)
    _result_frame("ignored", offset=0.1).drop(columns="subject").to_csv(second, index=False)

    out = tmp_path / "summary.csv"
    aggregated = aggregate_time_decode_csvs([first, second], out_path=out)

    assert out.exists()
    assert aggregated["n_subjects"].tolist() == [2, 2]


def test_aggregate_time_decode_results_keeps_decoder_groups_separate():
    first = _result_frame("s1")
    first["decoder"] = "logistic"
    second = _result_frame("s1", offset=0.1)
    second["decoder"] = "lda"

    aggregated = aggregate_time_decode_results(pd.concat([first, second], ignore_index=True))

    assert aggregated["decoder"].tolist() == ["lda", "lda", "logistic", "logistic"]
    assert aggregated["accuracy_mean"].round(3).tolist() == [0.8, 0.9, 0.7, 0.8]


def test_aggregate_time_decode_results_keeps_emission_modes_separate():
    calibrated = _result_frame("s1")
    calibrated["decoder"] = "linear_svm"
    calibrated["emission_mode"] = "calibrated"
    uncalibrated = _result_frame("s1", offset=0.1)
    uncalibrated["decoder"] = "linear_svm"
    uncalibrated["emission_mode"] = "uncalibrated"

    aggregated = aggregate_time_decode_results(pd.concat([calibrated, uncalibrated], ignore_index=True))

    assert aggregated["emission_mode"].tolist() == ["calibrated", "calibrated", "uncalibrated", "uncalibrated"]
    assert aggregated["accuracy_mean"].round(3).tolist() == [0.7, 0.8, 0.8, 0.9]


def test_aggregate_time_decode_results_keeps_preprocessing_and_tuning_separate():
    baseline = _result_frame("s1")
    baseline["decoder"] = "logistic"
    baseline["feature_preprocessor"] = "none"
    baseline["tuned_hyperparameters"] = False
    tuned = _result_frame("s1", offset=0.1)
    tuned["decoder"] = "logistic"
    tuned["feature_preprocessor"] = "pca_whiten"
    tuned["pca_components"] = "0.95"
    tuned["tuned_hyperparameters"] = True
    tuned["tuning_cv_splits"] = 2
    tuned["tuning_scoring"] = "balanced_accuracy"
    tuned["tuning_c_grid"] = "0.1|1.0|10.0"

    aggregated = aggregate_time_decode_results(pd.concat([baseline, tuned], ignore_index=True))

    assert aggregated["feature_preprocessor"].tolist() == ["none", "none", "pca_whiten", "pca_whiten"]
    assert aggregated["tuned_hyperparameters"].tolist() == [False, False, True, True]
    assert aggregated["accuracy_mean"].round(3).tolist() == [0.7, 0.8, 0.8, 0.9]


def test_aggregate_time_decode_results_keeps_temporal_train_windows_separate():
    early = _result_frame("s1")
    early["decoder"] = "logistic"
    early["temporal_mode"] = "train_window_ensemble"
    early["temporal_train_window_start"] = 0.12
    early["temporal_train_window_stop"] = 0.25
    late = _result_frame("s1", offset=0.1)
    late["decoder"] = "logistic"
    late["temporal_mode"] = "train_window_ensemble"
    late["temporal_train_window_start"] = 0.20
    late["temporal_train_window_stop"] = 0.35

    aggregated = aggregate_time_decode_results(pd.concat([early, late], ignore_index=True))

    assert aggregated["temporal_train_window_start"].tolist() == ["0.12", "0.12", "0.2", "0.2"]
    assert aggregated["temporal_train_window_stop"].tolist() == ["0.25", "0.25", "0.35", "0.35"]
    assert aggregated["accuracy_mean"].round(3).tolist() == [0.7, 0.8, 0.8, 0.9]


def test_summarize_metric_table_reports_participants_chance_and_scaled_values():
    frame = pd.DataFrame(
        {
            "decoder": ["logistic", "logistic", "logistic", "svm"],
            "window": [0.1, 0.1, 0.1, 0.1],
            "participant": ["s1", "s2", "s2", "s1"],
            "accuracy": [0.6, 0.8, 0.7, 0.55],
            "chance": [0.5, 0.5, 0.5, 0.5],
        }
    )

    summary = summarize_metric_table(frame, "accuracy", ("decoder", "window"), participant_column="participant", chance_column="chance", scale=100.0)

    logistic = summary.loc[summary["decoder"] == "logistic"].iloc[0]
    assert logistic["n_rows"] == 3
    assert logistic["n_participants"] == 2
    assert round(float(logistic["accuracy_mean"]), 3) == 70.0
    assert logistic["chance_mean"] == 50.0
    assert logistic["accuracy_above_chance_count"] == 3
    assert round(float(logistic["accuracy_minus_chance_mean"]), 3) == 20.0


def test_peak_metric_rows_breaks_ties_toward_preferred_time():
    frame = pd.DataFrame(
        {
            "decoder": ["logistic", "logistic", "logistic", "svm"],
            "participant": ["s1", "s1", "s1", "s1"],
            "time": [-0.1, 0.1, 0.3, 0.2],
            "accuracy": [0.8, 0.8, 0.7, 0.9],
        }
    )

    peaks = peak_metric_rows(frame, "accuracy", ("decoder", "participant"), prefer_time=0.0)

    logistic = peaks.loc[peaks["decoder"] == "logistic"].iloc[0]
    assert logistic["time"] == -0.1
    assert logistic["accuracy"] == 0.8
    assert round(float(logistic["peak_distance_to_prefer_time"]), 3) == 0.1


def test_summarize_metric_table_reports_rich_chance_and_permutation_fields():
    frame = pd.DataFrame(
        {
            "decoder": ["logistic", "logistic", "logistic"],
            "window": [0.1, 0.1, 0.1],
            "participant": ["s1", "s2", "s3"],
            "accuracy": [0.40, 0.30, 0.80],
            "chance_accuracy": [0.50, 0.25, None],
            "n_validation_classes": [2, 4, 2],
            "permutation_p_value": [0.04, 0.20, 0.006],
        }
    )

    summary = summarize_metric_table(
        frame,
        "accuracy",
        ("decoder", "window"),
        participant_column="participant",
        chance_column="chance_accuracy",
        percent_scale=100.0,
        chance_percent_column="chance_percent",
        chance_class_columns=("chance_classes", "n_chance_classes", "n_validation_classes"),
        permutation_p_column="permutation_p_value",
        zero_singleton_dispersion=True,
    )

    row = summary.iloc[0]
    assert row["n_participants"] == 3
    assert round(float(row["percent_mean"]), 3) == 50.0
    assert round(float(row["chance_accuracy_mean"]), 3) == 0.417
    assert round(float(row["chance_percent"]), 3) == 41.667
    assert row["chance_accuracy_min"] == 0.25
    assert row["chance_accuracy_max"] == 0.50
    assert round(float(row["chance_classes_mean"]), 3) == 2.667
    assert row["chance_classes_counts"] == "2:2;4:1"
    assert row["accuracy_above_chance_count"] == 2
    assert row["n_with_permutation"] == 3
    assert row["n_significant_p_0.05"] == 2
    assert row["n_significant_p_0.01"] == 1


def test_summarize_metric_table_can_zero_singleton_dispersion():
    frame = pd.DataFrame({"decoder": ["logistic"], "accuracy": [0.75], "chance": [0.5]})

    summary = summarize_metric_table(
        frame,
        "accuracy",
        "decoder",
        chance_column="chance",
        zero_singleton_dispersion=True,
    )

    row = summary.iloc[0]
    assert row["accuracy_std"] == 0.0
    assert row["accuracy_sem"] == 0.0
