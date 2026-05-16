from pathlib import Path

import pandas as pd
import pytest

from reptrace.paired_stats import paired_decoder_statistics, sign_flip_p_value, subject_decoder_metrics


def _write_decoder_csv(path: Path, subject: str, decoder: str, baseline: float, effect: float, *, emission_mode: str | None = None) -> None:
    rows = []
    for fold in [0, 1]:
        fold_rows = [
            {"subject": subject, "decoder": decoder, "fold": fold, "time": -0.05, "accuracy": baseline + 0.01 * fold, "log_loss": 0.8, "brier": 0.6, "ece": 0.2},
            {"subject": subject, "decoder": decoder, "fold": fold, "time": 0.15, "accuracy": effect + 0.01 * fold, "log_loss": 0.7 - effect / 10, "brier": 0.5 - effect / 10, "ece": 0.15 - effect / 20},
        ]
        if emission_mode is not None:
            for row in fold_rows:
                row["emission_mode"] = emission_mode
        rows.extend(fold_rows)
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_exact_ece_observations(path: Path, *, subject: str = "sub-01", decoder: str = "logistic") -> None:
    rows = []
    for time in [-0.05, 0.15]:
        rows.extend(
            [
                {"subject": subject, "decoder": decoder, "emission_mode": "calibrated", "fold": 0, "time": time, "true_label": 0, "prob_class_0": 0.6, "prob_class_1": 0.4},
                {"subject": subject, "decoder": decoder, "emission_mode": "calibrated", "fold": 1, "time": time, "true_label": 1, "prob_class_0": 0.6, "prob_class_1": 0.4},
            ]
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def test_subject_decoder_metrics_computes_baseline_corrected_effect(tmp_path: Path):
    csv_path = tmp_path / "sub-01_logistic_time_decode.csv"
    _write_decoder_csv(csv_path, "sub-01", "logistic", baseline=0.50, effect=0.62)
    metrics = subject_decoder_metrics([csv_path], baseline_window=(-0.1, 0.0), effect_window=(0.1, 0.2))
    row = metrics.iloc[0]
    assert row["decoder"] == "logistic"
    assert row["emission_mode"] == "calibrated"
    assert row["subject"] == "sub-01"
    assert round(row["baseline_accuracy"], 3) == 0.505
    assert round(row["effect_accuracy"], 3) == 0.625
    assert round(row["effect_minus_baseline"], 3) == 0.120
    assert "effect_ece" not in metrics.columns


def test_subject_decoder_metrics_weights_fold_means_by_n_test(tmp_path: Path):
    csv_path = tmp_path / "sub-01_logistic_time_decode.csv"
    pd.DataFrame(
        [
            {"subject": "sub-01", "decoder": "logistic", "fold": 0, "time": -0.05, "accuracy": 0.0, "log_loss": 1.0, "brier": 1.0, "ece": 1.0, "n_test": 1},
            {"subject": "sub-01", "decoder": "logistic", "fold": 1, "time": -0.05, "accuracy": 1.0, "log_loss": 0.0, "brier": 0.0, "ece": 0.0, "n_test": 3},
            {"subject": "sub-01", "decoder": "logistic", "fold": 0, "time": 0.15, "accuracy": 0.2, "log_loss": 1.0, "brier": 1.0, "ece": 1.0, "n_test": 1},
            {"subject": "sub-01", "decoder": "logistic", "fold": 1, "time": 0.15, "accuracy": 0.6, "log_loss": 0.0, "brier": 0.0, "ece": 0.0, "n_test": 3},
        ]
    ).to_csv(csv_path, index=False)
    metrics = subject_decoder_metrics([csv_path], baseline_window=(-0.1, 0.0), effect_window=(0.1, 0.2))
    row = metrics.iloc[0]
    assert row["baseline_accuracy"] == pytest.approx(0.75)
    assert row["effect_accuracy"] == pytest.approx(0.5)
    assert row["effect_log_loss"] == pytest.approx(0.25)


def test_subject_decoder_metrics_uses_exact_ece_observations(tmp_path: Path):
    csv_path = tmp_path / "sub-01_logistic_time_decode.csv"
    observations_path = tmp_path / "sub-01_logistic_observations.csv"
    rows = []
    for fold in [0, 1]:
        for time in [-0.05, 0.15]:
            rows.append(
                {
                    "subject": "sub-01",
                    "decoder": "logistic",
                    "emission_mode": "calibrated",
                    "fold": fold,
                    "time": time,
                    "accuracy": 0.6,
                    "log_loss": 0.7,
                    "brier": 0.5,
                    "ece": 0.9,
                    "n_test": 1,
                }
            )
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    _write_exact_ece_observations(observations_path)

    metrics = subject_decoder_metrics(
        [csv_path],
        baseline_window=(-0.1, 0.0),
        effect_window=(0.1, 0.2),
        observation_csv_paths=[observations_path],
        ece_bins=2,
    )

    assert metrics.loc[0, "effect_ece"] == pytest.approx(0.1)


def test_paired_decoder_statistics_compares_shared_subjects(tmp_path: Path):
    paths = []
    for idx in range(4):
        subject = f"sub-{idx + 1:02d}"
        logistic = tmp_path / f"{subject}_logistic_time_decode.csv"
        lda = tmp_path / f"{subject}_lda_time_decode.csv"
        _write_decoder_csv(logistic, subject, "logistic", baseline=0.50, effect=0.64 + idx * 0.01)
        _write_decoder_csv(lda, subject, "lda", baseline=0.54, effect=0.60 + idx * 0.01)
        paths.extend([logistic, lda])
    subject_metrics = subject_decoder_metrics(paths, baseline_window=(-0.1, 0.0), effect_window=(0.1, 0.2))
    stats = paired_decoder_statistics(subject_metrics, n_permutations=10_000)
    assert "effect_ece" not in set(stats["metric"])
    effect_row = stats[stats["metric"] == "effect_minus_baseline"].iloc[0]
    assert effect_row["emission_mode"] == "calibrated"
    assert effect_row["decoder_a"] == "lda"
    assert effect_row["decoder_b"] == "logistic"
    assert effect_row["n_subjects"] == 4
    assert effect_row["better_decoder_by_mean"] == "logistic"
    assert round(effect_row["mean_difference_a_minus_b"], 3) == -0.080


def test_paired_decoder_statistics_keeps_emission_modes_separate():
    rows = []
    for subject in ["sub-01", "sub-02"]:
        rows.extend(
            [
                {"emission_mode": "calibrated", "decoder": "lda", "subject": subject, "baseline_accuracy": 0.5, "baseline_abs_delta": 0.0, "effect_accuracy": 0.50, "effect_minus_baseline": 0.00, "effect_log_loss": 0.7, "effect_brier": 0.4, "effect_ece": 0.1},
                {"emission_mode": "calibrated", "decoder": "logistic", "subject": subject, "baseline_accuracy": 0.5, "baseline_abs_delta": 0.0, "effect_accuracy": 0.70, "effect_minus_baseline": 0.20, "effect_log_loss": 0.5, "effect_brier": 0.3, "effect_ece": 0.05},
                {"emission_mode": "uncalibrated", "decoder": "lda", "subject": subject, "baseline_accuracy": 0.5, "baseline_abs_delta": 0.0, "effect_accuracy": 0.80, "effect_minus_baseline": 0.30, "effect_log_loss": 0.4, "effect_brier": 0.2, "effect_ece": 0.05},
                {"emission_mode": "uncalibrated", "decoder": "logistic", "subject": subject, "baseline_accuracy": 0.5, "baseline_abs_delta": 0.0, "effect_accuracy": 0.40, "effect_minus_baseline": -0.10, "effect_log_loss": 0.8, "effect_brier": 0.6, "effect_ece": 0.2},
            ]
        )
    stats = paired_decoder_statistics(pd.DataFrame(rows), metrics=("effect_accuracy",), n_permutations=10_000)
    effect_rows = stats.set_index("emission_mode")
    assert set(effect_rows.index) == {"calibrated", "uncalibrated"}
    assert effect_rows.loc["calibrated", "mean_difference_a_minus_b"] == pytest.approx(-0.20)
    assert effect_rows.loc["calibrated", "better_decoder_by_mean"] == "logistic"
    assert effect_rows.loc["uncalibrated", "mean_difference_a_minus_b"] == pytest.approx(0.40)
    assert effect_rows.loc["uncalibrated", "better_decoder_by_mean"] == "lda"


def test_paired_decoder_statistics_rejects_duplicate_subject_decoder_modes():
    rows = []
    for duplicate in range(2):
        rows.append(
            {
                "emission_mode": "calibrated",
                "decoder": "lda",
                "subject": "sub-01",
                "baseline_accuracy": 0.5,
                "baseline_abs_delta": 0.0,
                "effect_accuracy": 0.5 + duplicate,
                "effect_minus_baseline": 0.0,
                "effect_log_loss": 0.7,
                "effect_brier": 0.4,
                "effect_ece": 0.1,
            }
        )
    with pytest.raises(ValueError, match="at most one row"):
        paired_decoder_statistics(pd.DataFrame(rows), metrics=("effect_accuracy",))


def test_sign_flip_p_value_uses_exact_test_when_small():
    p_value = sign_flip_p_value(pd.Series([1.0, 1.0, 1.0, 1.0]).to_numpy(), n_permutations=10_000)
    assert p_value == 0.125
