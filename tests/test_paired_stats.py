from pathlib import Path

import pandas as pd

from reptrace.paired_stats import paired_decoder_statistics, sign_flip_p_value, subject_decoder_metrics


def _write_decoder_csv(path: Path, subject: str, decoder: str, baseline: float, effect: float) -> None:
    rows = []
    for fold in [0, 1]:
        rows.extend(
            [
                {
                    "subject": subject,
                    "decoder": decoder,
                    "fold": fold,
                    "time": -0.05,
                    "accuracy": baseline + 0.01 * fold,
                    "log_loss": 0.8,
                    "brier": 0.6,
                    "ece": 0.2,
                },
                {
                    "subject": subject,
                    "decoder": decoder,
                    "fold": fold,
                    "time": 0.15,
                    "accuracy": effect + 0.01 * fold,
                    "log_loss": 0.7 - effect / 10,
                    "brier": 0.5 - effect / 10,
                    "ece": 0.15 - effect / 20,
                },
            ]
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def test_subject_decoder_metrics_computes_baseline_corrected_effect(tmp_path: Path):
    csv_path = tmp_path / "sub-01_logistic_time_decode.csv"
    _write_decoder_csv(csv_path, "sub-01", "logistic", baseline=0.50, effect=0.62)

    metrics = subject_decoder_metrics([csv_path], baseline_window=(-0.1, 0.0), effect_window=(0.1, 0.2))

    row = metrics.iloc[0]
    assert row["decoder"] == "logistic"
    assert row["subject"] == "sub-01"
    assert round(row["baseline_accuracy"], 3) == 0.505
    assert round(row["effect_accuracy"], 3) == 0.625
    assert round(row["effect_minus_baseline"], 3) == 0.120


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

    effect_row = stats[stats["metric"] == "effect_minus_baseline"].iloc[0]
    assert effect_row["decoder_a"] == "lda"
    assert effect_row["decoder_b"] == "logistic"
    assert effect_row["n_subjects"] == 4
    assert effect_row["better_decoder_by_mean"] == "logistic"
    assert round(effect_row["mean_difference_a_minus_b"], 3) == -0.080


def test_sign_flip_p_value_uses_exact_test_when_small():
    p_value = sign_flip_p_value(pd.Series([1.0, 1.0, 1.0, 1.0]).to_numpy(), n_permutations=10_000)

    assert p_value == 0.125
