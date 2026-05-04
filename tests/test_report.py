from pathlib import Path

import pandas as pd

from reptrace.report import (
    build_time_decode_report,
    summarize_decoder_comparison,
    summarize_aggregate_time_decode,
    summarize_subject_time_decode,
)


def _summary_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time": [-0.05, 0.15, 0.25],
            "accuracy_mean": [0.49, 0.61, 0.58],
            "accuracy_sem": [0.01, 0.02, 0.03],
            "log_loss_mean": [0.72, 0.65, 0.68],
            "brier_mean": [0.52, 0.45, 0.48],
            "ece_mean": [0.11, 0.07, 0.08],
            "n_subjects": [2, 2, 2],
        }
    )


def _subject_frame(subject: str, offset: float = 0.0) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "subject": [subject, subject, subject, subject],
            "fold": [0, 1, 0, 1],
            "time": [0.15, 0.15, 0.25, 0.25],
            "accuracy": [0.60 + offset, 0.62 + offset, 0.58 + offset, 0.60 + offset],
            "log_loss": [0.6, 0.6, 0.7, 0.7],
            "brier": [0.4, 0.4, 0.5, 0.5],
            "ece": [0.1, 0.1, 0.2, 0.2],
        }
    )


def test_summarize_aggregate_time_decode_reports_peak_and_effect_window():
    summary = summarize_aggregate_time_decode(
        _summary_frame(),
        chance=0.5,
        baseline_window=(-0.1, 0.0),
        effect_window=(0.1, 0.3),
    )

    assert summary["peak_time"] == 0.15
    assert summary["peak_accuracy"] == 0.61
    assert round(summary["effect_accuracy_mean"], 3) == 0.595
    assert round(summary["effect_accuracy_delta"], 3) == 0.095


def test_summarize_subject_time_decode_averages_folds(tmp_path: Path):
    first = tmp_path / "sub-01_time_decode.csv"
    second = tmp_path / "sub-02_time_decode.csv"
    _subject_frame("sub-01").to_csv(first, index=False)
    _subject_frame("sub-02", offset=0.1).to_csv(second, index=False)

    summary = summarize_subject_time_decode([first, second], effect_window=(0.1, 0.3))

    assert summary["subject"].tolist() == ["sub-01", "sub-02"]
    assert summary["peak_time"].tolist() == [0.15, 0.15]
    assert summary["peak_accuracy"].round(3).tolist() == [0.61, 0.71]


def test_build_time_decode_report_writes_markdown_summary(tmp_path: Path):
    summary_csv = tmp_path / "summary.csv"
    subject_csv = tmp_path / "sub-01_time_decode.csv"
    _summary_frame().to_csv(summary_csv, index=False)
    _subject_frame("sub-01").to_csv(subject_csv, index=False)

    report = build_time_decode_report(summary_csv, subject_csvs=[subject_csv])

    assert "# RepTrace Time-Decoding Report" in report
    assert "| Peak aggregate accuracy | 0.610 |" in report
    assert "| sub-01 | 0.150 | 0.610 | 0.600 |" in report


def test_summarize_decoder_comparison_baseline_corrects_decoder_effects():
    logistic = _summary_frame()
    logistic["decoder"] = "logistic"
    svm = _summary_frame()
    svm["decoder"] = "linear_svm"
    svm["accuracy_mean"] = [0.58, 0.62, 0.61]
    summary = pd.concat([logistic, svm], ignore_index=True)

    comparison = summarize_decoder_comparison(summary, baseline_window=(-0.1, 0.0), effect_window=(0.1, 0.3))

    assert comparison["decoder"].tolist() == ["logistic", "linear_svm"]
    assert comparison["effect_minus_baseline"].round(3).tolist() == [0.105, 0.035]


def test_build_time_decode_report_handles_decoder_summary(tmp_path: Path):
    summary = _summary_frame()
    summary["decoder"] = "logistic"
    summary_csv = tmp_path / "summary.csv"
    summary.to_csv(summary_csv, index=False)

    report = build_time_decode_report(summary_csv)

    assert "# RepTrace Decoder Comparison Report" in report
    assert "| logistic | 2 | 0.150 | 0.610 | 0.490 | 0.595 | 0.105 |" in report
