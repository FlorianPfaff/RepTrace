from pathlib import Path

import pandas as pd

from reptrace.report import build_time_decode_report, summarize_aggregate_time_decode, summarize_decoder_comparison, summarize_subject_time_decode


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


def _subject_frame(subject: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "subject": [subject, subject, subject, subject],
            "fold": [0, 1, 0, 1],
            "time": [0.15, 0.15, 0.25, 0.25],
            "accuracy": [0.60, 0.62, 0.58, 0.60],
            "log_loss": [0.7, 0.7, 0.4, 0.4],
            "brier": [0.5, 0.5, 0.3, 0.3],
            "ece": [0.2, 0.2, 0.1, 0.1],
        }
    )


def test_summarize_aggregate_time_decode_can_select_probability_metric():
    summary = summarize_aggregate_time_decode(
        _summary_frame(),
        baseline_window=(-0.1, 0.0),
        effect_window=(0.1, 0.3),
        selection_metric="brier",
    )

    assert summary["selection_metric"] == "brier"
    assert summary["selected_time"] == 0.15
    assert summary["selected_score"] == 0.45
    assert summary["selected_accuracy"] == 0.61
    assert round(summary["selection_improvement"], 3) == 0.055


def test_summarize_subject_time_decode_can_select_probability_metric(tmp_path: Path):
    subject_csv = tmp_path / "sub-01_time_decode.csv"
    _subject_frame("sub-01").to_csv(subject_csv, index=False)

    summary = summarize_subject_time_decode([subject_csv], effect_window=(0.1, 0.3), selection_metric="log_loss")

    assert summary.loc[0, "selection_metric"] == "log_loss"
    assert summary.loc[0, "selected_time"] == 0.25
    assert summary.loc[0, "selected_score"] == 0.4
    assert round(summary.loc[0, "peak_accuracy"], 3) == 0.59


def test_summarize_decoder_comparison_can_rank_by_log_loss_improvement():
    logistic = _summary_frame().assign(decoder="logistic")
    calibrated = _summary_frame().assign(decoder="calibrated_logistic")
    calibrated["accuracy_mean"] = [0.48, 0.60, 0.59]
    calibrated["log_loss_mean"] = [0.80, 0.50, 0.52]
    summary = pd.concat([logistic, calibrated], ignore_index=True)

    comparison = summarize_decoder_comparison(
        summary,
        baseline_window=(-0.1, 0.0),
        effect_window=(0.1, 0.3),
        selection_metric="log_loss",
    )

    assert comparison["decoder"].tolist() == ["calibrated_logistic", "logistic"]
    assert comparison["selected_score"].round(3).tolist() == [0.500, 0.650]
    assert comparison["selection_improvement"].round(3).tolist() == [0.290, 0.135]


def test_build_time_decode_report_describes_probability_metric_selection(tmp_path: Path):
    summary = pd.concat(
        [
            _summary_frame().assign(decoder="logistic"),
            _summary_frame().assign(decoder="calibrated_logistic", log_loss_mean=[0.80, 0.50, 0.52]),
        ],
        ignore_index=True,
    )
    summary_csv = tmp_path / "summary.csv"
    subject_csv = tmp_path / "sub-01_time_decode.csv"
    summary.to_csv(summary_csv, index=False)
    _subject_frame("sub-01").assign(decoder="logistic").to_csv(subject_csv, index=False)

    report = build_time_decode_report(summary_csv, subject_csvs=[subject_csv], selection_metric="log_loss")

    assert "- Selection metric: log loss (lower is better)" in report
    assert "Selected Log Loss" in report
    assert "Accuracy at selected time" in report
    assert "| Subject | Selected time (s) | Selected Log Loss | Accuracy at selected time | Effect-window mean Log Loss |" in report
