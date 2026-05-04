from pathlib import Path

import pandas as pd

from reptrace.calibration import aggregate_reliability_bins, build_calibration_report, summarize_calibration_metrics


def _summary_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "decoder": ["logistic", "logistic", "lda", "lda"],
            "time": [-0.05, 0.15, -0.05, 0.15],
            "accuracy_mean": [0.50, 0.60, 0.54, 0.61],
            "log_loss_mean": [0.70, 0.66, 0.78, 0.74],
            "brier_mean": [0.50, 0.47, 0.55, 0.52],
            "ece_mean": [0.09, 0.06, 0.15, 0.12],
            "n_subjects": [5, 5, 5, 5],
        }
    )


def test_summarize_calibration_metrics_orders_by_effect_ece():
    summary = summarize_calibration_metrics(
        _summary_frame(),
        baseline_window=(-0.1, 0.0),
        effect_window=(0.1, 0.2),
    )

    assert summary["decoder"].tolist() == ["logistic", "lda"]
    assert summary["effect_ece_mean"].round(3).tolist() == [0.06, 0.12]
    assert summary["best_ece_time"].tolist() == [0.15, 0.15]


def test_aggregate_reliability_bins_weights_by_samples(tmp_path: Path):
    first = tmp_path / "sub-01_calibration_bins.csv"
    second = tmp_path / "sub-02_calibration_bins.csv"
    pd.DataFrame(
        {
            "decoder": ["logistic"],
            "time": [0.1],
            "bin": [5],
            "bin_left": [0.5],
            "bin_right": [0.6],
            "n_samples": [10],
            "accuracy": [0.8],
            "confidence": [0.6],
        }
    ).to_csv(first, index=False)
    pd.DataFrame(
        {
            "decoder": ["logistic"],
            "time": [0.1],
            "bin": [5],
            "bin_left": [0.5],
            "bin_right": [0.6],
            "n_samples": [30],
            "accuracy": [0.4],
            "confidence": [0.5],
        }
    ).to_csv(second, index=False)

    aggregated = aggregate_reliability_bins([first, second])

    assert aggregated["n_samples"].tolist() == [40]
    assert aggregated["accuracy"].round(3).tolist() == [0.5]
    assert aggregated["confidence"].round(3).tolist() == [0.525]
    assert aggregated["gap"].round(3).tolist() == [-0.025]


def test_build_calibration_report_writes_markdown(tmp_path: Path):
    summary_csv = tmp_path / "summary.csv"
    _summary_frame().to_csv(summary_csv, index=False)

    report = build_calibration_report(summary_csv, effect_window=(0.1, 0.2))

    assert "# RepTrace Calibration Report" in report
    assert "| logistic | 5 | 0.060 | 0.470 | 0.660 | 0.600 |" in report
