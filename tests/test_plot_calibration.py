from pathlib import Path

import pandas as pd
import pytest

from reptrace.plot_calibration import plot_reliability_diagram, summarize_reliability_curve


def test_summarize_reliability_curve_weights_by_samples():
    bins = pd.DataFrame(
        {
            "decoder": ["logistic", "logistic", "logistic"],
            "time": [0.1, 0.2, 0.2],
            "bin": [1, 1, 2],
            "bin_left": [0.0, 0.0, 0.5],
            "bin_right": [0.5, 0.5, 1.0],
            "n_samples": [1, 3, 2],
            "accuracy": [1.0, 0.0, 0.5],
            "confidence": [0.2, 0.4, 0.8],
        }
    )

    curve = summarize_reliability_curve(bins, time_window=(0.1, 0.2))

    first_bin = curve[curve["bin"] == 1].iloc[0]
    assert first_bin["n_samples"] == 4
    assert first_bin["accuracy"] == 0.25
    assert first_bin["confidence"] == pytest.approx(0.35)


def test_plot_reliability_diagram_writes_png(tmp_path: Path):
    bins_csv = tmp_path / "reliability_bins.csv"
    out_path = tmp_path / "reliability.png"
    pd.DataFrame(
        {
            "decoder": ["logistic", "logistic", "lda", "lda"],
            "time": [0.1, 0.1, 0.1, 0.1],
            "bin": [1, 2, 1, 2],
            "bin_left": [0.0, 0.5, 0.0, 0.5],
            "bin_right": [0.5, 1.0, 0.5, 1.0],
            "n_samples": [10, 20, 8, 16],
            "accuracy": [0.4, 0.8, 0.3, 0.7],
            "confidence": [0.35, 0.75, 0.25, 0.65],
        }
    ).to_csv(bins_csv, index=False)

    plot_reliability_diagram(bins_csv, out_path=out_path, time_window=(0.0, 0.2))

    assert out_path.exists()
    assert out_path.stat().st_size > 0
