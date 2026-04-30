from pathlib import Path

import pandas as pd

from reptrace.plot_time_decode import plot_time_decode_results


def test_plot_time_decode_results_writes_png(tmp_path: Path):
    results_csv = tmp_path / "results.csv"
    out_path = tmp_path / "plot.png"
    pd.DataFrame(
        {
            "time": [0.1, 0.1, 0.2, 0.2],
            "accuracy": [0.6, 0.8, 0.7, 0.9],
            "log_loss": [0.5, 0.4, 0.45, 0.35],
            "brier": [0.3, 0.2, 0.25, 0.15],
            "ece": [0.1, 0.2, 0.15, 0.25],
        }
    ).to_csv(results_csv, index=False)

    plot_time_decode_results(results_csv, out_path=out_path, metrics=("accuracy",), chance=0.5)

    assert out_path.exists()
    assert out_path.stat().st_size > 0
