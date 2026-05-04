from pathlib import Path

import pandas as pd

from reptrace.emission_compare import compare_emission_modes, compare_temporal_summary


def _temporal_summary() -> pd.DataFrame:
    rows = []
    for emission_mode, observed, baseline, shuffled_time, shuffled_label in [
        ("calibrated", 0.12, 0.02, 0.04, 0.03),
        ("uncalibrated", 0.10, 0.05, 0.07, 0.06),
    ]:
        for condition, gain, p_value in [
            ("observed_effect", observed, None),
            ("baseline_window", baseline, None),
            ("shuffled_time", shuffled_time, 0.02 if emission_mode == "calibrated" else 0.12),
            ("shuffled_label", shuffled_label, 0.04 if emission_mode == "calibrated" else 0.16),
        ]:
            rows.append(
                {
                    "decoder": "linear_svm",
                    "emission_mode": emission_mode,
                    "condition": condition,
                    "n_sequences": 10,
                    "n_observations": 100,
                    "n_states": 2,
                    "best_stay_probability": 0.9,
                    "persistence_gain_per_observation": gain,
                    "empirical_p_value": p_value,
                }
            )
    return pd.DataFrame(rows)


def test_compare_emission_modes_reports_control_margin_delta():
    comparison = compare_emission_modes(_temporal_summary())

    row = comparison.iloc[0]
    assert row["decoder"] == "linear_svm"
    assert round(row["calibrated_control_margin"], 3) == 0.08
    assert round(row["uncalibrated_control_margin"], 3) == 0.03
    assert round(row["delta_control_margin"], 3) == 0.05
    assert row["preferred_emission_mode"] == "calibrated"


def test_compare_temporal_summary_writes_csv_and_report(tmp_path: Path):
    summary_csv = tmp_path / "temporal_model.csv"
    out_csv = tmp_path / "emission_compare.csv"
    out_report = tmp_path / "emission_compare.md"
    _temporal_summary().to_csv(summary_csv, index=False)

    comparison, report = compare_temporal_summary(summary_csv, out_csv=out_csv, out_report=out_report)

    assert out_csv.exists()
    assert out_report.exists()
    assert report is not None and "calibrated probabilities produce cleaner" in report
    assert comparison["preferred_emission_mode"].tolist() == ["calibrated"]
