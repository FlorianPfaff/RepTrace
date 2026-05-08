from __future__ import annotations

import pandas as pd

from reptrace.observation_schema import (
    probability_columns,
    read_validated_probability_observations,
    summarize_probability_observations,
    validate_probability_observations,
)
from reptrace.observation_schema import main as validate_observations_main


def _valid_observations() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "subject": ["sub-01", "sub-01"],
            "sequence_id": ["trial-001", "trial-001"],
            "time": [-0.1, -0.08],
            "decoder": ["logistic", "logistic"],
            "emission_mode": ["calibrated", "calibrated"],
            "true_class": ["face", "face"],
            "predicted_class": ["object", "face"],
            "prob_class_0": [0.6, 0.45],
            "prob_class_1": [0.4, 0.55],
            "class_0": ["object", "object"],
            "class_1": ["face", "face"],
        }
    )


def test_probability_columns_are_sorted_by_class_index() -> None:
    frame = pd.DataFrame({"prob_class_10": [0.1], "prob_class_2": [0.2], "prob_class_1": [0.7]})

    assert probability_columns(frame) == ["prob_class_1", "prob_class_2", "prob_class_10"]


def test_validate_minimal_generic_observations() -> None:
    report = validate_probability_observations(_valid_observations())

    assert report.is_valid
    assert not report.errors
    assert report.probability_columns == ("prob_class_0", "prob_class_1")


def test_missing_time_is_error() -> None:
    report = validate_probability_observations(_valid_observations().drop(columns=["time"]))

    assert not report.is_valid
    assert any(issue.code == "missing_time" for issue in report.errors)


def test_probability_sum_deviation_is_warning_by_default() -> None:
    frame = _valid_observations()
    frame.loc[0, "prob_class_0"] = 0.7

    report = validate_probability_observations(frame, probability_tolerance=1e-6)

    assert report.is_valid
    assert any(issue.code == "probability_sum_warning" for issue in report.warnings)


def test_probability_sum_deviation_can_be_required_as_error() -> None:
    frame = _valid_observations()
    frame.loc[0, "prob_class_0"] = 0.7

    report = validate_probability_observations(frame, probability_tolerance=1e-6, require_normalized=True)

    assert not report.is_valid
    assert any(issue.code == "probability_sum_error" for issue in report.errors)


def test_temporal_profile_requires_sequence_identifier() -> None:
    report = validate_probability_observations(_valid_observations().drop(columns=["sequence_id"]), profile="temporal-model")

    assert not report.is_valid
    assert any(issue.code == "missing_sequence_identifier" for issue in report.errors)


def test_temporal_profile_requires_at_least_one_multi_point_sequence() -> None:
    frame = _valid_observations()
    frame["sequence_id"] = ["trial-001", "trial-002"]

    report = validate_probability_observations(frame, profile="temporal-model")

    assert not report.is_valid
    assert any(issue.code == "no_multi_point_sequence" for issue in report.errors)


def test_stimulus_profile_accepts_stream_id() -> None:
    frame = _valid_observations().drop(columns=["sequence_id"])
    frame["stream_id"] = "stream-001"

    report = validate_probability_observations(frame, profile="stimulus-detection")

    assert report.is_valid


def test_read_validated_probability_observations_adds_reader_defaults(tmp_path) -> None:
    csv_path = tmp_path / "observations.csv"
    pd.DataFrame({"sample_index": [0, 0], "time": [0.0, 0.1], "prob_class_0": [0.4, 0.6], "prob_class_1": [0.6, 0.4]}).to_csv(csv_path, index=False)

    frame = read_validated_probability_observations([csv_path], profile="temporal-model")

    assert {"subject", "decoder", "emission_mode", "sequence_id", "source_file"}.issubset(frame.columns)


def test_summarize_probability_observations_returns_compact_row() -> None:
    summary = summarize_probability_observations(_valid_observations())

    assert summary.loc[0, "n_rows"] == 2
    assert summary.loc[0, "n_probability_columns"] == 2
    assert summary.loc[0, "n_subjects"] == 1


def test_validate_observations_cli_writes_report_and_summary(tmp_path) -> None:
    csv_path = tmp_path / "observations.csv"
    report_path = tmp_path / "validation.csv"
    summary_path = tmp_path / "summary.csv"
    _valid_observations().to_csv(csv_path, index=False)

    exit_code = validate_observations_main(
        [
            str(csv_path),
            "--profile",
            "temporal-model",
            "--report-out",
            str(report_path),
            "--summary-out",
            str(summary_path),
        ]
    )

    assert exit_code == 0
    assert report_path.exists()
    assert summary_path.exists()


def test_validate_observations_cli_returns_one_for_validation_errors(tmp_path) -> None:
    csv_path = tmp_path / "bad.csv"
    _valid_observations().drop(columns=["time"]).to_csv(csv_path, index=False)

    exit_code = validate_observations_main([str(csv_path)])

    assert exit_code == 1
