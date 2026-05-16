from pathlib import Path

import pandas as pd

from reptrace.results import aggregate_time_decode_csvs, aggregate_time_decode_results


def test_aggregate_time_decode_results_recomputes_ece_from_pooled_observations():
    results = pd.DataFrame(
        {
            "subject": ["s1", "s1"],
            "fold": [0, 1],
            "time": [0.1, 0.1],
            "accuracy": [1.0, 0.0],
            "log_loss": [0.4, 0.7],
            "brier": [0.3, 0.6],
            # Fold-level ECEs average to 0.54, but pooled top-label ECE is 0.15.
            "ece": [0.39, 0.69],
            "n_test": [1, 1],
        }
    )
    observations = pd.DataFrame(
        {
            "subject": ["s1", "s1"],
            "fold": [0, 1],
            "time": [0.1, 0.1],
            "true_label": [0, 0],
            "prob_class_0": [0.61, 0.31],
            "prob_class_1": [0.39, 0.69],
        }
    )

    aggregated = aggregate_time_decode_results(results, observations=observations)

    assert aggregated["ece_mean"].round(3).tolist() == [0.15]
    assert aggregated["accuracy_mean"].round(3).tolist() == [0.5]


def test_aggregate_time_decode_results_rejects_observation_count_mismatch():
    results = pd.DataFrame(
        {
            "subject": ["s1"],
            "fold": [0],
            "time": [0.1],
            "accuracy": [1.0],
            "log_loss": [0.4],
            "brier": [0.3],
            "ece": [0.39],
            "n_test": [2],
        }
    )
    observations = pd.DataFrame(
        {
            "subject": ["s1"],
            "time": [0.1],
            "true_label": [0],
            "prob_class_0": [0.61],
            "prob_class_1": [0.39],
        }
    )

    try:
        aggregate_time_decode_results(results, observations=observations)
    except ValueError as exc:
        assert "row counts do not match fold n_test totals" in str(exc)
    else:
        raise AssertionError("Expected incomplete observation rows to fail")


def test_aggregate_time_decode_results_requires_observations_to_match_result_groups():
    results = pd.DataFrame(
        {
            "subject": ["s1"],
            "time": [0.1],
            "accuracy": [1.0],
            "log_loss": [0.4],
            "brier": [0.3],
            "ece": [0.39],
        }
    )
    observations = pd.DataFrame(
        {
            "subject": ["s2"],
            "time": [0.1],
            "true_label": [0],
            "prob_class_0": [0.61],
            "prob_class_1": [0.39],
        }
    )

    try:
        aggregate_time_decode_results(results, observations=observations)
    except ValueError as exc:
        assert "do not cover all result subject/time groups" in str(exc)
    else:
        raise AssertionError("Expected mismatched observation groups to fail")


def test_aggregate_time_decode_csvs_fills_blank_observation_subject_from_matching_result_file(tmp_path: Path):
    metrics = tmp_path / "sub-01_metrics.csv"
    observations = tmp_path / "sub-01_observations.csv"
    pd.DataFrame(
        {
            "fold": [0, 1],
            "time": [0.1, 0.1],
            "accuracy": [1.0, 0.0],
            "log_loss": [0.4, 0.7],
            "brier": [0.3, 0.6],
            "ece": [0.39, 0.69],
            "n_test": [1, 1],
        }
    ).to_csv(metrics, index=False)
    pd.DataFrame(
        {
            "subject": ["", ""],
            "fold": [0, 1],
            "time": [0.1, 0.1],
            "true_label": [0, 0],
            "prob_class_0": [0.61, 0.31],
            "prob_class_1": [0.39, 0.69],
        }
    ).to_csv(observations, index=False)

    out = tmp_path / "summary.csv"
    aggregated = aggregate_time_decode_csvs([metrics], out_path=out, observation_csv_paths=[observations])

    assert out.exists()
    assert aggregated["ece_mean"].round(3).tolist() == [0.15]
