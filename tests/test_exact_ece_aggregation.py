from pathlib import Path

import pandas as pd
import pytest

from reptrace.results import aggregate_time_decode_csvs, aggregate_time_decode_results


def _fold_results(subjects=("s1",)) -> pd.DataFrame:
    rows = []
    for subject in subjects:
        rows.extend(
            [
                {"subject": subject, "fold": 0, "time": 0.1, "accuracy": 1.0, "log_loss": 0.4, "brier": 0.1, "ece": 0.4, "n_test": 1},
                {"subject": subject, "fold": 1, "time": 0.1, "accuracy": 0.0, "log_loss": 0.6, "brier": 0.2, "ece": 0.6, "n_test": 1},
            ]
        )
    return pd.DataFrame(rows)


def _observations(subject="s1") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "subject": [subject, subject],
            "fold": [0, 1],
            "time": [0.1, 0.1],
            "true_label": [0, 1],
            "prob_class_0": [0.6, 0.6],
            "prob_class_1": [0.4, 0.4],
        }
    )


def test_exact_ece_is_recomputed_from_pooled_observations():
    aggregated = aggregate_time_decode_results(_fold_results(), observations=_observations(), ece_bins=2)
    assert aggregated["ece_mean"].round(3).tolist() == [0.1]


def test_exact_ece_requires_complete_observation_coverage():
    with pytest.raises(ValueError, match="do not cover all result subject/time groups"):
        aggregate_time_decode_results(_fold_results(("s1", "s2")), observations=_observations("s1"), ece_bins=2)


def test_exact_ece_can_be_recomputed_from_observation_csvs(tmp_path: Path):
    results = tmp_path / "subject_results.csv"
    observations = tmp_path / "subject_observations.csv"
    out = tmp_path / "summary.csv"
    _fold_results().to_csv(results, index=False)
    _observations().to_csv(observations, index=False)

    aggregated = aggregate_time_decode_csvs([results], out_path=out, observation_csv_paths=[observations], ece_bins=2)

    assert out.exists()
    assert aggregated["ece_mean"].round(3).tolist() == [0.1]
