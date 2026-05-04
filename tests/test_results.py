from pathlib import Path

import pandas as pd

from reptrace.results import aggregate_time_decode_csvs, aggregate_time_decode_results


def _result_frame(subject: str, offset: float = 0.0) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "subject": [subject, subject, subject, subject],
            "fold": [0, 1, 0, 1],
            "time": [0.1, 0.1, 0.2, 0.2],
            "accuracy": [0.6 + offset, 0.8 + offset, 0.7 + offset, 0.9 + offset],
            "log_loss": [0.5, 0.4, 0.45, 0.35],
            "brier": [0.3, 0.2, 0.25, 0.15],
            "ece": [0.1, 0.2, 0.15, 0.25],
        }
    )


def test_aggregate_time_decode_results_averages_folds_then_subjects():
    results = pd.concat([_result_frame("s1"), _result_frame("s2", offset=0.1)], ignore_index=True)

    aggregated = aggregate_time_decode_results(results)

    assert aggregated["n_subjects"].tolist() == [2, 2]
    assert aggregated["accuracy_mean"].round(3).tolist() == [0.75, 0.85]


def test_aggregate_time_decode_csvs_uses_filename_as_subject(tmp_path: Path):
    first = tmp_path / "sub-01.csv"
    second = tmp_path / "sub-02.csv"
    _result_frame("ignored").drop(columns="subject").to_csv(first, index=False)
    _result_frame("ignored", offset=0.1).drop(columns="subject").to_csv(second, index=False)

    out = tmp_path / "summary.csv"
    aggregated = aggregate_time_decode_csvs([first, second], out_path=out)

    assert out.exists()
    assert aggregated["n_subjects"].tolist() == [2, 2]


def test_aggregate_time_decode_results_keeps_decoder_groups_separate():
    first = _result_frame("s1")
    first["decoder"] = "logistic"
    second = _result_frame("s1", offset=0.1)
    second["decoder"] = "lda"

    aggregated = aggregate_time_decode_results(pd.concat([first, second], ignore_index=True))

    assert aggregated["decoder"].tolist() == ["lda", "lda", "logistic", "logistic"]
    assert aggregated["accuracy_mean"].round(3).tolist() == [0.8, 0.9, 0.7, 0.8]
