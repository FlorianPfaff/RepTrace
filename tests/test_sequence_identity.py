from pathlib import Path

import pandas as pd

from reptrace.onset_detection import detect_onsets_from_csvs
from reptrace.temporal_model import fit_temporal_models


def _frame() -> pd.DataFrame:
    rows = []
    for sequence_id in range(3):
        for time, p0 in [(-0.1, 0.55), (0.1, 0.90), (0.2, 0.86)]:
            rows.append(
                {
                    "subject": "sub-01",
                    "fold": 0,
                    "decoder": "logistic",
                    "emission_mode": "calibrated",
                    "time": time,
                    "sequence_id": sequence_id,
                    "true_label": 0,
                    "true_class": "left",
                    "predicted_label": 0,
                    "predicted_class": "left",
                    "confidence": p0,
                    "class_0": "left",
                    "class_1": "right",
                    "prob_class_0": p0,
                    "prob_class_1": 1.0 - p0,
                }
            )
    return pd.DataFrame(rows)


def _paths(tmp_path: Path) -> list[Path]:
    first = tmp_path / "a" / "observations.csv"
    second = tmp_path / "b" / "observations.csv"
    first.parent.mkdir()
    second.parent.mkdir()
    _frame().to_csv(first, index=False)
    _frame().to_csv(second, index=False)
    return [first, second]


def test_temporal_model_keeps_reused_sequence_ids_from_different_files_separate(tmp_path: Path):
    summary, states = fit_temporal_models(
        _paths(tmp_path),
        effect_window=(0.1, 0.2),
        baseline_window=(-0.1, 0.0),
        n_permutations=0,
        stay_grid_size=10,
        out_states=tmp_path / "states.csv",
    )

    observed = summary.loc[summary["condition"] == "observed_effect"].iloc[0]
    assert observed["n_sequences"] == 6
    assert states is not None
    assert states["source_file"].nunique() == 1
    assert states["source_path"].nunique() == 2


def test_onset_detection_keeps_reused_sequence_ids_from_different_files_separate(tmp_path: Path):
    events, summary = detect_onsets_from_csvs(
        _paths(tmp_path),
        threshold_window=(-0.1, -0.1),
        threshold_quantile=0.5,
        detection_window=(0.0, float("inf")),
    )

    assert len(events) == 6
    assert events["source_file"].nunique() == 1
    assert events["source_path"].nunique() == 2
    assert summary["n_sequences"].iloc[0] == 6
