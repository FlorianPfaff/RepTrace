from __future__ import annotations

import pandas as pd
import pandas.testing as pdt
import pytest

from reptrace.onset_detection import _detection_runs, _first_detection_run


def _candidates() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time": [0.0, 0.1, 0.2, 0.4, 0.5, 0.6],
            "window_start": [-0.01, 0.09, 0.19, 0.39, 0.49, 0.59],
            "window_stop": [0.01, 0.11, 0.21, 0.41, 0.51, 0.61],
            "_onset_score": [0.90, 0.92, 0.20, 0.91, 0.93, 0.94],
            "predicted_label": [0, 0, 0, 1, 1, 1],
        }
    )


def test_detection_runs_returns_all_valid_segments_and_preserves_first_run_behavior():
    candidates = _candidates()

    runs = _detection_runs(
        candidates,
        threshold=0.80,
        min_consecutive=2,
        min_duration=None,
        require_stable_prediction=True,
    )
    first = _first_detection_run(
        candidates,
        threshold=0.80,
        min_consecutive=2,
        min_duration=None,
        require_stable_prediction=True,
    )

    assert len(runs) == 2
    assert [float(run.iloc[0]["time"]) for run in runs] == [0.0, 0.4]
    pdt.assert_frame_equal(first, runs[0])


def test_detection_runs_can_merge_close_segments_with_same_prediction():
    candidates = _candidates()
    candidates["predicted_label"] = 0

    runs = _detection_runs(
        candidates,
        threshold=0.80,
        min_consecutive=2,
        min_duration=None,
        require_stable_prediction=True,
        merge_gap=0.25,
    )

    assert len(runs) == 1
    assert runs[0]["time"].tolist() == [0.0, 0.1, 0.4, 0.5, 0.6]


def test_detection_runs_refractory_suppresses_close_duplicate_runs():
    runs = _detection_runs(
        _candidates(),
        threshold=0.80,
        min_consecutive=2,
        min_duration=None,
        require_stable_prediction=True,
        refractory=0.5,
    )

    assert len(runs) == 1
    assert runs[0]["time"].tolist() == [0.0, 0.1]


@pytest.mark.parametrize("kwargs", [{"merge_gap": -0.1}, {"refractory": -0.1}])
def test_detection_runs_rejects_negative_gap_controls(kwargs):
    with pytest.raises(ValueError):
        _detection_runs(
            _candidates(),
            threshold=0.80,
            min_consecutive=2,
            min_duration=None,
            require_stable_prediction=True,
            **kwargs,
        )
