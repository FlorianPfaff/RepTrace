import numpy as np
import pandas as pd

from reptrace.onset_detection import annotate_threshold_crossings, detect_onsets


def _cached_threshold_frame() -> pd.DataFrame:
    rows = []
    traces = {
        0: [(-0.20, 0.55), (-0.10, 0.58), (0.05, 0.62), (0.15, 0.92)],
        1: [(-0.20, 0.57), (-0.10, 0.59), (0.05, 0.90), (0.15, 0.86)],
    }
    for sequence_id, trace in traces.items():
        for time, confidence in trace:
            rows.append(
                {
                    "subject": "sub-01",
                    "decoder": "logistic",
                    "emission_mode": "calibrated",
                    "time": time,
                    "sequence_id": sequence_id,
                    "confidence": confidence,
                    "prob_class_0": confidence,
                    "prob_class_1": 1.0 - confidence,
                }
            )
    return pd.DataFrame(rows)


def test_detect_onsets_recomputes_cached_threshold_when_window_changes():
    frame = _cached_threshold_frame()
    cached = annotate_threshold_crossings(
        frame,
        threshold_window=(-0.20, -0.10),
        threshold_quantile=0.75,
    )

    reused = detect_onsets(
        cached,
        threshold_window=(-0.20, -0.20),
        threshold_quantile=0.75,
        detection_start=0.0,
    )
    fresh = detect_onsets(
        frame,
        threshold_window=(-0.20, -0.20),
        threshold_quantile=0.75,
        detection_start=0.0,
    )

    assert np.allclose(reused["score_threshold"], fresh["score_threshold"])
    assert not np.isclose(cached["score_threshold"].iloc[0], fresh["score_threshold"].iloc[0])
