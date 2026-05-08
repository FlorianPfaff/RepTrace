from __future__ import annotations

import numpy as np
import pandas as pd

from reptrace.stimulus_detection import detect_stimulus_events

THRESHOLD_WINDOW = (-0.65, -0.05)


def _row(stream_id: str, time: float, probabilities: tuple[float, float, float]) -> dict:
    predicted_label = int(np.argmax(probabilities))
    return {
        "subject": "sub-01",
        "stream_id": stream_id,
        "decoder": "logistic",
        "emission_mode": "calibrated",
        "time": time,
        "window_start": time - 0.025,
        "window_stop": time + 0.025,
        "predicted_label": predicted_label,
        "predicted_class": ["A", "B", "C"][predicted_label],
        "confidence": max(probabilities),
        "class_0": "A",
        "class_1": "B",
        "class_2": "C",
        "prob_class_0": probabilities[0],
        "prob_class_1": probabilities[1],
        "prob_class_2": probabilities[2],
    }


def _baseline_rows(stream_id: str = "run-1") -> list[dict]:
    return [
        _row(stream_id, -0.60, (0.60, 0.20, 0.20)),
        _row(stream_id, -0.50, (0.60, 0.20, 0.20)),
        _row(stream_id, -0.40, (0.20, 0.60, 0.20)),
        _row(stream_id, -0.30, (0.20, 0.60, 0.20)),
        _row(stream_id, -0.20, (0.20, 0.20, 0.60)),
        _row(stream_id, -0.10, (0.20, 0.20, 0.60)),
    ]


def _overlap_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            *_baseline_rows(),
            _row("run-1", 0.10, (0.45, 0.42, 0.13)),
            _row("run-1", 0.20, (0.46, 0.43, 0.11)),
        ]
    )


def _detect(frame: pd.DataFrame, conflict_resolution: str = "none") -> pd.DataFrame:
    return detect_stimulus_events(
        frame,
        stream_columns=("stream_id",),
        target_classes=["A", "B"],
        threshold_window=THRESHOLD_WINDOW,
        threshold_quantile=0.0,
        detection_window=(0.0, float("inf")),
        min_consecutive=2,
        conflict_resolution=conflict_resolution,
    )


def test_default_conflict_resolution_keeps_overlapping_class_events():
    events = _detect(_overlap_frame())

    assert events["stimulus_class"].tolist() == ["A", "B"]
    assert events["conflict_resolution"].eq("none").all()


def test_winner_take_all_keeps_strongest_overlapping_event_cluster():
    events = _detect(_overlap_frame(), conflict_resolution="winner_take_all")

    assert events["stimulus_class"].tolist() == ["A"]
    assert events["event_index"].tolist() == [0]
    assert events["conflict_resolution"].eq("winner_take_all").all()


def test_non_max_suppression_keeps_non_overlapping_high_scoring_events():
    frame = pd.DataFrame(
        [
            *_baseline_rows(),
            _row("run-1", 0.10, (0.46, 0.10, 0.44)),
            _row("run-1", 0.20, (0.47, 0.10, 0.43)),
            _row("run-1", 0.30, (0.46, 0.10, 0.44)),
            _row("run-1", 0.50, (0.10, 0.45, 0.10)),
            _row("run-1", 0.60, (0.10, 0.46, 0.10)),
        ]
    )

    events = detect_stimulus_events(
        frame,
        stream_columns=("stream_id",),
        target_classes=["A", "B", "C"],
        threshold_window=THRESHOLD_WINDOW,
        threshold_quantile=0.0,
        detection_window=(0.0, float("inf")),
        min_consecutive=2,
        conflict_resolution="non_max_suppression",
    )

    assert events["stimulus_class"].tolist() == ["A", "B"]
    assert events["event_index"].tolist() == [0, 1]


def test_highest_peak_per_window_keeps_one_event_per_peak_time():
    events = _detect(_overlap_frame(), conflict_resolution="highest_peak_per_window")

    assert events["stimulus_class"].tolist() == ["A"]
    assert events["conflict_resolution"].eq("highest_peak_per_window").all()
