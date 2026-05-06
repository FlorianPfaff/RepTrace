from __future__ import annotations

import numpy as np
import pandas as pd

from reptrace.stimulus_detection import (
    detect_stimulus_events,
    fit_stimulus_detection_thresholds,
    match_stimulus_annotations,
    summarize_stimulus_events,
)

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


def _stream_frame() -> pd.DataFrame:
    rows = [
        *_baseline_rows(),
        _row("run-1", 0.00, (0.34, 0.33, 0.33)),
        _row("run-1", 0.10, (0.86, 0.08, 0.06)),
        _row("run-1", 0.20, (0.88, 0.07, 0.05)),
        _row("run-1", 0.30, (0.84, 0.10, 0.06)),
        _row("run-1", 0.60, (0.34, 0.33, 0.33)),
        _row("run-1", 0.80, (0.08, 0.84, 0.08)),
        _row("run-1", 0.90, (0.07, 0.86, 0.07)),
        _row("run-1", 1.00, (0.09, 0.83, 0.08)),
        _row("run-1", 1.25, (0.34, 0.33, 0.33)),
        _row("run-1", 1.40, (0.82, 0.10, 0.08)),
        _row("run-1", 1.50, (0.85, 0.08, 0.07)),
    ]
    return pd.DataFrame(rows)


def test_detect_stimulus_events_returns_multiple_events_per_stream():
    events = detect_stimulus_events(
        _stream_frame(),
        stream_columns=("stream_id",),
        threshold_window=THRESHOLD_WINDOW,
        threshold_quantile=1.0,
        detection_window=(0.0, float("inf")),
        min_consecutive=2,
    )

    assert list(events["stimulus_class"]) == ["A", "B", "A"]
    assert list(events["event_index"]) == [0, 1, 2]
    assert list(events["onset_time"]) == [0.10, 0.80, 1.40]
    assert list(events["detection_confirmed_time"]) == [0.20, 0.90, 1.50]
    assert events["run_length"].tolist() == [3, 3, 2]


def test_fit_thresholds_can_be_reused_for_detection():
    frame = _stream_frame()
    thresholds = fit_stimulus_detection_thresholds(
        frame,
        stream_columns=("stream_id",),
        threshold_window=THRESHOLD_WINDOW,
        threshold_quantile=1.0,
    )

    events = detect_stimulus_events(
        frame,
        stream_columns=("stream_id",),
        thresholds=thresholds,
        detection_window=(0.0, float("inf")),
        min_consecutive=2,
    )

    assert set(thresholds["stimulus_class"]) == {"A", "B", "C"}
    assert thresholds.set_index("stimulus_class").loc[["A", "B", "C"], "score_threshold"].eq(0.60).all()
    assert len(events) == 3
    assert events["score_threshold"].notna().all()


def test_merge_gap_collapses_brief_interruptions():
    frame = pd.DataFrame(
        [
            *_baseline_rows(),
            _row("run-1", 0.10, (0.86, 0.08, 0.06)),
            _row("run-1", 0.20, (0.88, 0.07, 0.05)),
            _row("run-1", 0.30, (0.30, 0.35, 0.35)),
            _row("run-1", 0.40, (0.84, 0.10, 0.06)),
            _row("run-1", 0.50, (0.82, 0.10, 0.08)),
        ]
    )

    split = detect_stimulus_events(
        frame,
        stream_columns=("stream_id",),
        target_classes=["A"],
        threshold_window=THRESHOLD_WINDOW,
        threshold_quantile=1.0,
        detection_window=(0.0, float("inf")),
        min_consecutive=1,
    )
    merged = detect_stimulus_events(
        frame,
        stream_columns=("stream_id",),
        target_classes=["A"],
        threshold_window=THRESHOLD_WINDOW,
        threshold_quantile=1.0,
        detection_window=(0.0, float("inf")),
        min_consecutive=1,
        merge_gap=0.25,
    )

    assert len(split) == 2
    assert len(merged) == 1
    assert merged.iloc[0]["run_length"] == 4
    assert merged.iloc[0]["offset_time"] == 0.50


def test_refractory_suppresses_close_duplicates_for_same_class():
    frame = pd.DataFrame(
        [
            *_baseline_rows(),
            _row("run-1", 0.10, (0.86, 0.08, 0.06)),
            _row("run-1", 0.20, (0.30, 0.35, 0.35)),
            _row("run-1", 0.30, (0.84, 0.10, 0.06)),
            _row("run-1", 0.60, (0.34, 0.33, 0.33)),
            _row("run-1", 0.90, (0.83, 0.10, 0.07)),
        ]
    )

    events = detect_stimulus_events(
        frame,
        stream_columns=("stream_id",),
        target_classes=["A"],
        threshold_window=THRESHOLD_WINDOW,
        threshold_quantile=1.0,
        detection_window=(0.0, float("inf")),
        refractory=0.5,
    )

    assert list(events["onset_time"]) == [0.10, 0.90]


def test_annotation_matching_and_summary_report_event_level_metrics():
    events = detect_stimulus_events(
        _stream_frame(),
        stream_columns=("stream_id",),
        threshold_window=THRESHOLD_WINDOW,
        threshold_quantile=1.0,
        detection_window=(0.0, float("inf")),
        min_consecutive=2,
    )
    annotations = pd.DataFrame(
        [
            {"stream_id": "run-1", "annotation_id": "ann-1", "stimulus_class": "A", "onset_time": 0.10},
            {"stream_id": "run-1", "annotation_id": "ann-2", "stimulus_class": "B", "onset_time": 0.80},
            {"stream_id": "run-1", "annotation_id": "ann-3", "stimulus_class": "A", "onset_time": 1.40},
        ]
    )

    matched = match_stimulus_annotations(
        events,
        annotations,
        stream_columns=("stream_id",),
        match_tolerance=0.05,
    )
    summary = summarize_stimulus_events(matched, annotations=annotations)

    assert matched["is_true_positive"].all()
    assert matched["matched_annotation_id"].tolist() == ["ann-1", "ann-2", "ann-3"]
    assert summary.iloc[0]["precision"] == 1.0
    assert summary.iloc[0]["recall"] == 1.0
    assert summary.iloc[0]["f1"] == 1.0
