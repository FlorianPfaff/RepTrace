from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from reptrace.stimulus_detection import (
    detect_stimulus_events,
    fit_stimulus_detection_thresholds,
    main as stimulus_detection_main,
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


def _manual_event(stream_id: str, onset_time: float, stimulus_class: str, event_index: int) -> dict:
    return {
        "subject": "sub-01",
        "stream_id": stream_id,
        "decoder": "logistic",
        "emission_mode": "calibrated",
        "event_index": event_index,
        "detected": True,
        "stimulus_label": {"A": 0, "B": 1, "C": 2}[stimulus_class],
        "stimulus_class": stimulus_class,
        "onset_time": onset_time,
        "offset_time": onset_time + 0.05,
        "peak_time": onset_time,
        "peak_score": 0.9,
    }


def test_detect_stimulus_events_returns_empty_when_no_stimulus_crosses_threshold():
    frame = pd.DataFrame(
        [
            *_baseline_rows(),
            _row("run-1", 0.00, (0.34, 0.33, 0.33)),
            _row("run-1", 0.10, (0.35, 0.33, 0.32)),
            _row("run-1", 0.20, (0.33, 0.34, 0.33)),
        ]
    )

    events = detect_stimulus_events(
        frame,
        stream_columns=("stream_id",),
        threshold_window=THRESHOLD_WINDOW,
        threshold_quantile=1.0,
        detection_window=(0.0, float("inf")),
    )

    assert events.empty
    assert {"stream_id", "event_index", "stimulus_class", "onset_time", "score_threshold"}.issubset(events.columns)


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


def test_predicted_class_confidence_only_scores_matching_winning_class():
    frame = pd.DataFrame(
        [
            _row("run-1", 0.10, (0.45, 0.55, 0.00)),
            _row("run-1", 0.20, (0.46, 0.54, 0.00)),
        ]
    )
    thresholds = pd.DataFrame(
        [
            {
                "subject": "sub-01",
                "decoder": "logistic",
                "emission_mode": "calibrated",
                "stimulus_label": 0,
                "stimulus_class": "A",
                "score_column": "prob_class_0",
                "score_mode": "predicted_class_confidence",
                "score_threshold": 0.40,
                "threshold_method": "point",
                "threshold_quantile": 1.0,
                "threshold_window_start": THRESHOLD_WINDOW[0],
                "threshold_window_stop": THRESHOLD_WINDOW[1],
            },
            {
                "subject": "sub-01",
                "decoder": "logistic",
                "emission_mode": "calibrated",
                "stimulus_label": 1,
                "stimulus_class": "B",
                "score_column": "prob_class_1",
                "score_mode": "predicted_class_confidence",
                "score_threshold": 0.40,
                "threshold_method": "point",
                "threshold_quantile": 1.0,
                "threshold_window_start": THRESHOLD_WINDOW[0],
                "threshold_window_stop": THRESHOLD_WINDOW[1],
            },
        ]
    )

    events = detect_stimulus_events(
        frame,
        thresholds=thresholds,
        stream_columns=("stream_id",),
        min_consecutive=2,
    )

    assert events["stimulus_class"].tolist() == ["B"]
    assert events.iloc[0]["score_mode"] == "predicted_class_confidence"
    assert events.iloc[0]["score_at_onset"] == 0.55
    assert events.iloc[0]["peak_score"] == 0.55


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
            {"stream_id": "run-1", "annotation_id": 1, "stimulus_class": "A", "onset_time": 0.10},
            {"stream_id": "run-1", "annotation_id": 2, "stimulus_class": "B", "onset_time": 0.80},
            {"stream_id": "run-1", "annotation_id": 3, "stimulus_class": "A", "onset_time": 1.40},
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
    assert matched["matched_annotation_id"].tolist() == [1, 2, 3]
    assert summary.iloc[0]["true_positives"] == 3
    assert summary.iloc[0]["false_positives"] == 0
    assert summary.iloc[0]["false_negatives"] == 0
    assert summary.iloc[0]["precision"] == 1.0
    assert summary.iloc[0]["recall"] == 1.0
    assert summary.iloc[0]["f1"] == 1.0
    assert summary.iloc[0]["class_accuracy_for_matched_events"] == 1.0
    assert summary.iloc[0]["onset_latency_mean"] == 0.0


def test_summary_counts_duplicate_detections_and_false_alarms_per_minute():
    events = pd.DataFrame(
        [
            _manual_event("run-1", 0.00, "A", 0),
            _manual_event("run-1", 0.03, "A", 1),
            _manual_event("run-1", 10.00, "B", 2),
            _manual_event("run-1", 20.00, "A", 3),
            _manual_event("run-1", 50.00, "A", 4),
        ]
    )
    annotations = pd.DataFrame(
        [
            {"stream_id": "run-1", "annotation_id": 1, "stimulus_class": "A", "onset_time": 0.00},
            {"stream_id": "run-1", "annotation_id": 2, "stimulus_class": "B", "onset_time": 10.00},
            {"stream_id": "run-1", "annotation_id": 3, "stimulus_class": "A", "onset_time": 20.00},
        ]
    )
    observations = pd.DataFrame(
        [
            {"stream_id": "run-1", "time": 0.0},
            {"stream_id": "run-1", "time": 120.0},
        ]
    )

    matched = match_stimulus_annotations(events, annotations, stream_columns=("stream_id",), match_tolerance=0.05)
    summary = summarize_stimulus_events(
        matched,
        annotations=annotations,
        observations=observations,
        stream_columns=("stream_id",),
    )

    assert matched["is_true_positive"].tolist() == [True, False, True, True, False]
    assert matched["is_duplicate_detection"].tolist() == [False, True, False, False, False]
    assert summary.iloc[0]["n_annotations"] == 3
    assert summary.iloc[0]["n_detections"] == 5
    assert summary.iloc[0]["true_positives"] == 3
    assert summary.iloc[0]["false_positives"] == 2
    assert summary.iloc[0]["false_negatives"] == 0
    assert summary.iloc[0]["duplicate_detections"] == 1
    assert summary.iloc[0]["false_alarms_per_minute"] == 1.0
    assert summary.iloc[0]["precision"] == 3 / 5
    assert summary.iloc[0]["recall"] == 1.0
    assert summary.iloc[0]["f1"] == 2 * (3 / 5) * 1.0 / ((3 / 5) + 1.0)


def test_class_accuracy_for_matched_events_can_score_wrong_classes():
    events = pd.DataFrame(
        [
            _manual_event("run-1", 0.0, "A", 0),
            _manual_event("run-1", 1.0, "C", 1),
            _manual_event("run-1", 2.0, "B", 2),
        ]
    )
    annotations = pd.DataFrame(
        [
            {"stream_id": "run-1", "annotation_id": 1, "stimulus_class": "A", "onset_time": 0.0},
            {"stream_id": "run-1", "annotation_id": 2, "stimulus_class": "B", "onset_time": 1.0},
            {"stream_id": "run-1", "annotation_id": 3, "stimulus_class": "B", "onset_time": 2.0},
        ]
    )

    matched = match_stimulus_annotations(
        events,
        annotations,
        stream_columns=("stream_id",),
        match_tolerance=0.01,
        require_class_match=False,
    )
    summary = summarize_stimulus_events(matched, annotations=annotations)

    assert matched["is_true_positive"].all()
    assert summary.iloc[0]["class_accuracy_for_matched_events"] == 2 / 3


def test_stimulus_detection_cli_writes_events_and_summary_without_annotations(tmp_path, monkeypatch):
    observations_csv = tmp_path / "observations.csv"
    out_events = tmp_path / "stimulus_events.csv"
    out_summary = tmp_path / "stimulus_summary.csv"
    _stream_frame().to_csv(observations_csv, index=False)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "reptrace-stimulus-detect",
            str(observations_csv),
            "--stream-column",
            "stream_id",
            "--score-mode",
            "class_probability",
            "--threshold-window",
            str(THRESHOLD_WINDOW[0]),
            str(THRESHOLD_WINDOW[1]),
            "--threshold-method",
            "max_run",
            "--threshold-quantile",
            "1.0",
            "--detection-window",
            "0.0",
            "inf",
            "--min-consecutive",
            "2",
            "--merge-gap",
            "0.05",
            "--refractory",
            "0.20",
            "--out-events",
            str(out_events),
            "--out-summary",
            str(out_summary),
        ],
    )

    stimulus_detection_main()

    events = pd.read_csv(out_events)
    summary = pd.read_csv(out_summary)
    assert events["stimulus_class"].tolist() == ["A", "B", "A"]
    assert events["score_mode"].eq("class_probability").all()
    assert summary.iloc[0]["n_detections"] == 3
    assert pd.isna(summary.iloc[0]["n_annotations"])


def test_stimulus_detection_cli_writes_events_summary_and_thresholds(tmp_path, monkeypatch):
    observations_csv = tmp_path / "observations.csv"
    annotations_csv = tmp_path / "annotations.csv"
    out_events = tmp_path / "stimulus_events.csv"
    out_summary = tmp_path / "stimulus_summary.csv"
    out_thresholds = tmp_path / "stimulus_thresholds.csv"
    _stream_frame().to_csv(observations_csv, index=False)
    pd.DataFrame(
        [
            {"stream_id": "run-1", "annotation_id": 1, "stimulus_class": "A", "onset_time": 0.10},
            {"stream_id": "run-1", "annotation_id": 2, "stimulus_class": "B", "onset_time": 0.80},
            {"stream_id": "run-1", "annotation_id": 3, "stimulus_class": "A", "onset_time": 1.40},
        ]
    ).to_csv(annotations_csv, index=False)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "reptrace-stimulus-detect",
            str(observations_csv),
            "--annotations",
            str(annotations_csv),
            "--stream-column",
            "stream_id",
            "--threshold-window",
            str(THRESHOLD_WINDOW[0]),
            str(THRESHOLD_WINDOW[1]),
            "--threshold-quantile",
            "1.0",
            "--detection-window",
            "0.0",
            "inf",
            "--min-consecutive",
            "2",
            "--out-events",
            str(out_events),
            "--out-summary",
            str(out_summary),
            "--out-thresholds",
            str(out_thresholds),
        ],
    )

    stimulus_detection_main()

    events = pd.read_csv(out_events)
    summary = pd.read_csv(out_summary)
    thresholds = pd.read_csv(out_thresholds)
    assert events["stimulus_class"].tolist() == ["A", "B", "A"]
    assert events["is_true_positive"].all()
    assert summary.iloc[0]["precision"] == 1.0
    assert summary.iloc[0]["recall"] == 1.0
    assert summary.iloc[0]["f1"] == 1.0
    assert summary.iloc[0]["true_positives"] == 3
    assert summary.iloc[0]["class_accuracy_for_matched_events"] == 1.0
    assert set(thresholds["stimulus_class"]) == {"A", "B", "C"}
