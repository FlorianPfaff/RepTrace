from __future__ import annotations

import numpy as np
import pandas as pd

from reptrace.matched_filter_detection import (
    detect_matched_filter_stimulus_events,
    fit_matched_filter_thresholds,
    fit_stimulus_event_templates,
    score_stimulus_event_templates,
)

CLASS_NAMES = ("A", "B")


def _row(stream_id: str, time: float, probabilities: tuple[float, float]) -> dict:
    predicted_label = int(np.argmax(probabilities))
    return {
        "subject": "sub-01",
        "stream_id": stream_id,
        "decoder": "logistic",
        "emission_mode": "calibrated",
        "time": time,
        "window_start": time - 0.05,
        "window_stop": time + 0.05,
        "predicted_label": predicted_label,
        "predicted_class": CLASS_NAMES[predicted_label],
        "confidence": max(probabilities),
        "class_0": "A",
        "class_1": "B",
        "prob_class_0": probabilities[0],
        "prob_class_1": probabilities[1],
    }


def _stream(stream_id: str, *, event_onset: float | None = None) -> pd.DataFrame:
    rows = []
    bump = {0.0: 0.20, 0.1: 0.42, 0.2: 0.48, 0.3: 0.28}
    for time in np.round(np.arange(-0.5, 2.01, 0.1), 1):
        class_a = 0.25
        if event_onset is not None:
            class_a += bump.get(round(float(time - event_onset), 1), 0.0)
        rows.append(_row(stream_id, float(time), (class_a, 1.0 - class_a)))
    return pd.DataFrame(rows)


def test_fit_templates_and_detect_shifted_event_with_matched_filter():
    train_observations = _stream("train", event_onset=0.0)
    template_annotations = pd.DataFrame([{"stream_id": "train", "stimulus_class": "A", "onset_time": 0.0}])
    templates = fit_stimulus_event_templates(
        train_observations,
        template_annotations,
        template_window=(0.0, 0.3),
        template_step=0.1,
        target_classes=["A"],
        stream_columns=("stream_id",),
        min_template_coverage=1.0,
    )

    scan_observations = _stream("scan", event_onset=1.0)
    thresholds = fit_matched_filter_thresholds(
        scan_observations,
        templates,
        threshold_window=(-0.5, 0.5),
        threshold_quantile=1.0,
        stream_columns=("stream_id",),
        min_template_coverage=1.0,
    )
    events = detect_matched_filter_stimulus_events(
        scan_observations,
        templates=templates,
        thresholds=thresholds,
        stream_columns=("stream_id",),
        detection_window=(0.5, 1.5),
        refractory=0.4,
        min_template_coverage=1.0,
    )

    assert len(events) == 1
    assert events.iloc[0]["stimulus_class"] == "A"
    assert events.iloc[0]["onset_time"] == 1.0
    assert events.iloc[0]["detector_method"] == "matched_filter"
    assert events.iloc[0]["peak_score"] > events.iloc[0]["score_threshold"]


def test_matched_filter_scores_peak_at_event_onset():
    train_observations = _stream("train", event_onset=0.0)
    template_annotations = pd.DataFrame([{"stream_id": "train", "stimulus_class": "A", "onset_time": 0.0}])
    templates = fit_stimulus_event_templates(
        train_observations,
        template_annotations,
        template_window=(0.0, 0.3),
        template_step=0.1,
        target_classes=["A"],
        stream_columns=("stream_id",),
        min_template_coverage=1.0,
    )

    scores = score_stimulus_event_templates(
        _stream("scan", event_onset=1.0),
        templates,
        stream_columns=("stream_id",),
        min_template_coverage=1.0,
    )

    best = scores.sort_values("matched_filter_score", ascending=False).iloc[0]
    assert best["stimulus_class"] == "A"
    assert best["time"] == 1.0
