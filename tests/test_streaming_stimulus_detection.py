from __future__ import annotations

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from reptrace.stimulus_detection import detect_stimulus_events, fit_stimulus_detection_thresholds
from reptrace.streaming_stimulus_detection import StimulusDetectionConfig, StreamingStimulusDetector

THRESHOLD_WINDOW = (-0.65, -0.05)
CLASS_NAMES = ("A", "B", "C")


def _observation(stream_id: str, time: float, probabilities: tuple[float, float, float]) -> dict:
    predicted_label = int(np.argmax(probabilities))
    row = {
        "subject": "sub-01",
        "stream_id": stream_id,
        "decoder": "logistic",
        "emission_mode": "calibrated",
        "time": time,
        "window_start": time - 0.025,
        "window_stop": time + 0.025,
        "predicted_label": predicted_label,
        "predicted_class": CLASS_NAMES[predicted_label],
        "confidence": max(probabilities),
    }
    for class_index, class_name in enumerate(CLASS_NAMES):
        row[f"class_{class_index}"] = class_name
        row[f"prob_class_{class_index}"] = probabilities[class_index]
    return row


def _rows(points: list[tuple[float, tuple[float, float, float]]], stream_id: str = "run-1") -> list[dict]:
    return [_observation(stream_id, time, probabilities) for time, probabilities in points]


def _baseline_rows() -> list[dict]:
    return _rows(
        [
            (-0.60, (0.60, 0.20, 0.20)),
            (-0.50, (0.60, 0.20, 0.20)),
            (-0.40, (0.20, 0.60, 0.20)),
            (-0.30, (0.20, 0.60, 0.20)),
            (-0.20, (0.20, 0.20, 0.60)),
            (-0.10, (0.20, 0.20, 0.60)),
        ]
    )


def _stream_frame(*, final_gap: bool = False) -> pd.DataFrame:
    points = [
        (0.00, (0.34, 0.33, 0.33)),
        (0.10, (0.86, 0.08, 0.06)),
        (0.20, (0.88, 0.07, 0.05)),
        (0.30, (0.84, 0.10, 0.06)),
        (0.60, (0.34, 0.33, 0.33)),
        (0.80, (0.08, 0.84, 0.08)),
        (0.90, (0.07, 0.86, 0.07)),
        (1.00, (0.09, 0.83, 0.08)),
        (1.25, (0.34, 0.33, 0.33)),
        (1.40, (0.82, 0.10, 0.08)),
        (1.50, (0.85, 0.08, 0.07)),
    ]
    if final_gap:
        points.append((1.80, (0.34, 0.33, 0.33)))
    return pd.DataFrame([*_baseline_rows(), *_rows(points)])


def _thresholds(frame: pd.DataFrame, target_classes: list[str] | None = None) -> pd.DataFrame:
    return fit_stimulus_detection_thresholds(
        frame,
        stream_columns=("stream_id",),
        target_classes=target_classes,
        threshold_window=THRESHOLD_WINDOW,
        threshold_quantile=1.0,
    )


def _run_streaming_detector(
    frame: pd.DataFrame,
    thresholds: pd.DataFrame,
    config: StimulusDetectionConfig,
) -> pd.DataFrame:
    detector = StreamingStimulusDetector(config, thresholds)
    events = []
    for observation in frame.sort_values(["stream_id", "time"]).to_dict("records"):
        events.extend(detector.update(observation))
    events.extend(detector.flush())
    return pd.DataFrame(events).sort_values(["onset_time", "stimulus_class"]).reset_index(drop=True)


def test_streaming_detector_rejects_missing_threshold_group_column():
    frame = _stream_frame(final_gap=True)
    detector = StreamingStimulusDetector(
        StimulusDetectionConfig(
            stream_columns=("stream_id",),
            detection_window=(0.0, float("inf")),
        ),
        _thresholds(frame, target_classes=["A"]),
    )
    observation = _observation("run-1", 0.10, (0.86, 0.08, 0.06))
    observation.pop("emission_mode")

    with pytest.raises(ValueError, match="missing threshold group columns"):
        detector.update(observation)


def test_streaming_detector_matches_offline_events_with_causal_confirmation():
    frame = _stream_frame(final_gap=True)
    thresholds = _thresholds(frame)
    offline = detect_stimulus_events(
        frame,
        thresholds=thresholds,
        stream_columns=("stream_id",),
        detection_window=(0.0, float("inf")),
        min_consecutive=2,
    ).sort_values(["onset_time", "stimulus_class"]).reset_index(drop=True)
    online = _run_streaming_detector(
        frame,
        thresholds,
        StimulusDetectionConfig(
            stream_columns=("stream_id",),
            detection_window=(0.0, float("inf")),
            min_consecutive=2,
        ),
    )

    comparable = ["stimulus_class", "onset_time", "offset_time", "peak_time", "run_length"]
    pdt.assert_frame_equal(online[comparable], offline[comparable])
    assert np.allclose(online["peak_score"], offline["peak_score"])
    assert (online["detection_confirmed_time"] > offline["detection_confirmed_time"]).all()


def test_streaming_detector_respects_merge_gap():
    frame = pd.DataFrame(
        [
            *_baseline_rows(),
            *_rows(
                [
                    (0.10, (0.86, 0.08, 0.06)),
                    (0.20, (0.88, 0.07, 0.05)),
                    (0.30, (0.30, 0.35, 0.35)),
                    (0.40, (0.84, 0.10, 0.06)),
                    (0.50, (0.82, 0.10, 0.08)),
                    (0.90, (0.34, 0.33, 0.33)),
                ]
            ),
        ]
    )

    online = _run_streaming_detector(
        frame,
        _thresholds(frame, target_classes=["A"]),
        StimulusDetectionConfig(
            stream_columns=("stream_id",),
            detection_window=(0.0, float("inf")),
            min_consecutive=1,
            merge_gap=0.25,
        ),
    )

    assert len(online) == 1
    assert online.iloc[0]["stimulus_class"] == "A"
    assert online.iloc[0]["onset_time"] == 0.10
    assert online.iloc[0]["offset_time"] == 0.50
    assert online.iloc[0]["run_length"] == 4
    assert online.iloc[0]["detection_confirmed_time"] == 0.90


def test_streaming_detector_flushes_open_valid_run():
    frame = pd.DataFrame(
        [
            *_baseline_rows(),
            *_rows(
                [
                    (0.10, (0.86, 0.08, 0.06)),
                    (0.20, (0.88, 0.07, 0.05)),
                ]
            ),
        ]
    )
    detector = StreamingStimulusDetector(
        StimulusDetectionConfig(
            stream_columns=("stream_id",),
            detection_window=(0.0, float("inf")),
            min_consecutive=2,
        ),
        _thresholds(frame, target_classes=["A"]),
    )

    emitted = []
    for observation in frame.sort_values(["stream_id", "time"]).to_dict("records"):
        emitted.extend(detector.update(observation))
    assert emitted == []

    flushed = detector.flush()
    assert len(flushed) == 1
    assert flushed[0]["stimulus_class"] == "A"
    assert flushed[0]["onset_time"] == 0.10
    assert flushed[0]["offset_time"] == 0.20
    assert flushed[0]["detection_confirmed_time"] == 0.20
