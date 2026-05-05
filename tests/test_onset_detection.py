from pathlib import Path

import numpy as np
import pandas as pd

from reptrace.onset_detection import detect_onsets, detect_onsets_from_csvs, summarize_onset_events


def _observation_frame() -> pd.DataFrame:
    rows = []
    traces = {
        0: [(-0.20, 0.55), (-0.10, 0.58), (0.05, 0.62), (0.15, 0.92), (0.25, 0.88)],
        1: [(-0.20, 0.57), (-0.10, 0.59), (0.05, 0.90), (0.15, 0.86), (0.25, 0.84)],
        2: [(-0.20, 0.56), (-0.10, 0.91), (0.05, 0.85), (0.15, 0.80), (0.25, 0.77)],
        3: [(-0.20, 0.53), (-0.10, 0.54), (0.05, 0.55), (0.15, 0.56), (0.25, 0.57)],
    }
    for sequence_id, trace in traces.items():
        true_label = sequence_id % 2
        for time, confidence in trace:
            predicted_label = true_label if confidence >= 0.80 else 1 - true_label
            probabilities = np.array([0.0, 0.0])
            probabilities[predicted_label] = confidence
            probabilities[1 - predicted_label] = 1.0 - confidence
            rows.append(
                {
                    "subject": "sub-01",
                    "fold": sequence_id % 2,
                    "decoder": "logistic",
                    "emission_mode": "calibrated",
                    "time": time,
                    "window_start": time - 0.01,
                    "window_stop": time + 0.01,
                    "sample_index": sequence_id,
                    "sequence_id": sequence_id,
                    "true_label": true_label,
                    "true_class": f"class-{true_label}",
                    "predicted_label": predicted_label,
                    "predicted_class": f"class-{predicted_label}",
                    "probability_true_class": probabilities[true_label],
                    "confidence": confidence,
                    "class_0": "class-0",
                    "class_1": "class-1",
                    "prob_class_0": probabilities[0],
                    "prob_class_1": probabilities[1],
                }
            )
    return pd.DataFrame(rows)


def test_detect_onsets_finds_first_threshold_crossing():
    events = detect_onsets(
        _observation_frame(),
        threshold_window=(-0.20, -0.10),
        threshold_quantile=0.875,
    )

    by_sequence = events.set_index("sequence_id")

    assert len(events) == 4
    assert by_sequence.loc[0, "detected"]
    assert by_sequence.loc[0, "detection_time"] == 0.15
    assert by_sequence.loc[1, "detection_time"] == 0.05
    assert by_sequence.loc[2, "detected_before_zero"]
    assert not by_sequence.loc[3, "detected"]
    assert by_sequence.loc[0, "is_correct_at_detection"]
    assert by_sequence.loc[1, "is_correct_at_detection"]


def test_detection_start_excludes_baseline_false_alarms():
    events = detect_onsets(
        _observation_frame(),
        threshold_window=(-0.20, -0.10),
        threshold_quantile=0.875,
        detection_start=0.0,
    )

    row = events.set_index("sequence_id").loc[2]

    assert row["detected"]
    assert not row["detected_before_zero"]
    assert row["detection_time"] == 0.05


def test_summarize_onset_events_reports_detection_rates():
    events = detect_onsets(
        _observation_frame(),
        threshold_window=(-0.20, -0.10),
        threshold_quantile=0.875,
    )

    summary = summarize_onset_events(events)

    assert len(summary) == 1
    row = summary.iloc[0]
    assert row["n_sequences"] == 4
    assert row["detected_count"] == 3
    assert row["false_alarm_count"] == 1
    assert row["correct_detection_count"] == 3
    assert row["post_detection_latency_median"] == 0.10


def test_detect_onsets_from_csvs_writes_outputs(tmp_path: Path):
    observations_path = tmp_path / "observations.csv"
    events_path = tmp_path / "events.csv"
    summary_path = tmp_path / "summary.csv"
    _observation_frame().to_csv(observations_path, index=False)

    events, summary = detect_onsets_from_csvs(
        [observations_path],
        threshold_window=(-0.20, -0.10),
        threshold_quantile=0.875,
        out_events=events_path,
        out_summary=summary_path,
    )

    assert events_path.exists()
    assert summary_path.exists()
    assert len(events) == 4
    assert len(summary) == 1
    written = pd.read_csv(events_path)
    assert written["source_file"].isna().sum() == 0
