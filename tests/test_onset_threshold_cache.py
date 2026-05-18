import pandas as pd

from reptrace.onset_detection import annotate_threshold_crossings, detect_onsets


def _single_sequence_frame(scores: list[tuple[float, float]]) -> pd.DataFrame:
    rows = []
    for time, confidence in scores:
        rows.append(
            {
                "subject": "sub-01",
                "decoder": "logistic",
                "emission_mode": "calibrated",
                "time": time,
                "sequence_id": 0,
                "sample_index": 0,
                "confidence": confidence,
                "prob_class_0": confidence,
                "prob_class_1": 1.0 - confidence,
            }
        )
    return pd.DataFrame(rows)


def test_detect_onsets_recomputes_cached_threshold_when_window_changes() -> None:
    observations = _single_sequence_frame(
        [(-0.30, 0.20), (-0.20, 0.40), (-0.10, 0.90), (0.10, 0.50)]
    )
    thresholded_with_old_window = annotate_threshold_crossings(
        observations,
        threshold_window=(-0.20, -0.10),
        threshold_quantile=1.0,
    )

    events = detect_onsets(
        thresholded_with_old_window,
        threshold_window=(-0.30, -0.20),
        threshold_quantile=1.0,
        detection_start=0.0,
    )

    event = events.iloc[0]
    assert event["detected"]
    assert event["detection_time"] == 0.10
    assert event["score_threshold"] == 0.40
    assert event["threshold_window_start"] == -0.30
    assert event["threshold_window_stop"] == -0.20


def test_detect_onsets_recomputes_cached_threshold_when_persistence_changes() -> None:
    observations = _single_sequence_frame(
        [(-0.30, 0.90), (-0.20, 0.20), (0.10, 0.85), (0.20, 0.86)]
    )
    thresholded_with_single_bin_runs = annotate_threshold_crossings(
        observations,
        threshold_window=(-0.30, -0.20),
        threshold_quantile=1.0,
        threshold_method="max_run",
        min_consecutive=1,
    )

    events = detect_onsets(
        thresholded_with_single_bin_runs,
        threshold_window=(-0.30, -0.20),
        threshold_quantile=1.0,
        threshold_method="max_run",
        detection_start=0.0,
        min_consecutive=2,
    )

    event = events.iloc[0]
    assert event["detected"]
    assert event["detection_time"] == 0.10
    assert event["detection_run_length"] == 2
    assert event["score_threshold"] == 0.20
    assert event["min_consecutive"] == 2
