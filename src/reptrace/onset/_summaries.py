from __future__ import annotations

import numpy as np
import pandas as pd

from reptrace.onset._common import (
    DEFAULT_DETECTION_WINDOW,
    DEFAULT_THRESHOLD_WINDOW,
    _group_columns,
    _sequence_columns,
    _window_mask,
)


def _sequence_crossing_rate(frame: pd.DataFrame, sequence_columns: list[str]) -> tuple[int, float]:
    if frame.empty:
        return 0, np.nan
    sequence_count = 0
    crossing_count = 0
    for _, sequence_frame in frame.groupby(sequence_columns, sort=True):
        sequence_count += 1
        crossing_count += bool(sequence_frame["above_threshold"].any())
    return crossing_count, crossing_count / sequence_count if sequence_count else np.nan

def _window_threshold_stats(frame: pd.DataFrame, window: tuple[float, float], sequence_columns: list[str]) -> dict[str, float | int]:
    window_frame = frame.loc[_window_mask(frame, window)]
    above_threshold = window_frame["above_threshold"].astype(bool)
    sequence_crossing_count, sequence_crossing_rate = _sequence_crossing_rate(window_frame, sequence_columns)
    stats = {
        "n_observations": len(window_frame),
        "threshold_crossing_count": int(above_threshold.sum()),
        "threshold_crossing_rate": float(above_threshold.mean()) if len(above_threshold) else np.nan,
        "sequence_crossing_count": int(sequence_crossing_count),
        "sequence_crossing_rate": float(sequence_crossing_rate) if np.isfinite(sequence_crossing_rate) else np.nan,
    }
    if "is_correct" in window_frame.columns:
        correct_crossings = window_frame.loc[above_threshold, "is_correct"].astype(bool)
        stats["correct_crossing_count"] = int(correct_crossings.sum())
        stats["correct_crossing_rate"] = float(correct_crossings.mean()) if len(correct_crossings) else np.nan
    return stats

def summarize_threshold_crossings(
    thresholded_observations: pd.DataFrame,
    *,
    baseline_window: tuple[float, float] = DEFAULT_THRESHOLD_WINDOW,
    detection_window: tuple[float, float] = DEFAULT_DETECTION_WINDOW,
) -> pd.DataFrame:
    """Summarize baseline false positives separately from post-event detections."""

    group_columns = _group_columns(thresholded_observations)
    sequence_columns = _sequence_columns(thresholded_observations)
    rows = []
    grouped = thresholded_observations.groupby(group_columns, sort=True) if group_columns else [((), thresholded_observations)]
    for keys, group_frame in grouped:
        key_values = keys if isinstance(keys, tuple) else (keys,)
        group_values = dict(zip(group_columns, key_values, strict=True))
        baseline_stats = _window_threshold_stats(group_frame, baseline_window, sequence_columns)
        detection_stats = _window_threshold_stats(group_frame, detection_window, sequence_columns)
        rows.append(
            {
                **group_values,
                "score_threshold": group_frame["score_threshold"].iloc[0] if "score_threshold" in group_frame else np.nan,
                "score_column": group_frame["score_column"].iloc[0] if "score_column" in group_frame else "",
                "threshold_method": group_frame["threshold_method"].iloc[0] if "threshold_method" in group_frame else "",
                "threshold_quantile": group_frame["threshold_quantile"].iloc[0] if "threshold_quantile" in group_frame else np.nan,
                "baseline_window_start": baseline_window[0],
                "baseline_window_stop": baseline_window[1],
                "detection_window_start": detection_window[0],
                "detection_window_stop": detection_window[1],
                "baseline_n_observations": baseline_stats["n_observations"],
                "baseline_false_positive_count": baseline_stats["threshold_crossing_count"],
                "baseline_false_positive_rate": baseline_stats["threshold_crossing_rate"],
                "baseline_false_positive_sequence_count": baseline_stats["sequence_crossing_count"],
                "baseline_false_positive_sequence_rate": baseline_stats["sequence_crossing_rate"],
                "post_stimulus_n_observations": detection_stats["n_observations"],
                "post_stimulus_detection_count": detection_stats["threshold_crossing_count"],
                "post_stimulus_detection_rate": detection_stats["threshold_crossing_rate"],
                "post_stimulus_detection_sequence_count": detection_stats["sequence_crossing_count"],
                "post_stimulus_detection_sequence_rate": detection_stats["sequence_crossing_rate"],
                "post_stimulus_correct_detection_count": detection_stats.get("correct_crossing_count", np.nan),
                "post_stimulus_correct_detection_rate": detection_stats.get("correct_crossing_rate", np.nan),
            }
        )
    return pd.DataFrame(rows)


# pylint: disable-next=too-many-arguments,too-many-locals

def summarize_onset_events(events: pd.DataFrame) -> pd.DataFrame:
    """Summarize onset-detection events by subject/decoder/emission group."""

    group_columns = _group_columns(events)
    rows = []
    grouped = events.groupby(group_columns, sort=True) if group_columns else [((), events)]
    for keys, group_frame in grouped:
        key_values = keys if isinstance(keys, tuple) else (keys,)
        group_values = dict(zip(group_columns, key_values, strict=True))
        detected = group_frame["detected"].astype(bool)
        false_alarm = group_frame["detected_before_zero"].astype(bool)
        correct = group_frame["is_correct_at_detection"].astype(bool)
        post_detected = detected & ~false_alarm
        latencies = pd.to_numeric(group_frame.loc[post_detected, "detection_latency"], errors="coerce").dropna()
        run_durations = pd.to_numeric(
            group_frame.loc[post_detected, "detection_run_duration"],
            errors="coerce",
        ).dropna()
        run_lengths = pd.to_numeric(
            group_frame.loc[post_detected, "detection_run_length"],
            errors="coerce",
        ).dropna()
        rows.append(
            {
                **group_values,
                "n_sequences": len(group_frame),
                "detected_count": int(detected.sum()),
                "detected_rate": float(detected.mean()) if len(detected) else np.nan,
                "false_alarm_count": int(false_alarm.sum()),
                "false_alarm_rate": float(false_alarm.mean()) if len(false_alarm) else np.nan,
                "post_zero_detected_count": int(post_detected.sum()),
                "post_zero_detected_rate": float(post_detected.mean()) if len(post_detected) else np.nan,
                "correct_detection_count": int((detected & correct).sum()),
                "correct_detection_rate": float((detected & correct).mean()) if len(correct) else np.nan,
                "post_detection_latency_mean": float(latencies.mean()) if not latencies.empty else np.nan,
                "post_detection_latency_median": float(latencies.median()) if not latencies.empty else np.nan,
                "post_detection_run_duration_mean": float(run_durations.mean()) if not run_durations.empty else np.nan,
                "post_detection_run_duration_median": (
                    float(run_durations.median()) if not run_durations.empty else np.nan
                ),
                "post_detection_run_length_median": (
                    float(run_lengths.median()) if not run_lengths.empty else np.nan
                ),
                "score_threshold": (
                    group_frame["score_threshold"].iloc[0]
                    if "score_threshold" in group_frame
                    else np.nan
                ),
                "threshold_method": (
                    group_frame["threshold_method"].iloc[0]
                    if "threshold_method" in group_frame
                    else ""
                ),
                "threshold_quantile": (
                    group_frame["threshold_quantile"].iloc[0]
                    if "threshold_quantile" in group_frame
                    else np.nan
                ),
                "threshold_window_start": (
                    group_frame["threshold_window_start"].iloc[0]
                    if "threshold_window_start" in group_frame
                    else np.nan
                ),
                "threshold_window_stop": (
                    group_frame["threshold_window_stop"].iloc[0]
                    if "threshold_window_stop" in group_frame
                    else np.nan
                ),
                "min_consecutive": group_frame["min_consecutive"].iloc[0] if "min_consecutive" in group_frame else 1,
                "min_duration": group_frame["min_duration"].iloc[0] if "min_duration" in group_frame else np.nan,
                "require_stable_prediction": group_frame["require_stable_prediction"].iloc[0]
                if "require_stable_prediction" in group_frame
                else False,
            }
        )
    return pd.DataFrame(rows)


# pylint: disable-next=too-many-arguments,too-many-locals
