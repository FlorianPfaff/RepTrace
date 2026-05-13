from __future__ import annotations

import numpy as np
import pandas as pd

from reptrace.onset._common import (
    DEFAULT_THRESHOLD_QUANTILE,
    DEFAULT_THRESHOLD_WINDOW,
    THRESHOLD_METHODS,
    _group_columns,
    _is_correct_detection,
    _prediction_values,
    _run_duration,
    _sequence_columns,
    _sequence_identity,
)
from reptrace.onset._thresholds import _prepare_thresholded_observations, _threshold_for_group


def _candidate_segments(
    candidates: pd.DataFrame,
    *,
    threshold: float,
    require_stable_prediction: bool,
) -> list[pd.DataFrame]:
    """Return contiguous above-threshold candidate segments.

    When stable predictions are requested, a change in the predicted class breaks
    a segment even if the score remains above threshold.
    """
    if not np.isfinite(threshold) or candidates.empty:
        return []

    segments: list[pd.DataFrame] = []
    start_position: int | None = None
    previous_prediction: object = None
    scores = pd.to_numeric(candidates["_onset_score"], errors="coerce").to_numpy(dtype=float)
    predictions = _prediction_values(candidates)

    for position, score in enumerate(scores):
        above_threshold = bool(np.isfinite(score) and score >= threshold)
        prediction = predictions[position]
        if not above_threshold:
            if start_position is not None:
                segments.append(candidates.iloc[start_position:position])
                start_position = None
                previous_prediction = None
            continue

        prediction_changed = (
            require_stable_prediction
            and start_position is not None
            and previous_prediction is not None
            and prediction != previous_prediction
        )
        if prediction_changed:
            segments.append(candidates.iloc[start_position:position])
            start_position = position
        elif start_position is None:
            start_position = position
        previous_prediction = prediction

    if start_position is not None:
        segments.append(candidates.iloc[start_position:])
    return segments


def _detection_runs(
    candidates: pd.DataFrame,
    *,
    threshold: float,
    min_consecutive: int,
    min_duration: float | None,
    require_stable_prediction: bool,
) -> list[pd.DataFrame]:
    """Return all valid above-threshold onset runs."""

    if min_consecutive < 1:
        raise ValueError("min_consecutive must be at least 1.")
    if min_duration is not None and min_duration < 0:
        raise ValueError("min_duration must be non-negative when provided.")

    runs = []
    for segment in _candidate_segments(
        candidates,
        threshold=threshold,
        require_stable_prediction=require_stable_prediction,
    ):
        if len(segment) < min_consecutive:
            continue
        if min_duration is not None and _run_duration(segment) < min_duration:
            continue
        runs.append(segment)
    return runs


def _first_detection_run(
    candidates: pd.DataFrame,
    *,
    threshold: float,
    min_consecutive: int,
    min_duration: float | None,
    require_stable_prediction: bool,
) -> pd.DataFrame | None:
    runs = _detection_runs(
        candidates,
        threshold=threshold,
        min_consecutive=min_consecutive,
        min_duration=min_duration,
        require_stable_prediction=require_stable_prediction,
    )
    return runs[0] if runs else None


# pylint: disable-next=too-many-arguments,too-many-locals
def _event_row(
    group_values: dict,
    sequence_frame: pd.DataFrame,
    detection_run: pd.DataFrame | None,
    *,
    threshold: float,
    threshold_method: str,
    threshold_window: tuple[float, float],
    threshold_quantile: float,
    score_column: str,
    detection_start: float | None,
    detection_window: tuple[float, float] | None,
    min_consecutive: int,
    min_duration: float | None,
    require_stable_prediction: bool,
) -> dict:
    first_row = sequence_frame.iloc[0]
    detection_row = detection_run.iloc[0] if detection_run is not None and not detection_run.empty else None
    detected = detection_row is not None
    detection_time = float(detection_row["time"]) if detection_row is not None else np.nan
    run_duration = _run_duration(detection_run) if detection_run is not None else np.nan
    event = {
        **group_values,
        **_sequence_identity(first_row),
        "detected": detected,
        "detection_time": detection_time,
        "detection_latency": detection_time,
        "detected_before_zero": bool(detected and detection_time < 0.0),
        "score_threshold": threshold,
        "score_column": score_column,
        "threshold_method": threshold_method,
        "threshold_quantile": threshold_quantile,
        "threshold_window_start": threshold_window[0],
        "threshold_window_stop": threshold_window[1],
        "detection_start": detection_start if detection_start is not None else np.nan,
        "detection_scan_start": detection_window[0] if detection_window is not None else np.nan,
        "detection_scan_stop": detection_window[1] if detection_window is not None else np.nan,
        "min_consecutive": min_consecutive,
        "min_duration": min_duration if min_duration is not None else np.nan,
        "require_stable_prediction": require_stable_prediction,
        "n_time_points": len(sequence_frame),
        "detection_run_length": int(len(detection_run)) if detection_run is not None else 0,
        "detection_run_duration": run_duration,
        "detection_run_stop_time": float(detection_run.iloc[-1]["time"]) if detection_run is not None else np.nan,
        "score_peak_in_run": float(detection_run["_onset_score"].max()) if detection_run is not None else np.nan,
    }
    if detection_row is not None:
        event.update(
            {
                "detection_window_start": detection_row.get("window_start", np.nan),
                "detection_window_stop": detection_row.get("window_stop", np.nan),
                "predicted_label_at_detection": detection_row.get("predicted_label", np.nan),
                "predicted_class_at_detection": detection_row.get("predicted_class", ""),
                "score_at_detection": detection_row["_onset_score"],
                "is_correct_at_detection": _is_correct_detection(detection_row),
            }
        )
    else:
        event.update(
            {
                "detection_window_start": np.nan,
                "detection_window_stop": np.nan,
                "predicted_label_at_detection": np.nan,
                "predicted_class_at_detection": "",
                "score_at_detection": np.nan,
                "is_correct_at_detection": False,
            }
        )
    return event

def detect_onsets(
    observations: pd.DataFrame,
    *,
    threshold_window: tuple[float, float] = DEFAULT_THRESHOLD_WINDOW,
    threshold_quantile: float = DEFAULT_THRESHOLD_QUANTILE,
    score_column: str = "confidence",
    threshold_method: str = "point",
    detection_start: float | None = None,
    detection_window: tuple[float, float] | None = None,
    min_consecutive: int = 1,
    min_duration: float | None = None,
    require_stable_prediction: bool = False,
) -> pd.DataFrame:
    """Find the first threshold-crossing time for each probability-observation sequence.

    ``min_consecutive`` and ``min_duration`` can be used to suppress single-bin
    spikes by requiring the threshold crossing to be sustained. With
    ``require_stable_prediction=True``, an onset run is also broken when the
    predicted class changes across adjacent above-threshold bins.
    """

    if not 0.0 <= threshold_quantile <= 1.0:
        raise ValueError("threshold_quantile must be between 0 and 1.")
    if threshold_method not in THRESHOLD_METHODS:
        raise ValueError(f"threshold_method must be one of {THRESHOLD_METHODS}.")
    if min_consecutive < 1:
        raise ValueError("min_consecutive must be at least 1.")
    if min_duration is not None and min_duration < 0:
        raise ValueError("min_duration must be non-negative when provided.")
    if "time" not in observations.columns:
        raise ValueError("Observation rows must contain a time column.")

    observations = _prepare_thresholded_observations(
        observations,
        threshold_window=threshold_window,
        threshold_quantile=threshold_quantile,
        score_column=score_column,
        threshold_method=threshold_method,
        min_consecutive=min_consecutive,
        min_duration=min_duration,
        require_stable_prediction=require_stable_prediction,
    )
    group_columns = _group_columns(observations)
    sequence_columns = _sequence_columns(observations)
    event_rows = []

    grouped = observations.groupby(group_columns, sort=True) if group_columns else [((), observations)]
    for keys, group_frame in grouped:
        key_values = keys if isinstance(keys, tuple) else (keys,)
        group_values = dict(zip(group_columns, key_values, strict=True))
        threshold = (
            group_frame["score_threshold"].iloc[0]
            if "score_threshold" in group_frame
            else _threshold_for_group(
                group_frame,
                threshold_window=threshold_window,
                threshold_quantile=threshold_quantile,
                score_column=score_column,
                threshold_method=threshold_method,
                min_consecutive=min_consecutive,
                min_duration=min_duration,
                require_stable_prediction=require_stable_prediction,
            )
        )
        sorted_group = group_frame.sort_values([*sequence_columns, "time"])
        for _, sequence_frame in sorted_group.groupby(sequence_columns, sort=True):
            candidates = sequence_frame
            if detection_start is not None:
                candidates = candidates.loc[candidates["time"] >= detection_start]
            if detection_window is not None:
                start, stop = detection_window
                candidates = candidates.loc[(candidates["time"] >= start) & (candidates["time"] <= stop)]
            detection_run = _first_detection_run(
                candidates,
                threshold=threshold,
                min_consecutive=min_consecutive,
                min_duration=min_duration,
                require_stable_prediction=require_stable_prediction,
            )
            event_rows.append(
                _event_row(
                    group_values,
                    sequence_frame,
                    detection_run,
                    threshold=threshold,
                    threshold_method=threshold_method,
                    threshold_window=threshold_window,
                    threshold_quantile=threshold_quantile,
                    score_column=score_column,
                    detection_start=detection_start,
                    detection_window=detection_window,
                    min_consecutive=min_consecutive,
                    min_duration=min_duration,
                    require_stable_prediction=require_stable_prediction,
                )
            )
    return pd.DataFrame(event_rows)
