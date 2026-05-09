from __future__ import annotations

from collections.abc import Callable

import pandas as pd

Segmenter = Callable[..., list[pd.DataFrame]]
DurationFunction = Callable[[pd.DataFrame], float]


def _validate_persistence_constraints(
    *,
    min_consecutive: int,
    min_duration: float | None,
    merge_gap: float | None,
    refractory: float | None,
) -> None:
    if min_consecutive < 1:
        raise ValueError("min_consecutive must be at least 1.")
    if min_duration is not None and min_duration < 0:
        raise ValueError("min_duration must be non-negative when provided.")
    if merge_gap is not None and merge_gap < 0:
        raise ValueError("merge_gap must be non-negative when provided.")
    if refractory is not None and refractory < 0:
        raise ValueError("refractory must be non-negative when provided.")


def _merge_close_runs(
    runs: list[pd.DataFrame],
    *,
    merge_gap: float | None,
    require_stable_prediction: bool,
) -> list[pd.DataFrame]:
    if merge_gap is None or len(runs) <= 1:
        return runs

    merged = [runs[0]]
    for run in runs[1:]:
        previous = merged[-1]
        previous_stop = float(previous.iloc[-1]["time"])
        current_start = float(run.iloc[0]["time"])
        if require_stable_prediction and "predicted_label" in previous.columns and "predicted_label" in run.columns:
            previous_prediction = previous.iloc[-1]["predicted_label"]
            current_prediction = run.iloc[0]["predicted_label"]
            if previous_prediction != current_prediction:
                merged.append(run)
                continue
        if _gap_between_runs(previous, run, start=current_start, stop=previous_stop) <= merge_gap:
            merged[-1] = pd.concat([previous, run], ignore_index=False)
        else:
            merged.append(run)
    return merged


def _gap_between_runs(previous: pd.DataFrame, run: pd.DataFrame, *, start: float, stop: float) -> float:
    gap = start - stop
    nominal_step = _nominal_time_step(previous, run)
    if nominal_step is None:
        return gap
    return max(0.0, gap - nominal_step)


def _nominal_time_step(*runs: pd.DataFrame) -> float | None:
    steps: list[float] = []
    for run in runs:
        if "time" not in run.columns or len(run) < 2:
            continue
        times = run["time"].astype(float).sort_values()
        diffs = times.diff().dropna()
        steps.extend(float(diff) for diff in diffs if diff > 0)
    if not steps:
        return None
    return float(pd.Series(steps).median())


def detection_runs(
    candidates: pd.DataFrame,
    *,
    threshold: float,
    min_consecutive: int,
    min_duration: float | None,
    require_stable_prediction: bool,
    segmenter: Segmenter,
    duration: DurationFunction,
    merge_gap: float | None = None,
    refractory: float | None = None,
) -> list[pd.DataFrame]:
    """Return all valid above-threshold onset runs in temporal order.

    The low-level segmentation and duration callbacks keep this helper reusable
    while allowing ``reptrace.onset_detection`` to remain the source of truth for
    how adjacent threshold crossings and window durations are interpreted.
    """

    _validate_persistence_constraints(
        min_consecutive=min_consecutive,
        min_duration=min_duration,
        merge_gap=merge_gap,
        refractory=refractory,
    )
    segments = segmenter(
        candidates,
        threshold=threshold,
        require_stable_prediction=require_stable_prediction,
    )
    segments = _merge_close_runs(
        segments,
        merge_gap=merge_gap,
        require_stable_prediction=require_stable_prediction,
    )

    runs: list[pd.DataFrame] = []
    last_onset: float | None = None
    for segment in segments:
        if len(segment) < min_consecutive:
            continue
        if min_duration is not None and duration(segment) < min_duration:
            continue
        onset = float(segment.iloc[0]["time"])
        if refractory is not None and last_onset is not None and onset - last_onset < refractory:
            continue
        runs.append(segment)
        last_onset = onset
    return runs


def first_detection_run(
    candidates: pd.DataFrame,
    *,
    threshold: float,
    min_consecutive: int,
    min_duration: float | None,
    require_stable_prediction: bool,
    segmenter: Segmenter,
    duration: DurationFunction,
) -> pd.DataFrame | None:
    """Return the first valid above-threshold onset run, if any."""

    runs = detection_runs(
        candidates,
        threshold=threshold,
        min_consecutive=min_consecutive,
        min_duration=min_duration,
        require_stable_prediction=require_stable_prediction,
        segmenter=segmenter,
        duration=duration,
    )
    return runs[0] if runs else None
