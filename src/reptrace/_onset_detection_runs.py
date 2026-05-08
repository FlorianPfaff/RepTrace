from __future__ import annotations

from collections.abc import Callable

import pandas as pd

Segmenter = Callable[..., list[pd.DataFrame]]
DurationFunction = Callable[[pd.DataFrame], float]


def _validate_persistence_constraints(
    *,
    min_consecutive: int,
    min_duration: float | None,
) -> None:
    if min_consecutive < 1:
        raise ValueError("min_consecutive must be at least 1.")
    if min_duration is not None and min_duration < 0:
        raise ValueError("min_duration must be non-negative when provided.")


def detection_runs(
    candidates: pd.DataFrame,
    *,
    threshold: float,
    min_consecutive: int,
    min_duration: float | None,
    require_stable_prediction: bool,
    segmenter: Segmenter,
    duration: DurationFunction,
) -> list[pd.DataFrame]:
    """Return all valid above-threshold onset runs in temporal order.

    The low-level segmentation and duration callbacks keep this helper reusable
    while allowing ``reptrace.onset_detection`` to remain the source of truth for
    how adjacent threshold crossings and window durations are interpreted.
    """

    _validate_persistence_constraints(
        min_consecutive=min_consecutive,
        min_duration=min_duration,
    )
    runs: list[pd.DataFrame] = []
    for segment in segmenter(
        candidates,
        threshold=threshold,
        require_stable_prediction=require_stable_prediction,
    ):
        if len(segment) < min_consecutive:
            continue
        if min_duration is not None and duration(segment) < min_duration:
            continue
        runs.append(segment)
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
