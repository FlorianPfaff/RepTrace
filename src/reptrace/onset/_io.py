from __future__ import annotations

from pathlib import Path

import pandas as pd

from reptrace.onset._common import (
    DEFAULT_DETECTION_WINDOW,
    DEFAULT_THRESHOLD_QUANTILE,
    DEFAULT_THRESHOLD_WINDOW,
)
from reptrace.onset._events import detect_onsets
from reptrace.onset._summaries import summarize_onset_events, summarize_threshold_crossings
from reptrace.onset._thresholds import annotate_threshold_crossings
from reptrace.temporal_model import read_probability_observations


def detect_onsets_from_csvs(
    observation_csvs: list[Path],
    *,
    threshold_window: tuple[float, float] = DEFAULT_THRESHOLD_WINDOW,
    threshold_quantile: float = DEFAULT_THRESHOLD_QUANTILE,
    score_column: str = "confidence",
    threshold_method: str = "point",
    detection_start: float | None = None,
    event_window: tuple[float, float] | None = None,
    min_consecutive: int = 1,
    min_duration: float | None = None,
    require_stable_prediction: bool = False,
    out_events: Path | None = None,
    out_summary: Path | None = None,
    out_thresholded_observations: Path | None = None,
    out_threshold_summary: Path | None = None,
    detection_window: tuple[float, float] = DEFAULT_DETECTION_WINDOW,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read probability observations, detect onsets, and optionally write CSV outputs."""

    observations = read_probability_observations(observation_csvs)
    thresholded_observations = annotate_threshold_crossings(
        observations,
        threshold_window=threshold_window,
        threshold_quantile=threshold_quantile,
        score_column=score_column,
        threshold_method=threshold_method,
        min_consecutive=min_consecutive,
        min_duration=min_duration,
        require_stable_prediction=require_stable_prediction,
    )
    events = detect_onsets(
        thresholded_observations,
        threshold_window=threshold_window,
        threshold_quantile=threshold_quantile,
        score_column=score_column,
        threshold_method=threshold_method,
        detection_start=detection_start,
        detection_window=event_window,
        min_consecutive=min_consecutive,
        min_duration=min_duration,
        require_stable_prediction=require_stable_prediction,
    )
    summary = summarize_onset_events(events)
    threshold_summary = summarize_threshold_crossings(
        thresholded_observations,
        baseline_window=threshold_window,
        detection_window=detection_window,
    )
    if out_events is not None:
        out_events.parent.mkdir(parents=True, exist_ok=True)
        events.to_csv(out_events, index=False)
    if out_summary is not None:
        out_summary.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(out_summary, index=False)
    if out_thresholded_observations is not None:
        out_thresholded_observations.parent.mkdir(parents=True, exist_ok=True)
        thresholded_observations.drop(columns=["_onset_score"], errors="ignore").to_csv(out_thresholded_observations, index=False)
    if out_threshold_summary is not None:
        out_threshold_summary.parent.mkdir(parents=True, exist_ok=True)
        threshold_summary.to_csv(out_threshold_summary, index=False)
    return events, summary
