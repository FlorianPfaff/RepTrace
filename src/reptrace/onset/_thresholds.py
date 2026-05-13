from __future__ import annotations

import numpy as np
import pandas as pd

from reptrace.onset._common import (
    DEFAULT_THRESHOLD_QUANTILE,
    DEFAULT_THRESHOLD_WINDOW,
    THRESHOLD_METHODS,
    _ensure_prediction_columns,
    _group_columns,
    _prediction_values,
    _score_values,
    _sequence_columns,
)


def _threshold_for_group(
    frame: pd.DataFrame,
    *,
    threshold_window: tuple[float, float],
    threshold_quantile: float,
    score_column: str,
    threshold_method: str = "point",
    min_consecutive: int = 1,
    min_duration: float | None = None,
    require_stable_prediction: bool = False,
) -> float:
    if threshold_method not in THRESHOLD_METHODS:
        raise ValueError(f"threshold_method must be one of {THRESHOLD_METHODS}.")
    start, stop = threshold_window
    baseline = frame.loc[(frame["time"] >= start) & (frame["time"] <= stop)]
    if threshold_method == "max_run":
        sequence_columns = _sequence_columns(frame)
        scores = _baseline_run_null_scores(
            baseline,
            sequence_columns=sequence_columns,
            score_column=score_column,
            min_consecutive=min_consecutive,
            min_duration=min_duration,
            require_stable_prediction=require_stable_prediction,
        ).dropna()
    else:
        scores = _score_values(baseline, score_column).dropna()
    if scores.empty:
        return np.nan
    return float(scores.quantile(threshold_quantile))

def _sequence_run_duration(frame: pd.DataFrame, start: int, stop: int) -> float:
    if {"window_start", "window_stop"}.issubset(frame.columns):
        starts = pd.to_numeric(frame["window_start"], errors="coerce").to_numpy(dtype=float)
        stops = pd.to_numeric(frame["window_stop"], errors="coerce").to_numpy(dtype=float)
        if np.isfinite(starts[start]) and np.isfinite(stops[stop]):
            return float(stops[stop] - starts[start])
    times = pd.to_numeric(frame["time"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(times[start]) or not np.isfinite(times[stop]):
        return float("nan")
    return float(times[stop] - times[start])

def _valid_run_score_candidates(
    sequence_frame: pd.DataFrame,
    *,
    min_consecutive: int,
    min_duration: float | None,
    require_stable_prediction: bool,
) -> list[float]:
    if min_consecutive < 1:
        raise ValueError("min_consecutive must be at least 1.")
    if min_duration is not None and min_duration < 0:
        raise ValueError("min_duration must be non-negative when provided.")

    sequence_frame = sequence_frame.sort_values("time").reset_index(drop=True)
    score_values = pd.to_numeric(sequence_frame["_onset_score"], errors="coerce").to_numpy(dtype=float)
    predictions = _prediction_values(sequence_frame)
    scores: list[float] = []
    for start in range(len(score_values)):
        previous_prediction = None
        run_min = float("inf")
        for stop in range(start, len(score_values)):
            score = score_values[stop]
            if not np.isfinite(score):
                break
            run_min = min(run_min, float(score))
            prediction = predictions[stop]
            if (
                require_stable_prediction
                and previous_prediction is not None
                and prediction != previous_prediction
            ):
                break
            previous_prediction = prediction
            if stop - start + 1 < min_consecutive:
                continue
            if min_duration is not None and _sequence_run_duration(sequence_frame, start, stop) < min_duration:
                continue
            scores.append(run_min)
    return [score for score in scores if np.isfinite(score)]

def _baseline_run_null_scores(
    baseline: pd.DataFrame,
    *,
    sequence_columns: list[str],
    score_column: str,
    min_consecutive: int,
    min_duration: float | None,
    require_stable_prediction: bool,
) -> pd.Series:
    """Return one max-run null score per baseline sequence.

    Each score is the largest threshold that would still produce a valid
    baseline detection run under the same persistence constraints used for the
    event detector. Quantiling these sequence-level maxima corrects for scanning
    multiple time bins more directly than a pointwise score quantile.
    """

    if baseline.empty:
        return pd.Series(dtype=float)
    scored = baseline.copy()
    scored["_onset_score"] = _score_values(scored, score_column)
    rows = []
    for _, sequence_frame in scored.groupby(sequence_columns, sort=True):
        candidates = _valid_run_score_candidates(
            sequence_frame,
            min_consecutive=min_consecutive,
            min_duration=min_duration,
            require_stable_prediction=require_stable_prediction,
        )
        if candidates:
            rows.append(max(candidates))
    return pd.Series(rows, dtype=float)

def _annotate_group_threshold(
    frame: pd.DataFrame,
    *,
    threshold_window: tuple[float, float],
    threshold_quantile: float,
    score_column: str,
    threshold_method: str,
    min_consecutive: int,
    min_duration: float | None,
    require_stable_prediction: bool,
) -> pd.DataFrame:
    frame = frame.copy()
    frame["_onset_score"] = _score_values(frame, score_column)
    threshold = _threshold_for_group(
        frame,
        threshold_window=threshold_window,
        threshold_quantile=threshold_quantile,
        score_column=score_column,
        threshold_method=threshold_method,
        min_consecutive=min_consecutive,
        min_duration=min_duration,
        require_stable_prediction=require_stable_prediction,
    )
    frame["onset_score"] = frame["_onset_score"]
    frame["score_threshold"] = threshold
    frame["above_threshold"] = np.isfinite(threshold) & (frame["_onset_score"] >= threshold)
    frame["score_column"] = score_column
    frame["threshold_method"] = threshold_method
    frame["threshold_quantile"] = threshold_quantile
    frame["threshold_window_start"] = threshold_window[0]
    frame["threshold_window_stop"] = threshold_window[1]
    if "is_correct" not in frame.columns and {"true_label", "predicted_label"}.issubset(frame.columns):
        frame["is_correct"] = frame["true_label"].astype(int) == frame["predicted_label"].astype(int)
    return frame

def annotate_threshold_crossings(
    observations: pd.DataFrame,
    *,
    threshold_window: tuple[float, float] = DEFAULT_THRESHOLD_WINDOW,
    threshold_quantile: float = DEFAULT_THRESHOLD_QUANTILE,
    score_column: str = "confidence",
    threshold_method: str = "point",
    min_consecutive: int = 1,
    min_duration: float | None = None,
    require_stable_prediction: bool = False,
) -> pd.DataFrame:
    """Annotate observation rows with baseline-derived threshold crossings."""

    if not 0.0 <= threshold_quantile <= 1.0:
        raise ValueError("threshold_quantile must be between 0 and 1.")
    if threshold_method not in THRESHOLD_METHODS:
        raise ValueError(f"threshold_method must be one of {THRESHOLD_METHODS}.")
    if "time" not in observations.columns:
        raise ValueError("Observation rows must contain a time column.")

    observations = _ensure_prediction_columns(observations)
    group_columns = _group_columns(observations)
    frames = []
    grouped = observations.groupby(group_columns, sort=True) if group_columns else [((), observations)]
    for _, group_frame in grouped:
        frames.append(
            _annotate_group_threshold(
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
    return pd.concat(frames, ignore_index=True) if frames else observations.copy()

def _has_matching_threshold_annotation(
    observations: pd.DataFrame,
    *,
    threshold_method: str,
    threshold_quantile: float,
    score_column: str,
) -> bool:
    if "score_threshold" not in observations.columns:
        return False
    if "_onset_score" not in observations.columns and "onset_score" not in observations.columns:
        return False
    checks = {
        "threshold_method": threshold_method,
        "score_column": score_column,
    }
    for column, expected in checks.items():
        if column in observations.columns and not observations[column].dropna().astype(str).eq(str(expected)).all():
            return False
    if "threshold_quantile" in observations.columns:
        quantiles = pd.to_numeric(observations["threshold_quantile"], errors="coerce").dropna()
        if not quantiles.empty and not np.allclose(quantiles.to_numpy(dtype=float), threshold_quantile):
            return False
    return True

def _prepare_thresholded_observations(
    observations: pd.DataFrame,
    *,
    threshold_window: tuple[float, float],
    threshold_quantile: float,
    score_column: str,
    threshold_method: str,
    min_consecutive: int,
    min_duration: float | None,
    require_stable_prediction: bool,
) -> pd.DataFrame:
    if _has_matching_threshold_annotation(
        observations,
        threshold_method=threshold_method,
        threshold_quantile=threshold_quantile,
        score_column=score_column,
    ):
        thresholded = _ensure_prediction_columns(observations).copy()
        if "_onset_score" not in thresholded.columns:
            thresholded["_onset_score"] = pd.to_numeric(thresholded["onset_score"], errors="coerce")
        thresholded["above_threshold"] = thresholded["_onset_score"] >= pd.to_numeric(
            thresholded["score_threshold"],
            errors="coerce",
        )
        return thresholded
    return annotate_threshold_crossings(
        observations,
        threshold_window=threshold_window,
        threshold_quantile=threshold_quantile,
        score_column=score_column,
        threshold_method=threshold_method,
        min_consecutive=min_consecutive,
        min_duration=min_duration,
        require_stable_prediction=require_stable_prediction,
    )
