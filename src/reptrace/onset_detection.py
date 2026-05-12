from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd

from reptrace.temporal_model import probability_columns, read_probability_observations

DEFAULT_THRESHOLD_WINDOW = (-0.35, -0.05)
DEFAULT_DETECTION_WINDOW = (0.0, float("inf"))
DEFAULT_THRESHOLD_QUANTILE = 0.95
THRESHOLD_METHODS = ("point", "max_run")
GROUP_COLUMNS = ("subject", "decoder", "emission_mode")


def _expand_paths(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if matches:
            paths.extend(Path(match) for match in matches)
        else:
            paths.append(Path(pattern))
    return paths


def _group_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in GROUP_COLUMNS if column in frame.columns]


def _sequence_columns(frame: pd.DataFrame) -> list[str]:
    columns = [column for column in ("subject", "fold", "sequence_id") if column in frame.columns]
    if "sequence_id" not in columns:
        raise ValueError("Observation rows must contain sequence_id or sample_index.")
    return columns


def _window_mask(frame: pd.DataFrame, window: tuple[float, float]) -> pd.Series:
    start, stop = window
    return (frame["time"] >= start) & (frame["time"] <= stop)


def _score_values(frame: pd.DataFrame, score_column: str) -> pd.Series:
    if score_column in frame.columns:
        return pd.to_numeric(frame[score_column], errors="coerce")
    prob_columns = probability_columns(frame)
    if score_column == "confidence":
        return frame[prob_columns].max(axis=1)
    if score_column == "probability_true_class" and "true_label" in frame.columns:
        probabilities = frame[prob_columns].to_numpy(dtype=float)
        true_labels = pd.to_numeric(frame["true_label"], errors="coerce").to_numpy()
        scores = np.full(len(frame), np.nan, dtype=float)
        valid = np.isfinite(true_labels)
        valid_indices = true_labels[valid].astype(int)
        in_bounds = (valid_indices >= 0) & (valid_indices < probabilities.shape[1])
        valid_positions = np.flatnonzero(valid)[in_bounds]
        scores[valid_positions] = probabilities[valid_positions, valid_indices[in_bounds]]
        return pd.Series(scores, index=frame.index)
    raise ValueError(f"Score column '{score_column}' is missing and cannot be inferred.")


def _class_lookup(frame: pd.DataFrame) -> dict[int, str]:
    lookup: dict[int, str] = {}
    for column in frame.columns:
        if not column.startswith("class_"):
            continue
        try:
            class_index = int(column.removeprefix("class_"))
        except ValueError:
            continue
        values = frame[column].dropna()
        if not values.empty:
            lookup[class_index] = str(values.iloc[0])
    return lookup


def _ensure_prediction_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    if "predicted_label" in frame.columns and "predicted_class" in frame.columns:
        return frame
    prob_columns = probability_columns(frame)
    probabilities = frame[prob_columns].to_numpy(dtype=float)
    predicted_labels = probabilities.argmax(axis=1)
    if "predicted_label" in frame.columns:
        parsed_labels = pd.to_numeric(frame["predicted_label"], errors="coerce")
        if parsed_labels.notna().all():
            predicted_labels = parsed_labels.astype(int).to_numpy()
    if "predicted_label" not in frame.columns:
        frame["predicted_label"] = predicted_labels
    if "predicted_class" not in frame.columns:
        lookup = _class_lookup(frame)
        frame["predicted_class"] = [lookup.get(int(label), str(int(label))) for label in predicted_labels]
    return frame


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


def _prediction_values(frame: pd.DataFrame) -> np.ndarray:
    if "predicted_label" in frame.columns:
        return frame["predicted_label"].to_numpy(dtype=object)
    if "predicted_class" in frame.columns:
        return frame["predicted_class"].to_numpy(dtype=object)
    return np.full(len(frame), None, dtype=object)


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


def _sequence_identity(row: pd.Series) -> dict:
    identity = {
        "sequence_id": row["sequence_id"],
    }
    for optional_column in ("sample_index", "group", "source_file"):
        if optional_column in row:
            identity[optional_column] = row[optional_column]
    for truth_column in ("true_label", "true_class"):
        if truth_column in row:
            identity[truth_column] = row[truth_column]
    return identity


def _run_duration(run: pd.DataFrame) -> float:
    if run.empty:
        return float("nan")
    if {"window_start", "window_stop"}.issubset(run.columns):
        start = pd.to_numeric(run["window_start"], errors="coerce").iloc[0]
        stop = pd.to_numeric(run["window_stop"], errors="coerce").iloc[-1]
        if pd.notna(start) and pd.notna(stop):
            return float(stop - start)
    times = pd.to_numeric(run["time"], errors="coerce").dropna().to_numpy(dtype=float)
    if len(times) < 2:
        return 0.0
    return float(times[-1] - times[0])


def _prediction_value(row: pd.Series) -> object:
    if "predicted_label" in row and pd.notna(row["predicted_label"]):
        return row["predicted_label"]
    if "predicted_class" in row and pd.notna(row["predicted_class"]):
        return row["predicted_class"]
    return None


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


def _first_detection_run(
    candidates: pd.DataFrame,
    *,
    threshold: float,
    min_consecutive: int,
    min_duration: float | None,
    require_stable_prediction: bool,
) -> pd.DataFrame | None:
    if min_consecutive < 1:
        raise ValueError("min_consecutive must be at least 1.")
    if min_duration is not None and min_duration < 0:
        raise ValueError("min_duration must be non-negative when provided.")

    for segment in _candidate_segments(
        candidates,
        threshold=threshold,
        require_stable_prediction=require_stable_prediction,
    ):
        if len(segment) < min_consecutive:
            continue
        if min_duration is not None and _run_duration(segment) < min_duration:
            continue
        return segment
    return None


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


def _is_correct_detection(row: pd.Series) -> bool:
    has_label_columns = "true_label" in row and "predicted_label" in row
    if (
        has_label_columns
        and pd.notna(row["true_label"])
        and pd.notna(row["predicted_label"])
    ):
        return int(row["true_label"]) == int(row["predicted_label"])

    has_class_columns = "true_class" in row and "predicted_class" in row
    if (
        has_class_columns
        and pd.notna(row["true_class"])
        and pd.notna(row["predicted_class"])
    ):
        return str(row["true_class"]) == str(row["predicted_class"])
    return False


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
    event_detection_window = event_window if event_window is not None else detection_window
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
        detection_window=event_detection_window,
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detect representation onsets from RepTrace probability observation CSVs."
    )
    parser.add_argument(
        "observation_csv",
        nargs="+",
        help="Observation CSVs or glob patterns emitted by RepTrace/PyMEGDec adapters.",
    )
    parser.add_argument("--out-events", type=Path, required=True)
    parser.add_argument("--out-summary", type=Path, required=True)
    parser.add_argument(
        "--threshold-window",
        nargs=2,
        type=float,
        default=DEFAULT_THRESHOLD_WINDOW,
        metavar=("START", "STOP"),
    )
    parser.add_argument("--threshold-quantile", type=float, default=DEFAULT_THRESHOLD_QUANTILE)
    parser.add_argument("--threshold-method", choices=THRESHOLD_METHODS, default="point")
    parser.add_argument("--score-column", default="confidence")
    parser.add_argument("--detection-start", type=float)
    parser.add_argument("--event-window", nargs=2, type=float, metavar=("START", "STOP"))
    parser.add_argument("--detection-window", nargs=2, type=float, default=DEFAULT_DETECTION_WINDOW, metavar=("START", "STOP"))
    parser.add_argument("--out-thresholded-observations", type=Path)
    parser.add_argument("--out-threshold-summary", type=Path)
    parser.add_argument(
        "--min-consecutive",
        type=int,
        default=1,
        help="Minimum number of adjacent above-threshold windows required for an onset.",
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        help="Minimum duration in seconds for an above-threshold onset run.",
    )
    parser.add_argument(
        "--require-stable-prediction",
        action="store_true",
        help="Require the predicted class to remain stable across the onset run.",
    )
    args = parser.parse_args()

    events, summary = detect_onsets_from_csvs(
        _expand_paths(args.observation_csv),
        threshold_window=tuple(args.threshold_window),
        threshold_quantile=args.threshold_quantile,
        score_column=args.score_column,
        threshold_method=args.threshold_method,
        detection_start=args.detection_start,
        event_window=tuple(args.event_window) if args.event_window is not None else None,
        detection_window=tuple(args.detection_window),
        min_consecutive=args.min_consecutive,
        min_duration=args.min_duration,
        require_stable_prediction=args.require_stable_prediction,
        out_events=args.out_events,
        out_summary=args.out_summary,
        out_thresholded_observations=args.out_thresholded_observations,
        out_threshold_summary=args.out_threshold_summary,
    )
    print(f"Wrote onset events: {args.out_events}")
    print(f"Wrote onset summary: {args.out_summary}")
    print(summary.to_string(index=False))
    if events.empty:
        print("No event rows were generated.")


if __name__ == "__main__":
    main()
