from __future__ import annotations

import argparse
import glob
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from reptrace.temporal_model import probability_columns

DEFAULT_THRESHOLD_WINDOW = (-0.35, -0.05)
DEFAULT_THRESHOLD_QUANTILE = 0.95
DEFAULT_GROUP_COLUMNS = ("subject", "decoder", "emission_mode")
DEFAULT_STREAM_FALLBACKS = (
    ("subject", "stream_id"),
    ("subject", "sequence_id"),
    ("stream_id",),
    ("sequence_id",),
    ("sample_index",),
)
THRESHOLD_METHODS = ("point", "max_run")
SCORE_MODES = ("class_probability", "predicted_class_confidence")

EVENT_COLUMNS = [
    "event_index",
    "detected",
    "stimulus_label",
    "stimulus_class",
    "onset_time",
    "offset_time",
    "peak_time",
    "detection_confirmed_time",
    "run_length",
    "run_duration",
    "score_at_onset",
    "peak_score",
    "score_threshold",
    "score_column",
    "score_mode",
    "threshold_method",
    "threshold_quantile",
    "threshold_window_start",
    "threshold_window_stop",
    "min_consecutive",
    "min_duration",
    "merge_gap",
    "refractory",
    "predicted_label_at_peak",
    "predicted_class_at_peak",
]


def _expand_paths(patterns: Sequence[str | Path]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = sorted(glob.glob(str(pattern)))
        if matches:
            paths.extend(Path(match) for match in matches)
        else:
            paths.append(Path(pattern))
    return paths


def _present_columns(frame: pd.DataFrame, columns: Sequence[str]) -> list[str]:
    return [column for column in columns if column in frame.columns]


def _group_columns(frame: pd.DataFrame, group_columns: Sequence[str] | None = None) -> list[str]:
    columns = DEFAULT_GROUP_COLUMNS if group_columns is None else group_columns
    return _present_columns(frame, columns)


def _stream_columns(frame: pd.DataFrame, stream_columns: Sequence[str] | None = None) -> list[str]:
    if stream_columns is not None:
        missing = [column for column in stream_columns if column not in frame.columns]
        if missing:
            raise ValueError(f"Observation rows are missing stream columns: {missing}")
        return list(stream_columns)
    for candidates in DEFAULT_STREAM_FALLBACKS:
        if all(column in frame.columns for column in candidates):
            return list(candidates)
    raise ValueError("Observation rows must contain stream_id, sequence_id, or sample_index.")


def _window_mask(frame: pd.DataFrame, window: tuple[float, float]) -> pd.Series:
    start, stop = window
    return (frame["time"] >= start) & (frame["time"] <= stop)


def _class_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in probability_columns(frame):
        suffix = column.removeprefix("prob_class_")
        try:
            label: int | str = int(suffix)
        except ValueError:
            label = suffix
        class_column = f"class_{suffix}"
        class_name = str(label)
        if class_column in frame.columns and frame[class_column].notna().any():
            class_name = str(frame[class_column].dropna().iloc[0])
        rows.append({"stimulus_label": label, "stimulus_class": class_name, "score_column": column})
    return pd.DataFrame(rows)


def _target_class_table(frame: pd.DataFrame, target_classes: Sequence[str | int] | None) -> pd.DataFrame:
    classes = _class_table(frame)
    if target_classes is None:
        return classes
    requested = {str(value) for value in target_classes}
    keep = classes["stimulus_label"].astype(str).isin(requested) | classes["stimulus_class"].astype(str).isin(requested)
    selected = classes.loc[keep].reset_index(drop=True)
    if selected.empty:
        raise ValueError(f"No target classes matched {list(target_classes)!r}.")
    return selected


def _score_values(
    frame: pd.DataFrame,
    *,
    stimulus_label: int | str,
    stimulus_class: str,
    score_column: str,
    score_mode: str,
) -> pd.Series:
    if score_mode not in SCORE_MODES:
        raise ValueError(f"score_mode must be one of {SCORE_MODES}.")
    if score_mode == "class_probability":
        if score_column not in frame.columns:
            raise ValueError(f"Score column '{score_column}' is missing.")
        return pd.to_numeric(frame[score_column], errors="coerce")

    if "confidence" not in frame.columns:
        raise ValueError("predicted_class_confidence scoring requires a confidence column.")
    confidence = pd.to_numeric(frame["confidence"], errors="coerce").fillna(0.0)
    if "predicted_label" in frame.columns:
        matches = frame["predicted_label"].astype(str).eq(str(stimulus_label))
    elif "predicted_class" in frame.columns:
        matches = frame["predicted_class"].astype(str).eq(str(stimulus_class))
    else:
        probabilities = frame[probability_columns(frame)].to_numpy(dtype=float)
        matches = pd.Series(probabilities.argmax(axis=1).astype(str) == str(stimulus_label), index=frame.index)
    return confidence.where(matches, 0.0)


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


def _valid_run(run: pd.DataFrame, *, min_consecutive: int, min_duration: float | None) -> bool:
    if len(run) < min_consecutive:
        return False
    if min_duration is not None and _run_duration(run) < min_duration:
        return False
    return True


def _contiguous_runs(frame: pd.DataFrame, *, threshold: float) -> list[pd.DataFrame]:
    if frame.empty or not np.isfinite(threshold):
        return []
    sorted_frame = frame.sort_values("time").reset_index(drop=True)
    above = pd.to_numeric(sorted_frame["_stimulus_score"], errors="coerce") >= threshold
    runs: list[pd.DataFrame] = []
    start: int | None = None
    for position, is_above in enumerate(above.to_numpy(dtype=bool)):
        if is_above and start is None:
            start = position
        elif not is_above and start is not None:
            runs.append(sorted_frame.iloc[start:position].copy())
            start = None
    if start is not None:
        runs.append(sorted_frame.iloc[start:].copy())
    return runs


def _merge_close_runs(runs: list[pd.DataFrame], *, merge_gap: float | None) -> list[pd.DataFrame]:
    if merge_gap is None or merge_gap < 0 or len(runs) <= 1:
        return runs
    merged = [runs[0]]
    for run in runs[1:]:
        previous = merged[-1]
        gap = float(run["time"].iloc[0]) - float(previous["time"].iloc[-1])
        if gap <= merge_gap:
            merged[-1] = pd.concat([previous, run], ignore_index=True)
        else:
            merged.append(run)
    return merged


def _confirmed_time(run: pd.DataFrame, *, min_consecutive: int, min_duration: float | None) -> float:
    if run.empty:
        return np.nan
    limit = min(max(min_consecutive, 1), len(run)) - 1
    if min_duration is None:
        return float(run.iloc[limit]["time"])
    for position in range(limit, len(run)):
        segment = run.iloc[: position + 1]
        if _run_duration(segment) >= min_duration:
            return float(segment.iloc[-1]["time"])
    return float(run.iloc[-1]["time"])


def _run_score(run: pd.DataFrame) -> float:
    scores = pd.to_numeric(run["_stimulus_score"], errors="coerce").dropna()
    if scores.empty:
        return np.nan
    return float(scores.min())


def _valid_run_score_candidates(sequence_frame: pd.DataFrame, *, min_consecutive: int, min_duration: float | None) -> list[float]:
    sequence_frame = sequence_frame.sort_values("time").reset_index(drop=True)
    scores: list[float] = []
    for start in range(len(sequence_frame)):
        for stop in range(start, len(sequence_frame)):
            segment = sequence_frame.iloc[start : stop + 1]
            if _valid_run(segment, min_consecutive=min_consecutive, min_duration=min_duration):
                score = _run_score(segment)
                if np.isfinite(score):
                    scores.append(score)
    return scores


def _threshold_for_scores(
    scored: pd.DataFrame,
    *,
    threshold_window: tuple[float, float],
    threshold_quantile: float,
    threshold_method: str,
    stream_columns: Sequence[str],
    min_consecutive: int,
    min_duration: float | None,
) -> float:
    baseline = scored.loc[_window_mask(scored, threshold_window)].copy()
    if baseline.empty:
        return np.nan
    if threshold_method == "point":
        scores = pd.to_numeric(baseline["_stimulus_score"], errors="coerce").dropna()
    elif threshold_method == "max_run":
        null_scores = []
        grouped = baseline.groupby(list(stream_columns), sort=True) if stream_columns else [((), baseline)]
        for _, sequence_frame in grouped:
            candidates = _valid_run_score_candidates(sequence_frame, min_consecutive=min_consecutive, min_duration=min_duration)
            if candidates:
                null_scores.append(max(candidates))
        scores = pd.Series(null_scores, dtype=float)
    else:
        raise ValueError(f"threshold_method must be one of {THRESHOLD_METHODS}.")
    if scores.empty:
        return np.nan
    return float(scores.quantile(threshold_quantile))


def read_stimulus_probability_observations(csv_paths: Sequence[str | Path]) -> pd.DataFrame:
    """Read probability observations for stream-level stimulus detection.

    This reader accepts the usual RepTrace probability-observation columns but is
    intentionally slightly more permissive than the trial-oriented temporal
    model reader: a long stream may identify itself with ``stream_id`` instead of
    ``sequence_id``.
    """
    paths = _expand_paths(csv_paths)
    if not paths:
        raise ValueError("At least one observation CSV path is required.")
    frames = []
    for csv_path in paths:
        frame = pd.read_csv(csv_path)
        if "time" not in frame.columns:
            raise ValueError(f"{csv_path} is missing required column: time")
        probability_columns(frame)
        if "stream_id" not in frame.columns and "sequence_id" not in frame.columns:
            if "sample_index" in frame.columns:
                frame["sequence_id"] = frame["sample_index"]
            else:
                frame["stream_id"] = csv_path.stem
        if "subject" not in frame.columns:
            frame["subject"] = csv_path.stem
        if "decoder" not in frame.columns:
            frame["decoder"] = "decoder"
        if "emission_mode" not in frame.columns:
            frame["emission_mode"] = "calibrated"
        frame["source_file"] = csv_path.name
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def fit_stimulus_detection_thresholds(
    observations: pd.DataFrame,
    *,
    threshold_window: tuple[float, float] = DEFAULT_THRESHOLD_WINDOW,
    threshold_quantile: float = DEFAULT_THRESHOLD_QUANTILE,
    threshold_method: str = "point",
    score_mode: str = "class_probability",
    target_classes: Sequence[str | int] | None = None,
    group_columns: Sequence[str] | None = None,
    stream_columns: Sequence[str] | None = None,
    min_consecutive: int = 1,
    min_duration: float | None = None,
) -> pd.DataFrame:
    """Fit baseline-derived class-specific event-detection thresholds."""
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

    groups = _group_columns(observations, group_columns)
    streams = _stream_columns(observations, stream_columns)
    classes = _target_class_table(observations, target_classes)
    rows = []
    grouped = observations.groupby(groups, sort=True) if groups else [((), observations)]
    for keys, group_frame in grouped:
        key_values = keys if isinstance(keys, tuple) else (keys,)
        group_values = dict(zip(groups, key_values, strict=True))
        for _, class_row in classes.iterrows():
            scored = group_frame.copy()
            scored["_stimulus_score"] = _score_values(
                scored,
                stimulus_label=class_row["stimulus_label"],
                stimulus_class=str(class_row["stimulus_class"]),
                score_column=str(class_row["score_column"]),
                score_mode=score_mode,
            )
            threshold = _threshold_for_scores(
                scored,
                threshold_window=threshold_window,
                threshold_quantile=threshold_quantile,
                threshold_method=threshold_method,
                stream_columns=streams,
                min_consecutive=min_consecutive,
                min_duration=min_duration,
            )
            rows.append(
                {
                    **group_values,
                    "stimulus_label": class_row["stimulus_label"],
                    "stimulus_class": class_row["stimulus_class"],
                    "score_column": class_row["score_column"],
                    "score_mode": score_mode,
                    "score_threshold": threshold,
                    "threshold_method": threshold_method,
                    "threshold_quantile": threshold_quantile,
                    "threshold_window_start": threshold_window[0],
                    "threshold_window_stop": threshold_window[1],
                    "min_consecutive": min_consecutive,
                    "min_duration": np.nan if min_duration is None else min_duration,
                }
            )
    return pd.DataFrame(rows)


def _filter_group(frame: pd.DataFrame, group_values: dict[str, object]) -> pd.DataFrame:
    filtered = frame
    for column, value in group_values.items():
        filtered = filtered.loc[filtered[column].astype(str) == str(value)]
    return filtered


def _event_row(
    *,
    group_values: dict[str, object],
    stream_values: dict[str, object],
    event_index: int,
    run: pd.DataFrame,
    threshold_row: pd.Series,
    min_consecutive: int,
    min_duration: float | None,
    merge_gap: float | None,
    refractory: float | None,
) -> dict:
    peak_position = int(pd.to_numeric(run["_stimulus_score"], errors="coerce").to_numpy(dtype=float).argmax())
    peak_row = run.iloc[peak_position]
    return {
        **group_values,
        **stream_values,
        "event_index": event_index,
        "detected": True,
        "stimulus_label": threshold_row["stimulus_label"],
        "stimulus_class": threshold_row["stimulus_class"],
        "onset_time": float(run.iloc[0]["time"]),
        "offset_time": float(run.iloc[-1]["time"]),
        "peak_time": float(peak_row["time"]),
        "detection_confirmed_time": _confirmed_time(run, min_consecutive=min_consecutive, min_duration=min_duration),
        "run_length": int(len(run)),
        "run_duration": _run_duration(run),
        "score_at_onset": float(run.iloc[0]["_stimulus_score"]),
        "peak_score": float(peak_row["_stimulus_score"]),
        "score_threshold": float(threshold_row["score_threshold"]),
        "score_column": threshold_row["score_column"],
        "score_mode": threshold_row["score_mode"],
        "threshold_method": threshold_row["threshold_method"],
        "threshold_quantile": float(threshold_row["threshold_quantile"]),
        "threshold_window_start": float(threshold_row["threshold_window_start"]),
        "threshold_window_stop": float(threshold_row["threshold_window_stop"]),
        "min_consecutive": min_consecutive,
        "min_duration": np.nan if min_duration is None else min_duration,
        "merge_gap": np.nan if merge_gap is None else merge_gap,
        "refractory": np.nan if refractory is None else refractory,
        "predicted_label_at_peak": peak_row.get("predicted_label", np.nan),
        "predicted_class_at_peak": peak_row.get("predicted_class", ""),
    }


def detect_stimulus_events(
    observations: pd.DataFrame,
    *,
    thresholds: pd.DataFrame | None = None,
    threshold_window: tuple[float, float] = DEFAULT_THRESHOLD_WINDOW,
    threshold_quantile: float = DEFAULT_THRESHOLD_QUANTILE,
    threshold_method: str = "point",
    score_mode: str = "class_probability",
    target_classes: Sequence[str | int] | None = None,
    group_columns: Sequence[str] | None = None,
    stream_columns: Sequence[str] | None = None,
    detection_window: tuple[float, float] | None = None,
    min_consecutive: int = 1,
    min_duration: float | None = None,
    merge_gap: float | None = None,
    refractory: float | None = None,
) -> pd.DataFrame:
    """Detect zero, one, or many stimulus events in probability streams."""
    if min_consecutive < 1:
        raise ValueError("min_consecutive must be at least 1.")
    if min_duration is not None and min_duration < 0:
        raise ValueError("min_duration must be non-negative when provided.")
    if merge_gap is not None and merge_gap < 0:
        raise ValueError("merge_gap must be non-negative when provided.")
    if refractory is not None and refractory < 0:
        raise ValueError("refractory must be non-negative when provided.")
    if "time" not in observations.columns:
        raise ValueError("Observation rows must contain a time column.")

    groups = _group_columns(observations, group_columns)
    streams = _stream_columns(observations, stream_columns)
    if thresholds is None:
        thresholds = fit_stimulus_detection_thresholds(
            observations,
            threshold_window=threshold_window,
            threshold_quantile=threshold_quantile,
            threshold_method=threshold_method,
            score_mode=score_mode,
            target_classes=target_classes,
            group_columns=groups,
            stream_columns=streams,
            min_consecutive=min_consecutive,
            min_duration=min_duration,
        )
    else:
        thresholds = thresholds.copy()

    rows = []
    event_counters: dict[tuple[object, ...], int] = {}
    threshold_group_columns = _present_columns(thresholds, groups)
    for _, threshold_row in thresholds.iterrows():
        if not np.isfinite(float(threshold_row["score_threshold"])):
            continue
        group_values = {column: threshold_row[column] for column in threshold_group_columns}
        group_frame = _filter_group(observations, group_values) if group_values else observations
        if group_frame.empty:
            continue
        scored = group_frame.copy()
        scored["_stimulus_score"] = _score_values(
            scored,
            stimulus_label=threshold_row["stimulus_label"],
            stimulus_class=str(threshold_row["stimulus_class"]),
            score_column=str(threshold_row["score_column"]),
            score_mode=str(threshold_row["score_mode"]),
        )
        if detection_window is not None:
            scored = scored.loc[_window_mask(scored, detection_window)]
        if scored.empty:
            continue
        grouped_streams = scored.groupby(streams, sort=True) if streams else [((), scored)]
        for stream_key, stream_frame in grouped_streams:
            key_values = stream_key if isinstance(stream_key, tuple) else (stream_key,)
            stream_values = dict(zip(streams, key_values, strict=True))
            runs = _contiguous_runs(stream_frame, threshold=float(threshold_row["score_threshold"]))
            runs = _merge_close_runs(runs, merge_gap=merge_gap)
            runs = [run for run in runs if _valid_run(run, min_consecutive=min_consecutive, min_duration=min_duration)]
            stream_counter_key = tuple(stream_values[column] for column in streams)
            last_onset: float | None = None
            for run in runs:
                onset = float(run.iloc[0]["time"])
                if refractory is not None and last_onset is not None and onset - last_onset < refractory:
                    continue
                event_index = event_counters.get(stream_counter_key, 0)
                rows.append(
                    _event_row(
                        group_values=group_values,
                        stream_values=stream_values,
                        event_index=event_index,
                        run=run,
                        threshold_row=threshold_row,
                        min_consecutive=min_consecutive,
                        min_duration=min_duration,
                        merge_gap=merge_gap,
                        refractory=refractory,
                    )
                )
                event_counters[stream_counter_key] = event_index + 1
                last_onset = onset

    if not rows:
        return pd.DataFrame(columns=[*groups, *streams, *EVENT_COLUMNS])
    events = pd.DataFrame(rows)
    sort_columns = [*streams, "onset_time", "stimulus_class"]
    return events.sort_values(sort_columns).reset_index(drop=True)


def _annotation_id(row: pd.Series, fallback: int) -> object:
    for column in ("annotation_id", "event_id", "stimulus_id"):
        if column in row and pd.notna(row[column]):
            return row[column]
    return fallback


def match_stimulus_annotations(
    events: pd.DataFrame,
    annotations: pd.DataFrame,
    *,
    stream_columns: Sequence[str] | None = None,
    match_tolerance: float = 0.1,
    require_class_match: bool = True,
) -> pd.DataFrame:
    """Greedily match detected events to annotated stimulus onsets."""
    if match_tolerance < 0:
        raise ValueError("match_tolerance must be non-negative.")
    if events.empty:
        return events.copy()
    streams = _stream_columns(events, stream_columns)
    if "onset_time" not in annotations.columns:
        raise ValueError("Annotation rows must contain onset_time.")
    matched = events.copy()
    matched["matched_annotation_id"] = np.nan
    matched["matched_annotation_onset_time"] = np.nan
    matched["latency"] = np.nan
    matched["is_true_positive"] = False
    used: set[object] = set()

    for event_index, event in matched.sort_values("onset_time").iterrows():
        candidates = annotations.copy()
        for column in streams:
            if column in candidates.columns:
                candidates = candidates.loc[candidates[column].astype(str) == str(event[column])]
        if require_class_match:
            if "stimulus_class" in candidates.columns:
                candidates = candidates.loc[candidates["stimulus_class"].astype(str) == str(event["stimulus_class"])]
            elif "stimulus_label" in candidates.columns:
                candidates = candidates.loc[candidates["stimulus_label"].astype(str) == str(event["stimulus_label"])]
        if candidates.empty:
            continue
        candidates = candidates.copy()
        candidates["_annotation_id"] = [_annotation_id(row, index) for index, row in candidates.iterrows()]
        candidates = candidates.loc[~candidates["_annotation_id"].isin(used)].copy()
        if candidates.empty:
            continue
        candidates["_abs_latency"] = (pd.to_numeric(candidates["onset_time"], errors="coerce") - float(event["onset_time"])).abs()
        candidates = candidates.loc[candidates["_abs_latency"] <= match_tolerance].sort_values("_abs_latency")
        if candidates.empty:
            continue
        annotation = candidates.iloc[0]
        annotation_id = annotation["_annotation_id"]
        used.add(annotation_id)
        latency = float(event["onset_time"] - float(annotation["onset_time"]))
        matched.loc[event_index, "matched_annotation_id"] = annotation_id
        matched.loc[event_index, "matched_annotation_onset_time"] = float(annotation["onset_time"])
        matched.loc[event_index, "latency"] = latency
        matched.loc[event_index, "is_true_positive"] = True
    return matched


def summarize_stimulus_events(
    events: pd.DataFrame,
    *,
    annotations: pd.DataFrame | None = None,
    group_columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Summarize event-level detection quality."""
    groups = _group_columns(events, group_columns) if not events.empty else list(group_columns or [])
    rows = []
    grouped = events.groupby(groups, sort=True) if groups and not events.empty else [((), events)]
    for keys, group_frame in grouped:
        key_values = keys if isinstance(keys, tuple) else (keys,)
        group_values = dict(zip(groups, key_values, strict=True))
        detected = len(group_frame)
        if "is_true_positive" in group_frame.columns:
            true_positives = int(group_frame["is_true_positive"].fillna(False).astype(bool).sum())
        elif "matched_annotation_id" in group_frame.columns:
            true_positives = int(group_frame["matched_annotation_id"].notna().sum())
        else:
            true_positives = detected
        n_annotations = np.nan
        if annotations is not None:
            annotation_frame = annotations
            for column, value in group_values.items():
                if column in annotation_frame.columns:
                    annotation_frame = annotation_frame.loc[annotation_frame[column].astype(str) == str(value)]
            n_annotations = len(annotation_frame)
        false_positives = detected - true_positives
        false_negatives = int(n_annotations - true_positives) if np.isfinite(n_annotations) else np.nan
        precision = true_positives / detected if detected else np.nan
        recall = true_positives / n_annotations if np.isfinite(n_annotations) and n_annotations else np.nan
        f1 = 2.0 * precision * recall / (precision + recall) if np.isfinite(precision) and np.isfinite(recall) and precision + recall > 0 else np.nan
        latencies = pd.to_numeric(group_frame.get("latency", pd.Series(dtype=float)), errors="coerce").dropna()
        rows.append(
            {
                **group_values,
                "n_detections": detected,
                "n_annotations": n_annotations,
                "true_positive_count": true_positives,
                "false_positive_count": false_positives,
                "false_negative_count": false_negatives,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "latency_mean": float(latencies.mean()) if not latencies.empty else np.nan,
                "latency_median": float(latencies.median()) if not latencies.empty else np.nan,
                "latency_sd": float(latencies.std(ddof=1)) if len(latencies) > 1 else 0.0 if len(latencies) == 1 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def detect_stimulus_events_from_csvs(
    observation_csvs: Sequence[str | Path],
    *,
    annotations_csv: str | Path | None = None,
    thresholds_csv: str | Path | None = None,
    threshold_window: tuple[float, float] = DEFAULT_THRESHOLD_WINDOW,
    threshold_quantile: float = DEFAULT_THRESHOLD_QUANTILE,
    threshold_method: str = "point",
    score_mode: str = "class_probability",
    target_classes: Sequence[str | int] | None = None,
    group_columns: Sequence[str] | None = None,
    stream_columns: Sequence[str] | None = None,
    detection_window: tuple[float, float] | None = None,
    min_consecutive: int = 1,
    min_duration: float | None = None,
    merge_gap: float | None = None,
    refractory: float | None = None,
    match_tolerance: float = 0.1,
    out_events: Path | None = None,
    out_summary: Path | None = None,
    out_thresholds: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    observations = read_stimulus_probability_observations(observation_csvs)
    thresholds = pd.read_csv(thresholds_csv) if thresholds_csv is not None else None
    events = detect_stimulus_events(
        observations,
        thresholds=thresholds,
        threshold_window=threshold_window,
        threshold_quantile=threshold_quantile,
        threshold_method=threshold_method,
        score_mode=score_mode,
        target_classes=target_classes,
        group_columns=group_columns,
        stream_columns=stream_columns,
        detection_window=detection_window,
        min_consecutive=min_consecutive,
        min_duration=min_duration,
        merge_gap=merge_gap,
        refractory=refractory,
    )
    if thresholds is None:
        thresholds = fit_stimulus_detection_thresholds(
            observations,
            threshold_window=threshold_window,
            threshold_quantile=threshold_quantile,
            threshold_method=threshold_method,
            score_mode=score_mode,
            target_classes=target_classes,
            group_columns=group_columns,
            stream_columns=stream_columns,
            min_consecutive=min_consecutive,
            min_duration=min_duration,
        )
    annotations = pd.read_csv(annotations_csv) if annotations_csv is not None else None
    if annotations is not None:
        events = match_stimulus_annotations(events, annotations, stream_columns=stream_columns, match_tolerance=match_tolerance)
    summary = summarize_stimulus_events(events, annotations=annotations, group_columns=group_columns)
    for path, frame in ((out_events, events), (out_summary, summary), (out_thresholds, thresholds)):
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            frame.to_csv(path, index=False)
    return events, summary, thresholds


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect one or more stimulus events in long probability streams.")
    parser.add_argument("observation_csv", nargs="+", help="Observation CSVs or glob patterns with time and prob_class_* columns.")
    parser.add_argument("--annotations-csv", type=Path)
    parser.add_argument("--thresholds-csv", type=Path)
    parser.add_argument("--threshold-window", nargs=2, type=float, default=DEFAULT_THRESHOLD_WINDOW, metavar=("START", "STOP"))
    parser.add_argument("--threshold-quantile", type=float, default=DEFAULT_THRESHOLD_QUANTILE)
    parser.add_argument("--threshold-method", choices=THRESHOLD_METHODS, default="point")
    parser.add_argument("--score-mode", choices=SCORE_MODES, default="class_probability")
    parser.add_argument("--target-class", action="append", dest="target_classes")
    parser.add_argument("--group-column", action="append", dest="group_columns")
    parser.add_argument("--stream-column", action="append", dest="stream_columns")
    parser.add_argument("--detection-window", nargs=2, type=float, metavar=("START", "STOP"))
    parser.add_argument("--min-consecutive", type=int, default=1)
    parser.add_argument("--min-duration", type=float)
    parser.add_argument("--merge-gap", type=float)
    parser.add_argument("--refractory", type=float)
    parser.add_argument("--match-tolerance", type=float, default=0.1)
    parser.add_argument("--out-events", type=Path, required=True)
    parser.add_argument("--out-summary", type=Path, required=True)
    parser.add_argument("--out-thresholds", type=Path)
    args = parser.parse_args()

    events, summary, thresholds = detect_stimulus_events_from_csvs(
        args.observation_csv,
        annotations_csv=args.annotations_csv,
        thresholds_csv=args.thresholds_csv,
        threshold_window=tuple(args.threshold_window),
        threshold_quantile=args.threshold_quantile,
        threshold_method=args.threshold_method,
        score_mode=args.score_mode,
        target_classes=args.target_classes,
        group_columns=args.group_columns,
        stream_columns=args.stream_columns,
        detection_window=tuple(args.detection_window) if args.detection_window is not None else None,
        min_consecutive=args.min_consecutive,
        min_duration=args.min_duration,
        merge_gap=args.merge_gap,
        refractory=args.refractory,
        match_tolerance=args.match_tolerance,
        out_events=args.out_events,
        out_summary=args.out_summary,
        out_thresholds=args.out_thresholds,
    )
    print(f"Wrote stimulus events: {args.out_events}")
    print(f"Wrote stimulus summary: {args.out_summary}")
    if args.out_thresholds is not None:
        print(f"Wrote stimulus thresholds: {args.out_thresholds}")
    print(f"Detected {len(events)} events across {len(thresholds)} fitted class thresholds.")
    if not summary.empty:
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
