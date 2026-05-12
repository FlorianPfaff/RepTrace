from __future__ import annotations

import argparse
import importlib.util
from collections.abc import Sequence
from pathlib import Path
import sys

import numpy as np
import pandas as pd

_LEGACY_PATH = Path(__file__).with_name("stimulus_detection.py")
_SPEC = importlib.util.spec_from_file_location("reptrace._stimulus_detection_legacy", _LEGACY_PATH)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover
    raise ImportError(f"Cannot load stimulus detection implementation from {_LEGACY_PATH}")
_legacy = importlib.util.module_from_spec(_SPEC)
sys.modules.setdefault("reptrace._stimulus_detection_legacy", _legacy)
_SPEC.loader.exec_module(_legacy)

DEFAULT_THRESHOLD_WINDOW = _legacy.DEFAULT_THRESHOLD_WINDOW
DEFAULT_THRESHOLD_QUANTILE = _legacy.DEFAULT_THRESHOLD_QUANTILE
DEFAULT_GROUP_COLUMNS = _legacy.DEFAULT_GROUP_COLUMNS
DEFAULT_STREAM_FALLBACKS = _legacy.DEFAULT_STREAM_FALLBACKS
THRESHOLD_METHODS = _legacy.THRESHOLD_METHODS
SCORE_MODES = _legacy.SCORE_MODES
CONFLICT_RESOLUTION_MODES = (
    "none",
    "winner_take_all",
    "non_max_suppression",
    "highest_peak_per_window",
)

EVENT_COLUMNS = list(_legacy.EVENT_COLUMNS)
if "conflict_resolution" not in EVENT_COLUMNS:
    insert_at = EVENT_COLUMNS.index("predicted_label_at_peak") if "predicted_label_at_peak" in EVENT_COLUMNS else len(EVENT_COLUMNS)
    EVENT_COLUMNS.insert(insert_at, "conflict_resolution")

fit_stimulus_detection_thresholds = _legacy.fit_stimulus_detection_thresholds
read_stimulus_probability_observations = _legacy.read_stimulus_probability_observations
_present_columns = _legacy._present_columns
_group_columns = _legacy._group_columns
_stream_columns = _legacy._stream_columns
_run_duration = _legacy._run_duration
_annotation_id = _legacy._annotation_id


def _unique_columns(columns: Sequence[str]) -> list[str]:
    unique: list[str] = []
    for column in columns:
        if column not in unique:
            unique.append(column)
    return unique


def _event_row(*args, conflict_resolution: str = "none", **kwargs) -> dict:
    row = _legacy._event_row(*args, **kwargs)
    row["conflict_resolution"] = conflict_resolution
    return row


def _ranked_events(events: pd.DataFrame) -> pd.DataFrame:
    ranked = events.copy()
    ranked["_rank_peak_score"] = pd.to_numeric(ranked["peak_score"], errors="coerce").fillna(-np.inf)
    ranked["_rank_onset_time"] = pd.to_numeric(ranked["onset_time"], errors="coerce").fillna(np.inf)
    ranked["_rank_stimulus_class"] = ranked["stimulus_class"].astype(str)
    return ranked


def _best_event_index(events: pd.DataFrame) -> object:
    return _ranked_events(events).sort_values(
        ["_rank_peak_score", "_rank_onset_time", "_rank_stimulus_class"],
        ascending=[False, True, True],
        kind="mergesort",
    ).index[0]


def _events_overlap(left: pd.Series, right: pd.Series) -> bool:
    return float(left["onset_time"]) <= float(right["offset_time"]) and float(right["onset_time"]) <= float(left["offset_time"])


def _resolve_winner_take_all(events: pd.DataFrame) -> pd.DataFrame:
    ordered = events.sort_values(["onset_time", "offset_time", "stimulus_class"], kind="mergesort")
    selected: list[object] = []
    cluster: list[object] = []
    cluster_stop = -np.inf
    for index, event in ordered.iterrows():
        onset = float(event["onset_time"])
        offset = float(event["offset_time"])
        if cluster and onset > cluster_stop:
            selected.append(_best_event_index(ordered.loc[cluster]))
            cluster = [index]
            cluster_stop = offset
        else:
            cluster.append(index)
            cluster_stop = max(cluster_stop, offset)
    if cluster:
        selected.append(_best_event_index(ordered.loc[cluster]))
    return events.loc[selected]


def _resolve_non_max_suppression(events: pd.DataFrame) -> pd.DataFrame:
    ordered = _ranked_events(events).sort_values(
        ["_rank_peak_score", "_rank_onset_time", "_rank_stimulus_class"],
        ascending=[False, True, True],
        kind="mergesort",
    )
    kept: list[object] = []
    for index, event in ordered.iterrows():
        if any(_events_overlap(event, events.loc[kept_index]) for kept_index in kept):
            continue
        kept.append(index)
    return events.loc[kept]


def _resolve_highest_peak_per_window(events: pd.DataFrame) -> pd.DataFrame:
    selected = [_best_event_index(peak_events) for _, peak_events in events.groupby("peak_time", sort=True, dropna=False)]
    return events.loc[selected]


def _resolve_event_conflicts(events: pd.DataFrame, *, partition_columns: Sequence[str], conflict_resolution: str) -> pd.DataFrame:
    if conflict_resolution not in CONFLICT_RESOLUTION_MODES:
        raise ValueError(f"conflict_resolution must be one of {CONFLICT_RESOLUTION_MODES}.")
    if events.empty or conflict_resolution == "none":
        return events
    frames = []
    present_partition_columns = _present_columns(events, partition_columns)
    grouped = events.groupby(present_partition_columns, sort=True) if present_partition_columns else [((), events)]
    for _, partition_events in grouped:
        if conflict_resolution == "winner_take_all":
            frames.append(_resolve_winner_take_all(partition_events))
        elif conflict_resolution == "non_max_suppression":
            frames.append(_resolve_non_max_suppression(partition_events))
        elif conflict_resolution == "highest_peak_per_window":
            frames.append(_resolve_highest_peak_per_window(partition_events))
    return pd.concat(frames, ignore_index=False) if frames else events.iloc[0:0].copy()


def _reindex_events(events: pd.DataFrame, *, partition_columns: Sequence[str]) -> pd.DataFrame:
    if events.empty:
        return events
    sort_columns = [column for column in [*partition_columns, "onset_time", "stimulus_class"] if column in events.columns]
    events = events.sort_values(sort_columns, kind="mergesort").reset_index(drop=True)
    present_partition_columns = _present_columns(events, partition_columns)
    grouped = events.groupby(present_partition_columns, sort=False) if present_partition_columns else [((), events)]
    for _, partition_events in grouped:
        events.loc[partition_events.index, "event_index"] = range(len(partition_events))
    events["event_index"] = events["event_index"].astype(int)
    return events


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
    conflict_resolution: str = "none",
) -> pd.DataFrame:
    if conflict_resolution not in CONFLICT_RESOLUTION_MODES:
        raise ValueError(f"conflict_resolution must be one of {CONFLICT_RESOLUTION_MODES}.")
    events = _legacy.detect_stimulus_events(
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
    events["conflict_resolution"] = conflict_resolution
    groups = _group_columns(observations, group_columns)
    streams = _stream_columns(observations, stream_columns)
    partition_columns = _unique_columns([*groups, *streams])
    events = _resolve_event_conflicts(events, partition_columns=partition_columns, conflict_resolution=conflict_resolution)
    return _reindex_events(events, partition_columns=partition_columns)


def _annotation_value(annotation: pd.Series, *columns: str, default: object = np.nan) -> object:
    for column in columns:
        if column in annotation and pd.notna(annotation[column]):
            return annotation[column]
    return default


def _add_annotation_candidate_columns(events: pd.DataFrame) -> pd.DataFrame:
    events = events.copy()
    events["matched_annotation_id"] = pd.Series(pd.NA, index=events.index, dtype="object")
    events["matched_annotation_onset_time"] = np.nan
    events["matched_annotation_class"] = pd.Series("", index=events.index, dtype="object")
    events["matched_annotation_label"] = pd.Series(pd.NA, index=events.index, dtype="object")
    events["candidate_annotation_id"] = pd.Series(pd.NA, index=events.index, dtype="object")
    events["candidate_annotation_onset_time"] = np.nan
    events["candidate_annotation_class"] = pd.Series("", index=events.index, dtype="object")
    events["candidate_annotation_label"] = pd.Series(pd.NA, index=events.index, dtype="object")
    events["candidate_latency"] = np.nan
    events["latency"] = np.nan
    events["is_true_positive"] = False
    events["is_duplicate_detection"] = False
    return events


def match_stimulus_annotations(
    events: pd.DataFrame,
    annotations: pd.DataFrame,
    *,
    stream_columns: Sequence[str] | None = None,
    match_tolerance: float = 0.1,
    require_class_match: bool = True,
) -> pd.DataFrame:
    """Greedily match detected events to annotated stimulus onsets.

    Besides the one-to-one true-positive assignment, the matcher records the
    nearest within-tolerance annotation candidate for each event. If that
    candidate has already been claimed, the event is marked as a duplicate
    detection, which lets summaries distinguish duplicates from ordinary false
    alarms.
    """
    if match_tolerance < 0:
        raise ValueError("match_tolerance must be non-negative.")
    matched = _add_annotation_candidate_columns(events)
    if events.empty:
        return matched
    streams = _stream_columns(events, stream_columns)
    if "onset_time" not in annotations.columns:
        raise ValueError("Annotation rows must contain onset_time.")
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
        candidates["_latency"] = float(event["onset_time"]) - pd.to_numeric(candidates["onset_time"], errors="coerce")
        candidates["_abs_latency"] = candidates["_latency"].abs()
        candidates = candidates.loc[candidates["_abs_latency"] <= match_tolerance].sort_values("_abs_latency")
        if candidates.empty:
            continue

        nearest = candidates.iloc[0]
        nearest_id = nearest["_annotation_id"]
        matched.loc[event_index, "candidate_annotation_id"] = nearest_id
        matched.loc[event_index, "candidate_annotation_onset_time"] = float(nearest["onset_time"])
        matched.loc[event_index, "candidate_annotation_class"] = _annotation_value(nearest, "stimulus_class", default="")
        matched.loc[event_index, "candidate_annotation_label"] = _annotation_value(nearest, "stimulus_label", default=np.nan)
        matched.loc[event_index, "candidate_latency"] = float(nearest["_latency"])

        available = candidates.loc[~candidates["_annotation_id"].isin(used)]
        if available.empty:
            matched.loc[event_index, "is_duplicate_detection"] = True
            continue

        annotation = available.iloc[0]
        annotation_id = annotation["_annotation_id"]
        used.add(annotation_id)
        latency = float(annotation["_latency"])
        matched.loc[event_index, "matched_annotation_id"] = annotation_id
        matched.loc[event_index, "matched_annotation_onset_time"] = float(annotation["onset_time"])
        matched.loc[event_index, "matched_annotation_class"] = _annotation_value(annotation, "stimulus_class", default="")
        matched.loc[event_index, "matched_annotation_label"] = _annotation_value(annotation, "stimulus_label", default=np.nan)
        matched.loc[event_index, "latency"] = latency
        matched.loc[event_index, "is_true_positive"] = True
    return matched


def _duration_minutes(
    observations: pd.DataFrame | None,
    *,
    group_values: dict[str, object],
    stream_columns: Sequence[str] | None,
) -> float:
    if observations is None or observations.empty or "time" not in observations.columns:
        return np.nan
    frame = observations.copy()
    for column, value in group_values.items():
        if column in frame.columns:
            frame = frame.loc[frame[column].astype(str) == str(value)]
    if frame.empty:
        return np.nan

    streams = _present_columns(frame, stream_columns or [])
    grouped = frame.groupby(streams, sort=False) if streams else [((), frame)]
    seconds = 0.0
    for _, stream_frame in grouped:
        times = pd.to_numeric(stream_frame["time"], errors="coerce").dropna()
        if len(times) >= 2:
            seconds += float(times.max() - times.min())
    return seconds / 60.0 if seconds > 0 else np.nan


def _matched_events(group_frame: pd.DataFrame) -> pd.DataFrame:
    if "is_true_positive" in group_frame.columns:
        return group_frame.loc[group_frame["is_true_positive"].fillna(False).astype(bool)]
    if "matched_annotation_id" in group_frame.columns:
        return group_frame.loc[group_frame["matched_annotation_id"].notna()]
    return group_frame.iloc[0:0]


def _class_accuracy_for_matched_events(group_frame: pd.DataFrame) -> float:
    matched = _matched_events(group_frame)
    if matched.empty or "matched_annotation_class" not in matched.columns:
        return np.nan
    annotated_classes = matched["matched_annotation_class"].astype(str)
    known = annotated_classes.ne("") & annotated_classes.ne("nan")
    if not known.any():
        return np.nan
    return float(matched.loc[known, "stimulus_class"].astype(str).eq(annotated_classes.loc[known]).mean())


def _latency_sd(latencies: pd.Series) -> float:
    if len(latencies) > 1:
        return float(latencies.std(ddof=1))
    if len(latencies) == 1:
        return 0.0
    return np.nan


def summarize_stimulus_events(
    events: pd.DataFrame,
    *,
    annotations: pd.DataFrame | None = None,
    observations: pd.DataFrame | None = None,
    group_columns: Sequence[str] | None = None,
    stream_columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Summarize event-level detection quality.

    The summary exposes both the original ``*_count``/``latency_*`` columns and
    checklist-oriented aliases such as ``true_positives`` and
    ``onset_latency_mean`` for downstream compatibility.
    """
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
        duration_minutes = _duration_minutes(observations, group_values=group_values, stream_columns=stream_columns)
        false_alarms_per_minute = false_positives / duration_minutes if np.isfinite(duration_minutes) and duration_minutes > 0 else np.nan
        duplicate_detections = (
            int(group_frame["is_duplicate_detection"].fillna(False).astype(bool).sum())
            if "is_duplicate_detection" in group_frame.columns
            else 0
        )
        latencies = pd.to_numeric(group_frame.get("latency", pd.Series(dtype=float)), errors="coerce").dropna()
        latency_mean = float(latencies.mean()) if not latencies.empty else np.nan
        latency_median = float(latencies.median()) if not latencies.empty else np.nan
        latency_sd = _latency_sd(latencies)
        rows.append(
            {
                **group_values,
                "n_detections": detected,
                "n_annotations": n_annotations,
                "true_positives": true_positives,
                "false_positives": false_positives,
                "false_negatives": false_negatives,
                "true_positive_count": true_positives,
                "false_positive_count": false_positives,
                "false_negative_count": false_negatives,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "false_alarms_per_minute": false_alarms_per_minute,
                "duplicate_detections": duplicate_detections,
                "class_accuracy_for_matched_events": _class_accuracy_for_matched_events(group_frame),
                "onset_latency_mean": latency_mean,
                "onset_latency_median": latency_median,
                "onset_latency_sd": latency_sd,
                "latency_mean": latency_mean,
                "latency_median": latency_median,
                "latency_sd": latency_sd,
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
    conflict_resolution: str = "none",
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
        conflict_resolution=conflict_resolution,
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
    summary = summarize_stimulus_events(
        events,
        annotations=annotations,
        observations=observations,
        group_columns=group_columns,
        stream_columns=stream_columns,
    )
    for path, frame in ((out_events, events), (out_summary, summary), (out_thresholds, thresholds)):
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            frame.to_csv(path, index=False)
    return events, summary, thresholds


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect one or more stimulus events in long probability streams.")
    parser.add_argument("observation_csv", nargs="+", help="Observation CSVs or glob patterns with time and prob_class_* columns.")
    parser.add_argument("--annotations", "--annotations-csv", dest="annotations_csv", type=Path)
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
    parser.add_argument("--conflict-resolution", choices=CONFLICT_RESOLUTION_MODES, default="none")
    parser.add_argument("--match-tolerance", type=float, default=0.1)
    parser.add_argument("--out-events", type=Path, required=True)
    parser.add_argument("--out-summary", type=Path, required=True)
    parser.add_argument("--out-thresholds", type=Path)
    args = parser.parse_args()

    events, summary, _thresholds = detect_stimulus_events_from_csvs(
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
        conflict_resolution=args.conflict_resolution,
        match_tolerance=args.match_tolerance,
        out_events=args.out_events,
        out_summary=args.out_summary,
        out_thresholds=args.out_thresholds,
    )
    print(f"Wrote stimulus events: {args.out_events}")
    print(f"Wrote stimulus event summary: {args.out_summary}")
    if args.out_thresholds is not None:
        print(f"Wrote stimulus thresholds: {args.out_thresholds}")
    print(summary.to_string(index=False))
    if not events.empty:
        print(events.head().to_string(index=False))


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CONFLICT_RESOLUTION_MODES",
    "DEFAULT_GROUP_COLUMNS",
    "DEFAULT_STREAM_FALLBACKS",
    "DEFAULT_THRESHOLD_QUANTILE",
    "DEFAULT_THRESHOLD_WINDOW",
    "EVENT_COLUMNS",
    "SCORE_MODES",
    "THRESHOLD_METHODS",
    "detect_stimulus_events",
    "detect_stimulus_events_from_csvs",
    "fit_stimulus_detection_thresholds",
    "main",
    "match_stimulus_annotations",
    "read_stimulus_probability_observations",
    "summarize_stimulus_events",
]
