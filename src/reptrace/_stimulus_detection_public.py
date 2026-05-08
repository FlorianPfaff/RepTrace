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
match_stimulus_annotations = _legacy.match_stimulus_annotations
summarize_stimulus_events = _legacy.summarize_stimulus_events
_present_columns = _legacy._present_columns
_group_columns = _legacy._group_columns
_stream_columns = _legacy._stream_columns
_run_duration = _legacy._run_duration


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
