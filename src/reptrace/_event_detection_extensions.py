"""Runtime extensions for onset runs and stimulus conflict handling."""

from __future__ import annotations

import numpy as np
import pandas as pd

CONFLICT_RESOLUTION_MODES = (
    "none",
    "winner_take_all",
    "non_max_suppression",
    "highest_peak_per_window",
)


def _install_onset_detection_extensions() -> None:
    """Onset run helpers are implemented directly in reptrace.onset."""


def _ranked(events: pd.DataFrame) -> pd.DataFrame:
    ranked = events.copy()
    ranked["_rank_peak_score"] = pd.to_numeric(ranked["peak_score"], errors="coerce").fillna(-np.inf)
    ranked["_rank_onset_time"] = pd.to_numeric(ranked["onset_time"], errors="coerce").fillna(np.inf)
    ranked["_rank_stimulus_class"] = ranked["stimulus_class"].astype(str)
    return ranked


def _best_event_index(events: pd.DataFrame):
    return _ranked(events).sort_values(
        ["_rank_peak_score", "_rank_onset_time", "_rank_stimulus_class"],
        ascending=[False, True, True],
        kind="mergesort",
    ).index[0]


def _events_overlap(left: pd.Series, right: pd.Series) -> bool:
    return float(left["onset_time"]) <= float(right["offset_time"]) and float(right["onset_time"]) <= float(left["offset_time"])


def _resolve_winner_take_all(events: pd.DataFrame) -> pd.DataFrame:
    ordered = events.sort_values(["onset_time", "offset_time", "stimulus_class"], kind="mergesort")
    selected = []
    cluster = []
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
    ordered = _ranked(events).sort_values(
        ["_rank_peak_score", "_rank_onset_time", "_rank_stimulus_class"],
        ascending=[False, True, True],
        kind="mergesort",
    )
    kept = []
    for index, event in ordered.iterrows():
        if any(_events_overlap(event, events.loc[kept_index]) for kept_index in kept):
            continue
        kept.append(index)
    return events.loc[kept]


def _resolve_highest_peak_per_window(events: pd.DataFrame) -> pd.DataFrame:
    return events.loc[
        [_best_event_index(peak_events) for _, peak_events in events.groupby("peak_time", sort=True, dropna=False)]
    ]


def _resolve_event_conflicts(events: pd.DataFrame, *, stream_columns, conflict_resolution: str) -> pd.DataFrame:
    if conflict_resolution not in CONFLICT_RESOLUTION_MODES:
        raise ValueError(f"conflict_resolution must be one of {CONFLICT_RESOLUTION_MODES}.")
    if events.empty or conflict_resolution == "none":
        return events
    frames = []
    grouped = events.groupby(list(stream_columns), sort=True) if stream_columns else [((), events)]
    for _, stream_events in grouped:
        if conflict_resolution == "winner_take_all":
            frames.append(_resolve_winner_take_all(stream_events))
        elif conflict_resolution == "non_max_suppression":
            frames.append(_resolve_non_max_suppression(stream_events))
        elif conflict_resolution == "highest_peak_per_window":
            frames.append(_resolve_highest_peak_per_window(stream_events))
    return pd.concat(frames, ignore_index=False) if frames else events.iloc[0:0].copy()


def _reindex_events(events: pd.DataFrame, *, stream_columns) -> pd.DataFrame:
    if events.empty:
        return events
    sort_columns = [column for column in [*stream_columns, "onset_time", "stimulus_class"] if column in events.columns]
    events = events.sort_values(sort_columns, kind="mergesort").reset_index(drop=True)
    grouped = events.groupby(list(stream_columns), sort=False) if stream_columns else [((), events)]
    for _, stream_events in grouped:
        events.loc[stream_events.index, "event_index"] = range(len(stream_events))
    events["event_index"] = events["event_index"].astype(int)
    return events


def _install_stimulus_detection_extensions() -> None:
    from reptrace import stimulus_detection as stimulus

    original_detect = stimulus.detect_stimulus_events
    stimulus.CONFLICT_RESOLUTION_MODES = CONFLICT_RESOLUTION_MODES
    if "conflict_resolution" not in stimulus.EVENT_COLUMNS:
        insert_at = stimulus.EVENT_COLUMNS.index("predicted_label_at_peak")
        stimulus.EVENT_COLUMNS.insert(insert_at, "conflict_resolution")

    def detect_stimulus_events(observations, *args, conflict_resolution="none", **kwargs):
        """Detect stimulus events and optionally resolve competing classes."""
        if conflict_resolution not in CONFLICT_RESOLUTION_MODES:
            raise ValueError(f"conflict_resolution must be one of {CONFLICT_RESOLUTION_MODES}.")
        stream_columns = kwargs.get("stream_columns")
        streams = stimulus._stream_columns(observations, stream_columns)  # noqa: SLF001
        events = original_detect(observations, *args, **kwargs)
        events["conflict_resolution"] = conflict_resolution
        events = _resolve_event_conflicts(events, stream_columns=streams, conflict_resolution=conflict_resolution)
        return _reindex_events(events, stream_columns=streams)

    stimulus._resolve_event_conflicts = _resolve_event_conflicts  # noqa: SLF001
    stimulus.detect_stimulus_events = detect_stimulus_events


def install() -> None:
    _install_onset_detection_extensions()
    _install_stimulus_detection_extensions()
