from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from reptrace.stimulus_detection import (
    DEFAULT_GROUP_COLUMNS,
    DEFAULT_STREAM_FALLBACKS,
    _event_row,
    _run_duration,
)


@dataclass(frozen=True)
class StimulusDetectionConfig:
    """Configuration for online stimulus-event detection."""

    group_columns: Sequence[str] | None = DEFAULT_GROUP_COLUMNS
    stream_columns: Sequence[str] | None = None
    detection_window: tuple[float, float] | None = None
    min_consecutive: int = 1
    min_duration: float | None = None
    merge_gap: float | None = None
    refractory: float | None = None


@dataclass
class _RunState:
    threshold_index: int
    group_values: dict[str, object]
    stream_values: dict[str, object]
    stream_counter_key: tuple[object, ...]
    current_run_start_time: float | None = None
    current_run_rows: list[dict[str, object]] = field(default_factory=list)
    last_event_time: float | None = None
    current_predicted_class: object = None
    above_threshold_count: int = 0
    pending_gap: bool = False
    last_seen_time: float | None = None

    def reset_run(self) -> None:
        self.current_run_start_time = None
        self.current_run_rows = []
        self.current_predicted_class = None
        self.above_threshold_count = 0
        self.pending_gap = False


def _validate_config(config: StimulusDetectionConfig) -> None:
    if config.min_consecutive < 1:
        raise ValueError("min_consecutive must be at least 1.")
    if config.min_duration is not None and config.min_duration < 0:
        raise ValueError("min_duration must be non-negative when provided.")
    if config.merge_gap is not None and config.merge_gap < 0:
        raise ValueError("merge_gap must be non-negative when provided.")
    if config.refractory is not None and config.refractory < 0:
        raise ValueError("refractory must be non-negative when provided.")


def _stream_columns_for_observation(
    observation: Mapping[str, object],
    stream_columns: Sequence[str] | None,
) -> list[str]:
    if stream_columns is not None:
        missing = [column for column in stream_columns if column not in observation]
        if missing:
            raise ValueError(f"Observation row is missing stream columns: {missing}")
        return list(stream_columns)
    for candidates in DEFAULT_STREAM_FALLBACKS:
        if all(column in observation for column in candidates):
            return list(candidates)
    raise ValueError("Observation row must contain stream_id, sequence_id, or sample_index.")


def _threshold_group_columns(thresholds: pd.DataFrame, group_columns: Sequence[str] | None) -> list[str]:
    candidates = DEFAULT_GROUP_COLUMNS if group_columns is None else group_columns
    return [column for column in candidates if column in thresholds.columns]


def _threshold_matches_observation(
    threshold_row: pd.Series,
    observation: Mapping[str, object],
    group_columns: Sequence[str],
) -> bool:
    missing = [column for column in group_columns if column not in observation]
    if missing:
        raise ValueError(f"Observation row is missing threshold group columns: {missing}")
    return all(str(observation[column]) == str(threshold_row[column]) for column in group_columns)


def _observation_time(observation: Mapping[str, object]) -> float:
    if "time" not in observation:
        raise ValueError("Observation row must contain a time column.")
    return float(observation["time"])


def _prediction_value(row: Mapping[str, object]) -> object:
    if "predicted_class" in row and pd.notna(row["predicted_class"]):
        return row["predicted_class"]
    if "predicted_label" in row and pd.notna(row["predicted_label"]):
        return row["predicted_label"]
    return None


def _probability_columns_from_observation(observation: Mapping[str, object]) -> list[str]:
    columns = [str(column) for column in observation if str(column).startswith("prob_class_")]
    if not columns:
        raise ValueError("Observation rows must contain probability columns named 'prob_class_*'.")

    def sort_key(column: str) -> tuple[int, str]:
        suffix = column.removeprefix("prob_class_")
        return (int(suffix), suffix) if suffix.isdigit() else (10_000, suffix)

    return sorted(columns, key=sort_key)


def _as_float(value: object) -> float:
    return float(pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0])


def _score_observation(observation: Mapping[str, object], threshold_row: pd.Series) -> float:
    score_mode = str(threshold_row["score_mode"])
    score_column = str(threshold_row["score_column"])
    if score_mode == "class_probability":
        if score_column not in observation:
            raise ValueError(f"Score column '{score_column}' is missing.")
        return _as_float(observation[score_column])

    if score_mode != "predicted_class_confidence":
        raise ValueError("score_mode must be 'class_probability' or 'predicted_class_confidence'.")
    if "confidence" not in observation:
        raise ValueError("predicted_class_confidence scoring requires a confidence column.")
    confidence = 0.0 if pd.isna(observation["confidence"]) else _as_float(observation["confidence"])
    stimulus_label = str(threshold_row["stimulus_label"])
    stimulus_class = str(threshold_row["stimulus_class"])
    if "predicted_label" in observation:
        return confidence if str(observation["predicted_label"]) == stimulus_label else 0.0
    if "predicted_class" in observation:
        return confidence if str(observation["predicted_class"]) == stimulus_class else 0.0

    probability_columns = _probability_columns_from_observation(observation)
    probabilities = np.array([float(observation[column]) for column in probability_columns], dtype=float)
    predicted_label = int(np.nanargmax(probabilities))
    return confidence if str(predicted_label) == stimulus_label else 0.0


def _valid_run(run: pd.DataFrame, *, config: StimulusDetectionConfig) -> bool:
    if len(run) < config.min_consecutive:
        return False
    return not (config.min_duration is not None and _run_duration(run) < config.min_duration)


class StreamingStimulusDetector:
    """Stateful online detector for stream-level stimulus events.

    The detector consumes one probability-observation row at a time and returns
    events once an above-threshold run is causally finalized. With no
    ``merge_gap``, finalization occurs on the first below-threshold row after a
    valid run. With ``merge_gap``, finalization is delayed until the gap is too
    large to merge. Call :meth:`flush` after the last observation to emit any
    still-open valid runs.
    """

    def __init__(self, config: StimulusDetectionConfig, thresholds: pd.DataFrame):
        _validate_config(config)
        self.config = config
        self.thresholds = thresholds.copy().reset_index(drop=True)
        self._validate_thresholds()
        self._group_columns = _threshold_group_columns(self.thresholds, config.group_columns)
        self._states: dict[tuple[object, ...], _RunState] = {}
        self._event_counters: dict[tuple[object, ...], int] = {}

    def update(self, observation: Mapping[str, object]) -> list[dict[str, object]]:
        """Process one observation and return newly finalized stimulus events."""
        time = _observation_time(observation)
        stream_columns = _stream_columns_for_observation(observation, self.config.stream_columns)
        stream_values = {column: observation[column] for column in stream_columns}
        events: list[dict[str, object]] = []

        for threshold_index, threshold_row in self.thresholds.iterrows():
            threshold = float(threshold_row["score_threshold"])
            if not np.isfinite(threshold):
                continue
            if not _threshold_matches_observation(threshold_row, observation, self._group_columns):
                continue
            state = self._state_for(threshold_index, threshold_row, stream_values)
            self._validate_time_order(state, time)
            window_status = self._window_status(time)
            if window_status == "before":
                continue
            score = _score_observation(observation, threshold_row)
            above_threshold = window_status == "inside" and np.isfinite(score) and score >= threshold
            events.extend(
                self._advance_state(
                    state,
                    threshold_row,
                    self._scored_row(observation, score),
                    above_threshold,
                    emitted_at=time,
                )
            )
        return events

    def flush(self) -> list[dict[str, object]]:
        """Emit any valid open runs at the end of a finite stream."""
        events: list[dict[str, object]] = []
        for state in list(self._states.values()):
            if not state.current_run_rows:
                continue
            emitted_at = state.last_seen_time or float(state.current_run_rows[-1]["time"])
            threshold_row = self.thresholds.iloc[state.threshold_index]
            event = self._finalize_run(state, threshold_row, emitted_at=float(emitted_at))
            if event is not None:
                events.append(event)
        return sorted(events, key=lambda event: (event["onset_time"], event["stimulus_class"]))

    def reset(self) -> None:
        """Clear all active runs, refractory state, and event counters."""
        self._states.clear()
        self._event_counters.clear()

    def _validate_thresholds(self) -> None:
        required = {"stimulus_label", "stimulus_class", "score_column", "score_mode", "score_threshold"}
        missing = sorted(required.difference(self.thresholds.columns))
        if missing:
            raise ValueError(f"Threshold rows are missing required columns: {missing}")

    def _validate_time_order(self, state: _RunState, time: float) -> None:
        if state.last_seen_time is not None and time < state.last_seen_time:
            raise ValueError(
                "Streaming observations must be ordered by non-decreasing time "
                "within each stream and class."
            )
        state.last_seen_time = time

    def _window_status(self, time: float) -> str:
        if self.config.detection_window is None:
            return "inside"
        start, stop = self.config.detection_window
        if time < start:
            return "before"
        if time > stop:
            return "after"
        return "inside"

    def _state_for(
        self,
        threshold_index: int,
        threshold_row: pd.Series,
        stream_values: dict[str, object],
    ) -> _RunState:
        stream_counter_key = tuple(stream_values.values())
        state_key = (threshold_index, *stream_counter_key)
        if state_key not in self._states:
            group_values = {column: threshold_row[column] for column in self._group_columns}
            self._states[state_key] = _RunState(
                threshold_index=threshold_index,
                group_values=group_values,
                stream_values=dict(stream_values),
                stream_counter_key=stream_counter_key,
            )
        return self._states[state_key]

    @staticmethod
    def _scored_row(observation: Mapping[str, object], score: float) -> dict[str, object]:
        row = dict(observation)
        row["_stimulus_score"] = score
        return row

    def _advance_state(
        self,
        state: _RunState,
        threshold_row: pd.Series,
        row: dict[str, object],
        above_threshold: bool,
        *,
        emitted_at: float,
    ) -> list[dict[str, object]]:
        if above_threshold:
            return self._handle_above_threshold(state, threshold_row, row, emitted_at=emitted_at)
        return self._handle_below_threshold(state, threshold_row, emitted_at=emitted_at)

    def _handle_above_threshold(
        self,
        state: _RunState,
        threshold_row: pd.Series,
        row: dict[str, object],
        *,
        emitted_at: float,
    ) -> list[dict[str, object]]:
        events: list[dict[str, object]] = []
        if state.current_run_rows and state.pending_gap:
            gap = float(row["time"]) - float(state.current_run_rows[-1]["time"])
            if self.config.merge_gap is not None and gap > self.config.merge_gap:
                event = self._finalize_run(state, threshold_row, emitted_at=emitted_at)
                if event is not None:
                    events.append(event)
                self._start_run(state, row)
                return events
        if not state.current_run_rows:
            self._start_run(state, row)
        else:
            self._append_to_run(state, row)
        return events

    def _handle_below_threshold(
        self,
        state: _RunState,
        threshold_row: pd.Series,
        *,
        emitted_at: float,
    ) -> list[dict[str, object]]:
        if not state.current_run_rows:
            return []
        if self.config.merge_gap is None:
            event = self._finalize_run(state, threshold_row, emitted_at=emitted_at)
            return [] if event is None else [event]
        gap = emitted_at - float(state.current_run_rows[-1]["time"])
        if gap > self.config.merge_gap:
            event = self._finalize_run(state, threshold_row, emitted_at=emitted_at)
            return [] if event is None else [event]
        state.pending_gap = True
        return []

    def _start_run(self, state: _RunState, row: dict[str, object]) -> None:
        state.current_run_rows = [row]
        state.current_run_start_time = float(row["time"])
        state.current_predicted_class = _prediction_value(row)
        state.above_threshold_count = 1
        state.pending_gap = False

    def _append_to_run(self, state: _RunState, row: dict[str, object]) -> None:
        state.current_run_rows.append(row)
        state.current_predicted_class = _prediction_value(row)
        state.above_threshold_count = len(state.current_run_rows)
        state.pending_gap = False

    def _finalize_run(
        self,
        state: _RunState,
        threshold_row: pd.Series,
        *,
        emitted_at: float,
    ) -> dict[str, object] | None:
        run = pd.DataFrame(state.current_run_rows).sort_values("time").reset_index(drop=True)
        state.reset_run()
        if not _valid_run(run, config=self.config):
            return None
        onset = float(run.iloc[0]["time"])
        if self._is_refractory_duplicate(state, onset):
            return None
        event_index = self._event_counters.get(state.stream_counter_key, 0)
        event = _event_row(
            group_values=state.group_values,
            stream_values=state.stream_values,
            event_index=event_index,
            run=run,
            threshold_row=threshold_row,
            min_consecutive=self.config.min_consecutive,
            min_duration=self.config.min_duration,
            merge_gap=self.config.merge_gap,
            refractory=self.config.refractory,
        )
        event["detection_confirmed_time"] = emitted_at
        self._event_counters[state.stream_counter_key] = event_index + 1
        state.last_event_time = onset
        return event

    def _is_refractory_duplicate(self, state: _RunState, onset: float) -> bool:
        if self.config.refractory is None or state.last_event_time is None:
            return False
        return onset - state.last_event_time < self.config.refractory
