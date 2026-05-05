from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd

from reptrace.temporal_model import probability_columns, read_probability_observations

DEFAULT_THRESHOLD_WINDOW = (-0.35, -0.05)
DEFAULT_THRESHOLD_QUANTILE = 0.95
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


def _class_name_for_row(row: pd.Series, class_index: int) -> str:
    class_column = f"class_{class_index}"
    if class_column in row and pd.notna(row[class_column]):
        return str(row[class_column])
    return str(class_index)


def _ensure_prediction_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    if "predicted_label" in frame.columns and "predicted_class" in frame.columns:
        return frame
    prob_columns = probability_columns(frame)
    probabilities = frame[prob_columns].to_numpy(dtype=float)
    predicted_labels = probabilities.argmax(axis=1)
    if "predicted_label" not in frame.columns:
        frame["predicted_label"] = predicted_labels
    if "predicted_class" not in frame.columns:
        frame["predicted_class"] = [
            _class_name_for_row(row, int(label))
            for label, (_, row) in zip(predicted_labels, frame.iterrows(), strict=True)
        ]
    return frame


def _threshold_for_group(
    frame: pd.DataFrame,
    *,
    threshold_window: tuple[float, float],
    threshold_quantile: float,
    score_column: str,
) -> float:
    start, stop = threshold_window
    baseline = frame.loc[(frame["time"] >= start) & (frame["time"] <= stop)]
    scores = _score_values(baseline, score_column).dropna()
    if scores.empty:
        return np.nan
    return float(scores.quantile(threshold_quantile))


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
    rows = list(candidates.iterrows())

    for position, (_, row) in enumerate(rows):
        above_threshold = bool(row["_onset_score"] >= threshold) if pd.notna(row["_onset_score"]) else False
        prediction = _prediction_value(row)
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


# pylint: disable-next=too-many-arguments
def _event_row(
    group_values: dict,
    sequence_frame: pd.DataFrame,
    detection_run: pd.DataFrame | None,
    *,
    threshold: float,
    threshold_window: tuple[float, float],
    threshold_quantile: float,
    score_column: str,
    detection_start: float | None,
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
        "threshold_quantile": threshold_quantile,
        "threshold_window_start": threshold_window[0],
        "threshold_window_stop": threshold_window[1],
        "detection_start": detection_start if detection_start is not None else np.nan,
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


# pylint: disable-next=too-many-locals
def detect_onsets(
    observations: pd.DataFrame,
    *,
    threshold_window: tuple[float, float] = DEFAULT_THRESHOLD_WINDOW,
    threshold_quantile: float = DEFAULT_THRESHOLD_QUANTILE,
    score_column: str = "confidence",
    detection_start: float | None = None,
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
    if min_consecutive < 1:
        raise ValueError("min_consecutive must be at least 1.")
    if min_duration is not None and min_duration < 0:
        raise ValueError("min_duration must be non-negative when provided.")
    if "time" not in observations.columns:
        raise ValueError("Observation rows must contain a time column.")

    observations = _ensure_prediction_columns(observations)
    observations = observations.copy()
    observations["_onset_score"] = _score_values(observations, score_column)
    group_columns = _group_columns(observations)
    sequence_columns = _sequence_columns(observations)
    event_rows = []

    grouped = observations.groupby(group_columns, sort=True) if group_columns else [((), observations)]
    for keys, group_frame in grouped:
        key_values = keys if isinstance(keys, tuple) else (keys,)
        group_values = dict(zip(group_columns, key_values, strict=True))
        threshold = _threshold_for_group(
            group_frame,
            threshold_window=threshold_window,
            threshold_quantile=threshold_quantile,
            score_column=score_column,
        )
        sorted_group = group_frame.sort_values([*sequence_columns, "time"])
        for _, sequence_frame in sorted_group.groupby(sequence_columns, sort=True):
            candidates = sequence_frame
            if detection_start is not None:
                candidates = candidates.loc[candidates["time"] >= detection_start]
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
                    threshold_window=threshold_window,
                    threshold_quantile=threshold_quantile,
                    score_column=score_column,
                    detection_start=detection_start,
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


# pylint: disable-next=too-many-arguments
def detect_onsets_from_csvs(
    observation_csvs: list[Path],
    *,
    threshold_window: tuple[float, float] = DEFAULT_THRESHOLD_WINDOW,
    threshold_quantile: float = DEFAULT_THRESHOLD_QUANTILE,
    score_column: str = "confidence",
    detection_start: float | None = None,
    min_consecutive: int = 1,
    min_duration: float | None = None,
    require_stable_prediction: bool = False,
    out_events: Path | None = None,
    out_summary: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read probability observations, detect onsets, and optionally write CSV outputs."""

    observations = read_probability_observations(observation_csvs)
    events = detect_onsets(
        observations,
        threshold_window=threshold_window,
        threshold_quantile=threshold_quantile,
        score_column=score_column,
        detection_start=detection_start,
        min_consecutive=min_consecutive,
        min_duration=min_duration,
        require_stable_prediction=require_stable_prediction,
    )
    summary = summarize_onset_events(events)
    if out_events is not None:
        out_events.parent.mkdir(parents=True, exist_ok=True)
        events.to_csv(out_events, index=False)
    if out_summary is not None:
        out_summary.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(out_summary, index=False)
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
    parser.add_argument("--score-column", default="confidence")
    parser.add_argument("--detection-start", type=float)
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
        detection_start=args.detection_start,
        min_consecutive=args.min_consecutive,
        min_duration=args.min_duration,
        require_stable_prediction=args.require_stable_prediction,
        out_events=args.out_events,
        out_summary=args.out_summary,
    )
    print(f"Wrote onset events: {args.out_events}")
    print(f"Wrote onset summary: {args.out_summary}")
    print(summary.to_string(index=False))
    if events.empty:
        print("No event rows were generated.")


if __name__ == "__main__":
    main()
