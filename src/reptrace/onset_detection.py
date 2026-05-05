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
        class_names = []
        for class_index in predicted_labels:
            class_column = f"class_{class_index}"
            if class_column in frame.columns:
                class_names.append(frame[class_column].iloc[0])
            else:
                class_names.append(str(class_index))
        frame["predicted_class"] = class_names
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


# pylint: disable-next=too-many-arguments
def _event_row(
    group_values: dict,
    sequence_frame: pd.DataFrame,
    detection_row: pd.Series | None,
    *,
    threshold: float,
    threshold_window: tuple[float, float],
    threshold_quantile: float,
    score_column: str,
    detection_start: float | None,
) -> dict:
    first_row = sequence_frame.iloc[0]
    detected = detection_row is not None
    detection_time = float(detection_row["time"]) if detection_row is not None else np.nan
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
        "n_time_points": len(sequence_frame),
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
    if "true_label" in row and "predicted_label" in row and pd.notna(row["true_label"]) and pd.notna(row["predicted_label"]):
        return int(row["true_label"]) == int(row["predicted_label"])
    if "true_class" in row and "predicted_class" in row and pd.notna(row["true_class"]) and pd.notna(row["predicted_class"]):
        return str(row["true_class"]) == str(row["predicted_class"])
    return False


def _annotate_group_threshold(
    frame: pd.DataFrame,
    *,
    threshold_window: tuple[float, float],
    threshold_quantile: float,
    score_column: str,
) -> pd.DataFrame:
    frame = frame.copy()
    frame["_onset_score"] = _score_values(frame, score_column)
    threshold = _threshold_for_group(
        frame,
        threshold_window=threshold_window,
        threshold_quantile=threshold_quantile,
        score_column=score_column,
    )
    frame["onset_score"] = frame["_onset_score"]
    frame["score_threshold"] = threshold
    frame["above_threshold"] = np.isfinite(threshold) & (frame["_onset_score"] >= threshold)
    frame["score_column"] = score_column
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
) -> pd.DataFrame:
    """Annotate observation rows with baseline-derived threshold crossings."""

    if not 0.0 <= threshold_quantile <= 1.0:
        raise ValueError("threshold_quantile must be between 0 and 1.")
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
            )
        )
    return pd.concat(frames, ignore_index=True) if frames else observations.copy()


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


# pylint: disable-next=too-many-locals
def detect_onsets(
    observations: pd.DataFrame,
    *,
    threshold_window: tuple[float, float] = DEFAULT_THRESHOLD_WINDOW,
    threshold_quantile: float = DEFAULT_THRESHOLD_QUANTILE,
    score_column: str = "confidence",
    detection_start: float | None = None,
) -> pd.DataFrame:
    """Find the first threshold-crossing time for each probability-observation sequence."""

    if not 0.0 <= threshold_quantile <= 1.0:
        raise ValueError("threshold_quantile must be between 0 and 1.")
    if "time" not in observations.columns:
        raise ValueError("Observation rows must contain a time column.")

    observations = annotate_threshold_crossings(
        observations,
        threshold_window=threshold_window,
        threshold_quantile=threshold_quantile,
        score_column=score_column,
    )
    group_columns = _group_columns(observations)
    sequence_columns = _sequence_columns(observations)
    event_rows = []

    grouped = observations.groupby(group_columns, sort=True) if group_columns else [((), observations)]
    for keys, group_frame in grouped:
        key_values = keys if isinstance(keys, tuple) else (keys,)
        group_values = dict(zip(group_columns, key_values, strict=True))
        threshold = group_frame["score_threshold"].iloc[0]
        for _, sequence_frame in group_frame.sort_values([*sequence_columns, "time"]).groupby(sequence_columns, sort=True):
            candidates = sequence_frame
            if detection_start is not None:
                candidates = candidates.loc[candidates["time"] >= detection_start]
            detected_rows = candidates.loc[np.isfinite(threshold) & (candidates["_onset_score"] >= threshold)]
            detection_row = detected_rows.iloc[0] if not detected_rows.empty else None
            event_rows.append(
                _event_row(
                    group_values,
                    sequence_frame,
                    detection_row,
                    threshold=threshold,
                    threshold_window=threshold_window,
                    threshold_quantile=threshold_quantile,
                    score_column=score_column,
                    detection_start=detection_start,
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
                "score_threshold": group_frame["score_threshold"].iloc[0] if "score_threshold" in group_frame else np.nan,
                "threshold_quantile": group_frame["threshold_quantile"].iloc[0] if "threshold_quantile" in group_frame else np.nan,
                "threshold_window_start": group_frame["threshold_window_start"].iloc[0] if "threshold_window_start" in group_frame else np.nan,
                "threshold_window_stop": group_frame["threshold_window_stop"].iloc[0] if "threshold_window_stop" in group_frame else np.nan,
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
    )
    events = detect_onsets(
        thresholded_observations,
        threshold_window=threshold_window,
        threshold_quantile=threshold_quantile,
        score_column=score_column,
        detection_start=detection_start,
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
    parser = argparse.ArgumentParser(description="Detect representation onsets from RepTrace probability observation CSVs.")
    parser.add_argument("observation_csv", nargs="+", help="Observation CSVs or glob patterns emitted by RepTrace/PyMEGDec adapters.")
    parser.add_argument("--out-events", type=Path, required=True)
    parser.add_argument("--out-summary", type=Path, required=True)
    parser.add_argument("--threshold-window", nargs=2, type=float, default=DEFAULT_THRESHOLD_WINDOW, metavar=("START", "STOP"))
    parser.add_argument("--threshold-quantile", type=float, default=DEFAULT_THRESHOLD_QUANTILE)
    parser.add_argument("--score-column", default="confidence")
    parser.add_argument("--detection-start", type=float)
    parser.add_argument("--detection-window", nargs=2, type=float, default=DEFAULT_DETECTION_WINDOW, metavar=("START", "STOP"))
    parser.add_argument("--out-thresholded-observations", type=Path)
    parser.add_argument("--out-threshold-summary", type=Path)
    args = parser.parse_args()

    events, summary = detect_onsets_from_csvs(
        _expand_paths(args.observation_csv),
        threshold_window=tuple(args.threshold_window),
        threshold_quantile=args.threshold_quantile,
        score_column=args.score_column,
        detection_start=args.detection_start,
        detection_window=tuple(args.detection_window),
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
