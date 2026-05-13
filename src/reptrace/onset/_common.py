from __future__ import annotations

import numpy as np
import pandas as pd

from reptrace.temporal_model import probability_columns

DEFAULT_THRESHOLD_WINDOW = (-0.35, -0.05)
DEFAULT_DETECTION_WINDOW = (0.0, float("inf"))
DEFAULT_THRESHOLD_QUANTILE = 0.95
THRESHOLD_METHODS = ("point", "max_run")
GROUP_COLUMNS = ("subject", "decoder", "emission_mode")


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

def _prediction_values(frame: pd.DataFrame) -> np.ndarray:
    if "predicted_label" in frame.columns:
        return frame["predicted_label"].to_numpy(dtype=object)
    if "predicted_class" in frame.columns:
        return frame["predicted_class"].to_numpy(dtype=object)
    return np.full(len(frame), None, dtype=object)

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
