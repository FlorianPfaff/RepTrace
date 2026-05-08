from __future__ import annotations

import glob
from pathlib import Path
from collections.abc import Sequence

import numpy as np
import pandas as pd

from reptrace._onset_constants import GROUP_COLUMNS
from reptrace.temporal_model import probability_columns


def expand_paths(patterns: Sequence[str | Path]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = sorted(glob.glob(str(pattern)))
        if matches:
            paths.extend(Path(match) for match in matches)
        else:
            paths.append(Path(pattern))
    return paths


def group_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in GROUP_COLUMNS if column in frame.columns]


def sequence_columns(frame: pd.DataFrame) -> list[str]:
    identifier = "sequence_id" if "sequence_id" in frame.columns else "sample_index" if "sample_index" in frame.columns else None
    if identifier is None:
        raise ValueError("Observation rows must contain sequence_id or sample_index.")
    return [column for column in ("subject", "fold", identifier) if column in frame.columns]


def window_mask(frame: pd.DataFrame, window: tuple[float, float]) -> pd.Series:
    start, stop = window
    return (frame["time"] >= start) & (frame["time"] <= stop)


def score_values(frame: pd.DataFrame, score_column: str) -> pd.Series:
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


def class_lookup(frame: pd.DataFrame) -> dict[int, str]:
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


def ensure_prediction_columns(frame: pd.DataFrame) -> pd.DataFrame:
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
        lookup = class_lookup(frame)
        frame["predicted_class"] = [lookup.get(int(label), str(int(label))) for label in predicted_labels]
    return frame


def prediction_values(frame: pd.DataFrame) -> np.ndarray:
    if "predicted_label" in frame.columns:
        return frame["predicted_label"].to_numpy(dtype=object)
    if "predicted_class" in frame.columns:
        return frame["predicted_class"].to_numpy(dtype=object)
    return np.full(len(frame), None, dtype=object)


def prediction_value(row: pd.Series) -> object:
    if "predicted_label" in row and pd.notna(row["predicted_label"]):
        return row["predicted_label"]
    if "predicted_class" in row and pd.notna(row["predicted_class"]):
        return row["predicted_class"]
    return None


def sequence_identity(row: pd.Series) -> dict:
    identity = {"sequence_id": row.get("sequence_id", row.get("sample_index", np.nan))}
    for optional_column in ("sample_index", "group", "source_file"):
        if optional_column in row:
            identity[optional_column] = row[optional_column]
    for truth_column in ("true_label", "true_class"):
        if truth_column in row:
            identity[truth_column] = row[truth_column]
    return identity


def is_correct_detection(row: pd.Series) -> bool:
    if "true_label" in row and "predicted_label" in row and pd.notna(row["true_label"]) and pd.notna(row["predicted_label"]):
        return int(row["true_label"]) == int(row["predicted_label"])
    if "true_class" in row and "predicted_class" in row and pd.notna(row["true_class"]) and pd.notna(row["predicted_class"]):
        return str(row["true_class"]) == str(row["predicted_class"])
    return False


def validate_detection_options(
    *,
    threshold_quantile: float | None = None,
    threshold_method: str | None = None,
    threshold_methods: tuple[str, ...] | None = None,
    min_consecutive: int = 1,
    min_duration: float | None = None,
) -> None:
    if threshold_quantile is not None and not 0.0 <= threshold_quantile <= 1.0:
        raise ValueError("threshold_quantile must be between 0 and 1.")
    if threshold_method is not None and threshold_methods is not None and threshold_method not in threshold_methods:
        raise ValueError(f"threshold_method must be one of {threshold_methods}.")
    if min_consecutive < 1:
        raise ValueError("min_consecutive must be at least 1.")
    if min_duration is not None and min_duration < 0:
        raise ValueError("min_duration must be non-negative when provided.")
