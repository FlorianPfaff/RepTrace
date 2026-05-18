from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TemporalFeatureWindow:
    """Feature matrix and labels for one train or test time window."""

    center: float
    features: Any
    labels: Sequence
    start: float | None = None
    stop: float | None = None
    metadata: Mapping[str, object] | None = None


def compute_temporal_generalization_matrix(
    train_windows: Sequence[TemporalFeatureWindow],
    test_windows: Sequence[TemporalFeatureWindow],
    *,
    fit_model: Callable[[TemporalFeatureWindow], Any],
    predict_labels: Callable[[Any, TemporalFeatureWindow], Sequence],
    chance_accuracy: float | None = None,
    metadata: Mapping[str, object] | None = None,
    model_metadata: Callable[[Any], Mapping[str, object]] | None = None,
    center_decimals: int = 10,
) -> pd.DataFrame:
    """Compute a train-time by test-time temporal-generalization score table.

    RepTrace owns the dataset-independent orchestration and scoring: train one
    model per train window, evaluate it on every test window, and emit a compact
    figure-independent table. Dataset-specific projects remain responsible for
    loading data and constructing ``TemporalFeatureWindow`` objects.
    """

    if not train_windows:
        raise ValueError("Need at least one train window.")
    if not test_windows:
        raise ValueError("Need at least one test window.")

    base_metadata = dict(metadata or {})
    rows: list[dict[str, object]] = []
    for train_window in sorted(train_windows, key=lambda window: _center_key(window.center, center_decimals)):
        _validate_window_labels(train_window, role="train")
        model = fit_model(train_window)
        fitted_metadata = dict(model_metadata(model) if model_metadata is not None else {})
        for test_window in sorted(test_windows, key=lambda window: _center_key(window.center, center_decimals)):
            _validate_window_labels(test_window, role="test")
            predictions = np.asarray(predict_labels(model, test_window))
            labels = np.asarray(test_window.labels)
            if len(predictions) != len(labels):
                raise ValueError(
                    "Predicted label count must match test label count "
                    f"for test window {test_window.center}: {len(predictions)} != {len(labels)}."
                )
            accuracy = float(np.mean(predictions == labels)) if len(labels) else np.nan
            chance = _chance_accuracy(chance_accuracy, labels)
            rows.append(
                {
                    **base_metadata,
                    **_window_metadata("train", train_window),
                    **_window_metadata("test", test_window),
                    "is_diagonal": _center_key(train_window.center, center_decimals) == _center_key(test_window.center, center_decimals),
                    "accuracy": accuracy,
                    "percent": 100.0 * accuracy if np.isfinite(accuracy) else np.nan,
                    "chance_accuracy": chance,
                    "chance_percent": 100.0 * chance if np.isfinite(chance) else np.nan,
                    "above_chance": bool(np.isfinite(accuracy) and np.isfinite(chance) and accuracy > chance),
                    "n_train_trials": len(train_window.labels),
                    "n_validation_trials": len(labels),
                    "n_train_classes": len(np.unique(np.asarray(train_window.labels))),
                    "n_validation_classes": len(np.unique(labels)),
                    **fitted_metadata,
                }
            )
    return pd.DataFrame(rows)


def summarize_temporal_generalization_matrix(
    frame: pd.DataFrame,
    *,
    group_columns: Sequence[str] = (),
    accuracy_column: str = "accuracy",
    chance_column: str | None = "chance_accuracy",
) -> pd.DataFrame:
    """Summarize temporal-generalization rows across participants or repeats."""

    if frame.empty:
        return pd.DataFrame()
    required_columns = set(group_columns) | {accuracy_column}
    missing_columns = required_columns.difference(frame.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required columns: {missing}.")

    grouped = frame.groupby(list(group_columns), sort=True, dropna=False) if group_columns else [((), frame)]
    rows: list[dict[str, object]] = []
    for keys, group in grouped:
        key_values = keys if isinstance(keys, tuple) else (keys,)
        values = pd.to_numeric(group[accuracy_column], errors="coerce").dropna().to_numpy(dtype=float)
        mean_value = float(np.mean(values)) if len(values) else np.nan
        median_value = float(np.median(values)) if len(values) else np.nan
        std_value = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        sem_value = float(std_value / np.sqrt(len(values))) if len(values) > 1 else 0.0
        row: dict[str, object] = dict(zip(group_columns, key_values, strict=True))
        row.update(
            {
                "n_rows": int(len(group)),
                f"{accuracy_column}_mean": mean_value,
                f"{accuracy_column}_median": median_value,
                f"{accuracy_column}_std": std_value,
                f"{accuracy_column}_sem": sem_value,
            }
        )
        row.update(
            {
                "percent_mean": 100.0 * mean_value if np.isfinite(mean_value) else np.nan,
                "percent_median": 100.0 * median_value if np.isfinite(median_value) else np.nan,
                "percent_std": 100.0 * std_value if np.isfinite(std_value) else np.nan,
                "percent_sem": 100.0 * sem_value if np.isfinite(sem_value) else np.nan,
            }
        )
        if chance_column is not None and chance_column in group.columns:
            chance_values = pd.to_numeric(group[chance_column], errors="coerce").dropna()
            chance = float(chance_values.iloc[0]) if not chance_values.empty else np.nan
            row["chance_accuracy"] = chance
            row["chance_percent"] = 100.0 * chance if np.isfinite(chance) else np.nan
            row["above_chance_count"] = int((values > chance).sum()) if np.isfinite(chance) else 0
        if "is_diagonal" in group.columns:
            diagonal_values = set(group["is_diagonal"].astype(bool))
            row["is_diagonal"] = bool(diagonal_values == {True})
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_train_time_stability(
    frame: pd.DataFrame,
    *,
    group_columns: Sequence[str] = (),
    train_time_column: str = "train_window_center_s",
    test_time_column: str = "test_window_center_s",
    score_column: str = "accuracy",
    chance_column: str | None = "chance_accuracy",
    diagonal_column: str = "is_diagonal",
) -> pd.DataFrame:
    """Rank train-time windows by cross-time generalization stability.

    The input is a temporal-generalization matrix in long form. Each output row
    corresponds to one train-time window within the requested grouping columns.
    The primary stability score is the mean held-out score over all tested times
    and folds, so high ranks identify train-time regions that generalize beyond a
    single diagonal time bin.
    """

    if frame.empty:
        return pd.DataFrame()
    required_columns = set(group_columns) | {train_time_column, score_column}
    missing_columns = required_columns.difference(frame.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required columns: {missing}.")

    grouped = frame.groupby([*group_columns, train_time_column], sort=True, dropna=False)
    rows: list[dict[str, object]] = []
    for keys, group in grouped:
        key_values = keys if isinstance(keys, tuple) else (keys,)
        row: dict[str, object] = dict(zip([*group_columns, train_time_column], key_values, strict=True))

        values = pd.to_numeric(group[score_column], errors="coerce").dropna().to_numpy(dtype=float)
        mean_value = float(np.mean(values)) if len(values) else np.nan
        median_value = float(np.median(values)) if len(values) else np.nan
        std_value = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        sem_value = float(std_value / np.sqrt(len(values))) if len(values) > 1 else 0.0

        row.update(
            {
                "n_rows": int(len(group)),
                "n_test_windows": int(group[test_time_column].nunique(dropna=False))
                if test_time_column in group.columns
                else int(len(group)),
                f"{score_column}_mean": mean_value,
                f"{score_column}_median": median_value,
                f"{score_column}_std": std_value,
                f"{score_column}_sem": sem_value,
                "percent_mean": 100.0 * mean_value if np.isfinite(mean_value) else np.nan,
                "percent_median": 100.0 * median_value if np.isfinite(median_value) else np.nan,
            }
        )

        if chance_column is not None and chance_column in group.columns:
            chance_values = pd.to_numeric(group[chance_column], errors="coerce").dropna()
            chance = float(chance_values.mean()) if not chance_values.empty else np.nan
            row["chance_accuracy"] = chance
            row["chance_percent"] = 100.0 * chance if np.isfinite(chance) else np.nan
            row[f"{score_column}_above_chance_mean"] = (
                mean_value - chance if np.isfinite(mean_value) and np.isfinite(chance) else np.nan
            )
            row["above_chance_count"] = int((values > chance).sum()) if np.isfinite(chance) else 0

        if diagonal_column in group.columns:
            diagonal_mask = group[diagonal_column].astype(bool).to_numpy()
            score_values = pd.to_numeric(group[score_column], errors="coerce").to_numpy(dtype=float)
            diagonal_values = score_values[diagonal_mask]
            off_diagonal_values = score_values[~diagonal_mask]
            row[f"{score_column}_diagonal_mean"] = _nanmean_or_nan(diagonal_values)
            row[f"{score_column}_off_diagonal_mean"] = _nanmean_or_nan(off_diagonal_values)
            row["n_diagonal_rows"] = int(np.isfinite(diagonal_values).sum())
            row["n_off_diagonal_rows"] = int(np.isfinite(off_diagonal_values).sum())

        rows.append(row)

    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary

    rank_column = f"{score_column}_mean"
    if group_columns:
        summary["stability_rank"] = (
            summary.groupby(list(group_columns), dropna=False)[rank_column]
            .rank(method="dense", ascending=False)
            .astype(int)
        )
        sort_columns = [*group_columns, "stability_rank", train_time_column]
    else:
        summary["stability_rank"] = summary[rank_column].rank(method="dense", ascending=False).astype(int)
        sort_columns = ["stability_rank", train_time_column]
    return summary.sort_values(sort_columns, kind="mergesort").reset_index(drop=True)


def _validate_window_labels(window: TemporalFeatureWindow, *, role: str) -> None:
    labels = np.asarray(window.labels)
    if labels.ndim != 1:
        raise ValueError(f"{role} window labels must be one-dimensional.")
    if len(labels) == 0:
        raise ValueError(f"{role} window labels must not be empty.")


def _center_key(center: float, decimals: int) -> float:
    return round(float(center), decimals)


def _chance_accuracy(chance_accuracy: float | None, labels: np.ndarray) -> float:
    if chance_accuracy is not None:
        return float(chance_accuracy)
    n_classes = len(np.unique(labels))
    return 1.0 / n_classes if n_classes else np.nan


def _window_metadata(prefix: str, window: TemporalFeatureWindow) -> dict[str, object]:
    metadata = dict(window.metadata or {})
    return {
        f"{prefix}_window_center_s": float(window.center),
        f"{prefix}_window_start_s": float(window.start) if window.start is not None else np.nan,
        f"{prefix}_window_stop_s": float(window.stop) if window.stop is not None else np.nan,
        **metadata,
    }


def _nanmean_or_nan(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(np.mean(finite)) if len(finite) else np.nan
