from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from reptrace.decoding.temporal_generalization import TemporalFeatureWindow


def compute_time_resolved_decoding(
    train_windows: Sequence[TemporalFeatureWindow],
    test_windows: Sequence[TemporalFeatureWindow],
    *,
    fit_model: Callable[[TemporalFeatureWindow], Any],
    predict_labels: Callable[[Any, TemporalFeatureWindow], Sequence],
    chance_accuracy: float | None = None,
    metadata: Mapping[str, object] | None = None,
    model_metadata: Callable[[Any], Mapping[str, object]] | None = None,
    score_metadata: Callable[[Any, TemporalFeatureWindow, TemporalFeatureWindow, np.ndarray], Mapping[str, object]] | None = None,
    prediction_centers: Sequence[float] = (),
    prediction_metadata: Callable[[TemporalFeatureWindow, int, object, object], Mapping[str, object]] | None = None,
    center_decimals: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run supervised decoding at matched train/test time windows.

    Dataset-specific code supplies feature windows and callbacks for fitting and
    prediction. RepTrace owns the time-window loop, accuracy scoring, chance
    comparison, and optional trial-level prediction table.
    """

    if not train_windows:
        raise ValueError("Need at least one train window.")
    if not test_windows:
        raise ValueError("Need at least one test window.")

    test_by_center = _window_mapping(test_windows, center_decimals=center_decimals, role="test")
    prediction_keys = {_center_key(center, center_decimals) for center in prediction_centers}
    base_metadata = dict(metadata or {})
    score_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []

    for train_window in sorted(train_windows, key=lambda window: _center_key(window.center, center_decimals)):
        train_key = _center_key(train_window.center, center_decimals)
        if train_key not in test_by_center:
            raise ValueError(f"Missing test window for train-window center {train_window.center}.")
        test_window = test_by_center[train_key]
        _validate_window_labels(train_window, role="train")
        _validate_window_labels(test_window, role="test")

        model = fit_model(train_window)
        predictions = np.asarray(predict_labels(model, test_window))
        labels = np.asarray(test_window.labels)
        if len(predictions) != len(labels):
            raise ValueError(
                "Predicted label count must match test label count "
                f"for window {test_window.center}: {len(predictions)} != {len(labels)}."
            )

        accuracy = float(np.mean(predictions == labels)) if len(labels) else np.nan
        chance = _chance_accuracy(chance_accuracy, labels)
        row = {
            **base_metadata,
            **_window_metadata(test_window),
            "accuracy": accuracy,
            "percent": 100.0 * accuracy if np.isfinite(accuracy) else np.nan,
            "chance_accuracy": chance,
            "chance_percent": 100.0 * chance if np.isfinite(chance) else np.nan,
            "above_chance": bool(np.isfinite(accuracy) and np.isfinite(chance) and accuracy > chance),
            "n_train_trials": len(train_window.labels),
            "n_validation_trials": len(labels),
            "n_train_classes": len(np.unique(np.asarray(train_window.labels))),
            "n_validation_classes": len(np.unique(labels)),
        }
        if model_metadata is not None:
            row.update(model_metadata(model))
        if score_metadata is not None:
            row.update(score_metadata(model, train_window, test_window, predictions))
        score_rows.append(row)

        if train_key in prediction_keys:
            prediction_rows.extend(
                _prediction_rows(
                    base_metadata,
                    row,
                    test_window,
                    labels,
                    predictions,
                    prediction_metadata=prediction_metadata,
                )
            )

    return pd.DataFrame(score_rows), pd.DataFrame(prediction_rows)


def _window_mapping(
    windows: Sequence[TemporalFeatureWindow],
    *,
    center_decimals: int,
    role: str,
) -> dict[float, TemporalFeatureWindow]:
    mapped: dict[float, TemporalFeatureWindow] = {}
    for window in windows:
        key = _center_key(window.center, center_decimals)
        if key in mapped:
            raise ValueError(f"Duplicate {role} window center {window.center}.")
        mapped[key] = window
    return mapped


def _prediction_rows(
    base_metadata: Mapping[str, object],
    score_row: Mapping[str, object],
    window: TemporalFeatureWindow,
    labels: np.ndarray,
    predictions: np.ndarray,
    *,
    prediction_metadata: Callable[[TemporalFeatureWindow, int, object, object], Mapping[str, object]] | None,
) -> list[dict[str, object]]:
    rows = []
    for row_index, (true_label, predicted_label) in enumerate(zip(labels, predictions, strict=True)):
        row = {
            **base_metadata,
            **_model_prediction_metadata(score_row),
            **_window_metadata(window),
            "sample_index": row_index,
            "true_label": true_label,
            "predicted_label": predicted_label,
            "correct": bool(predicted_label == true_label),
        }
        if prediction_metadata is not None:
            row.update(prediction_metadata(window, row_index, true_label, predicted_label))
        rows.append(row)
    return rows


def _model_prediction_metadata(score_row: Mapping[str, object]) -> dict[str, object]:
    excluded_columns = {
        "window_center_s",
        "window_start_s",
        "window_stop_s",
        "accuracy",
        "percent",
        "chance_accuracy",
        "chance_percent",
        "above_chance",
        "n_train_trials",
        "n_validation_trials",
        "n_train_classes",
        "n_validation_classes",
    }
    return {
        column: value
        for column, value in score_row.items()
        if column not in excluded_columns
    }


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


def _window_metadata(window: TemporalFeatureWindow) -> dict[str, object]:
    metadata = dict(window.metadata or {})
    return {
        "window_center_s": float(window.center),
        "window_start_s": float(window.start) if window.start is not None else np.nan,
        "window_stop_s": float(window.stop) if window.stop is not None else np.nan,
        **metadata,
    }
