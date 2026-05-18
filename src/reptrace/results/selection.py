from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

METRIC_OBJECTIVES = {
    "accuracy": "max",
    "log_loss": "min",
    "brier": "min",
    "ece": "min",
}
SELECTION_METRIC_CHOICES = tuple(METRIC_OBJECTIVES)
PROBABILITY_QUALITY_METRICS = ("log_loss", "brier", "ece")


def metric_objective(metric_column: str) -> str:
    """Return whether a decoding metric should be maximized or minimized."""
    try:
        return METRIC_OBJECTIVES[metric_column]
    except KeyError as exc:
        choices = ", ".join(SELECTION_METRIC_CHOICES)
        raise ValueError(f"Unknown selection metric '{metric_column}'. Expected one of: {choices}.") from exc


def metric_sort_ascending(metric_column: str) -> bool:
    """Return ``True`` when smaller metric values are better."""
    return metric_objective(metric_column) == "min"


def metric_value_column(frame: pd.DataFrame, metric_column: str) -> str:
    """Resolve a raw or aggregated metric column name in ``frame``."""
    if metric_column in frame.columns:
        return metric_column
    mean_column = f"{metric_column}_mean"
    if mean_column in frame.columns:
        return mean_column
    raise ValueError(f"Data frame is missing metric column '{metric_column}' or '{mean_column}'.")


def select_metric_rows(
    frame: pd.DataFrame,
    metric_column: str,
    group_columns: Sequence[str] | str | None = None,
    *,
    time_column: str = "time",
    prefer_time: float = 0.0,
) -> pd.DataFrame:
    """Select the best metric row per group with calibration-aware directionality.

    Accuracy is maximized. Proper scoring and calibration losses are minimized.
    Ties are resolved toward ``prefer_time`` and then toward the earlier time bin.
    """
    group_columns = _normalize_columns(group_columns)
    resolved_metric_column = metric_value_column(frame, metric_column)
    missing = [column for column in (*group_columns, time_column, resolved_metric_column) if column not in frame.columns]
    if missing:
        raise ValueError(f"Data frame is missing required columns: {missing}")

    rows: list[pd.Series] = []
    metric_ascending = metric_sort_ascending(metric_column)
    for _, group in _iter_groups(frame, group_columns):
        ranked = group.copy()
        ranked["_selection_metric_value"] = pd.to_numeric(ranked[resolved_metric_column], errors="coerce")
        ranked["_selection_distance_to_prefer_time"] = (pd.to_numeric(ranked[time_column], errors="coerce") - prefer_time).abs()
        ranked = ranked.sort_values(
            ["_selection_metric_value", "_selection_distance_to_prefer_time", time_column],
            ascending=[metric_ascending, True, True],
            na_position="last",
            kind="mergesort",
        )
        selected = ranked.iloc[0].drop(labels=["_selection_metric_value", "_selection_distance_to_prefer_time"])
        selected["selection_metric"] = metric_column
        selected["selection_metric_column"] = resolved_metric_column
        selected["selection_objective"] = metric_objective(metric_column)
        selected["selection_metric_value"] = float(ranked.iloc[0]["_selection_metric_value"])
        selected["selection_distance_to_prefer_time"] = float(ranked.iloc[0]["_selection_distance_to_prefer_time"])
        rows.append(selected)

    result = pd.DataFrame(rows)
    if group_columns and not result.empty:
        result = result.sort_values(group_columns, kind="mergesort")
    return result.reset_index(drop=True)


def _normalize_columns(columns: Sequence[str] | str | None) -> list[str]:
    if columns is None:
        return []
    if isinstance(columns, str):
        return [columns]
    return list(dict.fromkeys(columns))


def _iter_groups(frame: pd.DataFrame, group_columns: Sequence[str]):
    if not group_columns:
        yield (), frame
        return
    grouper: str | list[str] = group_columns[0] if len(group_columns) == 1 else list(group_columns)
    yield from frame.groupby(grouper, dropna=False, sort=True)
