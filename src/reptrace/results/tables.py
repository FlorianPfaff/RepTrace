from __future__ import annotations

from collections.abc import Sequence

import pandas as pd


def summarize_metric_table(
    frame: pd.DataFrame,
    value_column: str,
    group_columns: Sequence[str],
    participant_column: str | None = None,
    chance_column: str | None = None,
    scale: float = 1.0,
) -> pd.DataFrame:
    """Summarize a figure-independent metric table across rows or participants."""
    group_columns = _normalize_columns(group_columns)
    required_columns = [value_column, *group_columns]
    if participant_column is not None:
        required_columns.append(participant_column)
    if chance_column is not None:
        required_columns.append(chance_column)
    _require_columns(frame, required_columns)

    working = frame.copy()
    working[value_column] = pd.to_numeric(working[value_column], errors="coerce") * scale
    if chance_column is not None:
        working[chance_column] = pd.to_numeric(working[chance_column], errors="coerce") * scale

    rows: list[dict[str, object]] = []
    for group_key, group in _iter_groups(working, group_columns):
        row = _group_row(group_columns, group_key)
        values = group[value_column]
        row.update(
            {
                "n_rows": int(len(group)),
                f"{value_column}_mean": _float_or_nan(values.mean()),
                f"{value_column}_std": _float_or_nan(values.std()),
                f"{value_column}_sem": _float_or_nan(values.sem()),
                f"{value_column}_median": _float_or_nan(values.median()),
            }
        )
        if participant_column is not None:
            row["n_participants"] = int(group[participant_column].nunique(dropna=True))
        if chance_column is not None:
            chance_values = group[chance_column]
            difference = values - chance_values
            row[f"{chance_column}_mean"] = _float_or_nan(chance_values.mean())
            row[f"{value_column}_above_chance_count"] = int((difference > 0).sum())
            row[f"{value_column}_minus_{chance_column}_mean"] = _float_or_nan(difference.mean())
        rows.append(row)

    return _sorted_frame(rows, group_columns)


def peak_metric_rows(
    frame: pd.DataFrame,
    metric_column: str,
    group_columns: Sequence[str],
    time_column: str = "time",
    prefer_time: float = 0.0,
) -> pd.DataFrame:
    """Select the peak metric row in each group, breaking ties toward a preferred time."""
    group_columns = _normalize_columns(group_columns)
    _require_columns(frame, [metric_column, time_column, *group_columns])

    rows: list[pd.Series] = []
    for _, group in _iter_groups(frame, group_columns):
        ranked = group.copy()
        ranked["_peak_distance_to_prefer_time"] = (pd.to_numeric(ranked[time_column], errors="coerce") - prefer_time).abs()
        ranked = ranked.sort_values([metric_column, "_peak_distance_to_prefer_time", time_column], ascending=[False, True, True], na_position="last", kind="mergesort")
        selected = ranked.iloc[0].drop(labels=["_peak_distance_to_prefer_time"])
        selected["peak_distance_to_prefer_time"] = float(ranked.iloc[0]["_peak_distance_to_prefer_time"])
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


def _require_columns(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Data frame is missing required columns: {missing}")


def _iter_groups(frame: pd.DataFrame, group_columns: Sequence[str]):
    if not group_columns:
        yield (), frame
        return
    yield from frame.groupby(list(group_columns), dropna=False, sort=True)


def _group_row(group_columns: Sequence[str], group_key: object) -> dict[str, object]:
    if not group_columns:
        return {}
    if len(group_columns) == 1 and not isinstance(group_key, tuple):
        group_key = (group_key,)
    return dict(zip(group_columns, group_key))


def _sorted_frame(rows: list[dict[str, object]], group_columns: Sequence[str]) -> pd.DataFrame:
    result = pd.DataFrame(rows)
    if group_columns and not result.empty:
        result = result.sort_values(list(group_columns), kind="mergesort")
    return result.reset_index(drop=True)


def _float_or_nan(value: object) -> float:
    return float(value) if pd.notna(value) else float("nan")
