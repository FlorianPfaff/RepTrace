from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

import numpy as np
import pandas as pd


def summarize_metric_table(
    frame: pd.DataFrame,
    value_column: str,
    group_columns: Sequence[str] | str | None,
    participant_column: str | None = None,
    chance_column: str | None = None,
    scale: float = 1.0,
    *,
    percent_scale: float | None = None,
    percent_prefix: str = "percent",
    chance_percent_column: str | None = None,
    chance_class_columns: Sequence[str] | str | None = None,
    permutation_p_column: str | None = None,
    p_value_thresholds: Sequence[float] = (0.05, 0.01),
    zero_singleton_dispersion: bool = False,
) -> pd.DataFrame:
    """Summarize a figure-independent metric table across rows or participants.

    Optional keyword arguments add common grouped-reporting fields while keeping
    the default output backward-compatible: percentage-scaled metric summaries,
    chance-level ranges and class-count summaries, permutation p-value counts,
    and zero-valued dispersion for singleton groups.
    """
    group_columns = _normalize_columns(group_columns)
    chance_class_columns = _normalize_columns(chance_class_columns)
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
        mean, std, sem, median = _series_summary(values, zero_singleton_dispersion=zero_singleton_dispersion)
        row.update(
            {
                "n_rows": int(len(group)),
                f"{value_column}_mean": mean,
                f"{value_column}_std": std,
                f"{value_column}_sem": sem,
                f"{value_column}_median": median,
            }
        )
        if percent_scale is not None:
            row.update(
                {
                    f"{percent_prefix}_mean": _scaled_or_nan(mean, percent_scale),
                    f"{percent_prefix}_median": _scaled_or_nan(median, percent_scale),
                    f"{percent_prefix}_std": _scaled_or_nan(std, percent_scale),
                    f"{percent_prefix}_sem": _scaled_or_nan(sem, percent_scale),
                }
            )
        if participant_column is not None:
            row["n_participants"] = int(group[participant_column].nunique(dropna=True))
        if chance_column is not None:
            chance_values = _chance_values_for_group(
                group,
                chance_column,
                chance_class_columns=chance_class_columns,
                scale=scale,
            )
            difference = values - chance_values
            chance_mean = _nanmean(chance_values)
            row[f"{chance_column}_mean"] = chance_mean
            row[f"{value_column}_above_chance_count"] = int((difference > 0).sum())
            row[f"{value_column}_minus_{chance_column}_mean"] = _nanmean(difference)
            if chance_percent_column is not None and percent_scale is not None:
                row[chance_percent_column] = _scaled_or_nan(chance_mean, percent_scale)
            if chance_class_columns:
                chance_classes = _chance_classes_for_group(
                    group,
                    chance_column,
                    chance_class_columns=chance_class_columns,
                    scale=scale,
                )
                row[f"{chance_column}_min"] = _nanmin(chance_values)
                row[f"{chance_column}_max"] = _nanmax(chance_values)
                row["chance_classes_mean"] = _nanmean(chance_classes)
                row["chance_classes_counts"] = _chance_classes_counts(chance_classes)
        if permutation_p_column is not None:
            p_values = _numeric_column_values(group, permutation_p_column)
            finite_p_values = p_values[np.isfinite(p_values)]
            row["n_with_permutation"] = int(finite_p_values.size)
            for threshold in p_value_thresholds:
                row[f"n_significant_p_{_threshold_suffix(threshold)}"] = int(np.sum(finite_p_values < float(threshold)))
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
    grouper: str | list[str] = group_columns[0] if len(group_columns) == 1 else list(group_columns)
    yield from frame.groupby(grouper, dropna=False, sort=True)


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


def _series_summary(values: pd.Series, *, zero_singleton_dispersion: bool) -> tuple[float, float, float, float]:
    finite = _finite_array(values)
    if finite.size == 0:
        return np.nan, np.nan, np.nan, np.nan
    mean = float(np.mean(finite))
    median = float(np.median(finite))
    if finite.size == 1:
        dispersion = 0.0 if zero_singleton_dispersion else np.nan
        return mean, float(dispersion), float(dispersion), median
    std = float(np.std(finite, ddof=1))
    sem = float(std / np.sqrt(finite.size))
    return mean, std, sem, median


def _finite_array(values: object) -> np.ndarray:
    parsed = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    return parsed[np.isfinite(parsed)]


def _scaled_or_nan(value: object, scale: float) -> float:
    return _float_or_nan(value) * scale if pd.notna(value) else float("nan")


def _chance_values_for_group(
    group: pd.DataFrame,
    chance_column: str,
    *,
    chance_class_columns: Sequence[str],
    scale: float,
) -> pd.Series:
    if not chance_class_columns:
        return group[chance_column]
    return pd.Series(
        [_row_chance_accuracy(row, chance_column, chance_class_columns, scale=scale) for _, row in group.iterrows()],
        index=group.index,
        dtype=float,
    )


def _chance_classes_for_group(
    group: pd.DataFrame,
    chance_column: str,
    *,
    chance_class_columns: Sequence[str],
    scale: float,
) -> pd.Series:
    return pd.Series(
        [_row_chance_classes(row, chance_column, chance_class_columns, scale=scale) for _, row in group.iterrows()],
        index=group.index,
        dtype=float,
    )


def _row_chance_accuracy(row: pd.Series, chance_column: str, chance_class_columns: Sequence[str], *, scale: float) -> float:
    chance = _positive_float(row.get(chance_column))
    if chance is not None:
        return chance
    class_count = _row_chance_classes(row, chance_column, chance_class_columns, scale=scale)
    if not np.isfinite(class_count) or class_count <= 0.0:
        return np.nan
    return float(scale) / class_count


def _row_chance_classes(row: pd.Series, chance_column: str, chance_class_columns: Sequence[str], *, scale: float) -> float:
    for key in chance_class_columns:
        if key not in row:
            continue
        class_count = _positive_int(row.get(key))
        if class_count is not None:
            return float(class_count)
    chance = _positive_float(row.get(chance_column))
    if chance is None or scale <= 0.0:
        return np.nan
    return float(round(float(scale) / chance))


def _chance_classes_counts(chance_classes: object) -> str:
    counter: Counter[int] = Counter()
    for class_count in _finite_array(chance_classes):
        if class_count > 0.0:
            counter[int(round(class_count))] += 1
    return ";".join(f"{key}:{counter[key]}" for key in sorted(counter))


def _numeric_column_values(frame: pd.DataFrame, column: str) -> np.ndarray:
    if column not in frame.columns:
        return np.asarray([], dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)


def _positive_int(value: object) -> int | None:
    try:
        parsed = int(float(value))
    except (TypeError, ValueError, OverflowError):
        return None
    return parsed if parsed > 0 else None


def _positive_float(value: object) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return parsed if np.isfinite(parsed) and parsed > 0.0 else None


def _nanmean(values: object) -> float:
    finite = _finite_array(values)
    return float(np.mean(finite)) if finite.size else np.nan


def _nanmin(values: object) -> float:
    finite = _finite_array(values)
    return float(np.min(finite)) if finite.size else np.nan


def _nanmax(values: object) -> float:
    finite = _finite_array(values)
    return float(np.max(finite)) if finite.size else np.nan


def _threshold_suffix(threshold: float) -> str:
    return f"{float(threshold):g}"
