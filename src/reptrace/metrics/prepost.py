from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

Window = tuple[float, float]


def summarize_window_metric(
    frame: pd.DataFrame,
    metric_column: str,
    window: Window,
    time_column: str = "time",
    group_columns: Sequence[str] = (),
) -> pd.DataFrame:
    """Summarize one metric inside an inclusive time window."""
    group_columns = _normalize_columns(group_columns)
    _require_columns(frame, [time_column, metric_column, *group_columns])
    window_start, window_stop = _validate_window(window)

    window_frame = frame.loc[(frame[time_column] >= window_start) & (frame[time_column] <= window_stop)]
    if window_frame.empty:
        raise ValueError(f"No rows fall inside window [{window_start}, {window_stop}].")

    rows: list[dict[str, object]] = []
    for group_key, group in _iter_groups(window_frame, group_columns):
        row = _group_row(group_columns, group_key)
        values = pd.to_numeric(group[metric_column], errors="coerce")
        row.update(
            {
                "window_start": window_start,
                "window_stop": window_stop,
                "n_rows": int(values.notna().sum()),
                f"{metric_column}_mean": _float_or_nan(values.mean()),
                f"{metric_column}_std": _float_or_nan(values.std()),
                f"{metric_column}_sem": _float_or_nan(values.sem()),
            }
        )
        rows.append(row)

    return _sorted_frame(rows, group_columns)


def compare_prepost_windows(
    frame: pd.DataFrame,
    metric_column: str,
    pre_window: Window,
    post_window: Window,
    time_column: str = "time",
    group_columns: Sequence[str] = (),
) -> pd.DataFrame:
    """Compare a metric between inclusive pre and post time windows."""
    group_columns = _normalize_columns(group_columns)
    pre_summary = summarize_window_metric(frame, metric_column, pre_window, time_column=time_column, group_columns=group_columns)
    post_summary = summarize_window_metric(frame, metric_column, post_window, time_column=time_column, group_columns=group_columns)

    pre_summary = pre_summary.rename(
        columns={
            "window_start": "pre_window_start",
            "window_stop": "pre_window_stop",
            "n_rows": "n_pre_rows",
            f"{metric_column}_mean": f"{metric_column}_pre_mean",
            f"{metric_column}_std": f"{metric_column}_pre_std",
            f"{metric_column}_sem": f"{metric_column}_pre_sem",
        }
    )
    post_summary = post_summary.rename(
        columns={
            "window_start": "post_window_start",
            "window_stop": "post_window_stop",
            "n_rows": "n_post_rows",
            f"{metric_column}_mean": f"{metric_column}_post_mean",
            f"{metric_column}_std": f"{metric_column}_post_std",
            f"{metric_column}_sem": f"{metric_column}_post_sem",
        }
    )

    if group_columns:
        merged = pre_summary.merge(post_summary, on=group_columns, how="outer")
    else:
        merged = pd.concat([pre_summary.reset_index(drop=True), post_summary.reset_index(drop=True)], axis=1)
    merged[f"{metric_column}_post_minus_pre"] = merged[f"{metric_column}_post_mean"] - merged[f"{metric_column}_pre_mean"]
    return _sorted_frame(merged.to_dict("records"), group_columns)


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


def _validate_window(window: Window) -> Window:
    if len(window) != 2:
        raise ValueError("window must contain exactly two values")
    window_start, window_stop = float(window[0]), float(window[1])
    if window_start > window_stop:
        raise ValueError("window start must be less than or equal to window stop")
    return window_start, window_stop


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
