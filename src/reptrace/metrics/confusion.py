from __future__ import annotations

from collections.abc import Sequence

import pandas as pd


def confusion_counts(
    frame: pd.DataFrame,
    true_column: str = "true_label",
    predicted_column: str = "predicted_label",
    group_columns: Sequence[str] = (),
) -> pd.DataFrame:
    """Count true/predicted label pairs in a trial-level prediction table."""
    group_columns = _normalize_columns(group_columns)
    _require_columns(frame, [true_column, predicted_column, *group_columns])

    working = frame[[*group_columns, true_column, predicted_column]].rename(
        columns={true_column: "true_label", predicted_column: "predicted_label"}
    )
    keys = [*group_columns, "true_label", "predicted_label"]
    counts = working.groupby(keys, dropna=False, sort=True).size().reset_index(name="count")
    return counts.reset_index(drop=True)


def per_class_accuracy(
    frame: pd.DataFrame,
    true_column: str = "true_label",
    predicted_column: str = "predicted_label",
    participant_column: str | None = None,
    group_columns: Sequence[str] = (),
) -> pd.DataFrame:
    """Summarize one-vs-rest recall/accuracy for each true class."""
    group_columns = _normalize_columns(group_columns)
    required_columns = [true_column, predicted_column, *group_columns]
    if participant_column is not None:
        required_columns.append(participant_column)
    _require_columns(frame, required_columns)

    working_columns = [*group_columns, true_column, predicted_column]
    if participant_column is not None:
        working_columns.append(participant_column)
    working = frame[working_columns].rename(columns={true_column: "true_label", predicted_column: "predicted_label"})
    working["_correct"] = working["true_label"] == working["predicted_label"]

    rows: list[dict[str, object]] = []
    keys = [*group_columns, "true_label"]
    for group_key, group in working.groupby(keys, dropna=False, sort=True):
        row = _group_row(keys, group_key)
        row.update(
            {
                "n_trials": int(len(group)),
                "n_correct": int(group["_correct"].sum()),
                "accuracy": float(group["_correct"].mean()),
            }
        )
        if participant_column is not None:
            row["n_participants"] = int(group[participant_column].nunique(dropna=True))
        rows.append(row)

    return pd.DataFrame(rows).reset_index(drop=True)


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


def _group_row(group_columns: Sequence[str], group_key: object) -> dict[str, object]:
    if len(group_columns) == 1 and not isinstance(group_key, tuple):
        group_key = (group_key,)
    return dict(zip(group_columns, group_key))
