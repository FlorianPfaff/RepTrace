from __future__ import annotations

from collections.abc import Sequence
from collections import Counter
from typing import Any

import numpy as np
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


def most_confused_class_pairs(
    frame: pd.DataFrame,
    true_column: str = "true_label",
    predicted_column: str = "predicted_label",
    participant_column: str | None = None,
    group_columns: Sequence[str] = (),
    metadata: pd.DataFrame | Sequence[dict[str, Any]] | None = None,
    category_columns: Sequence[str] | str | None = None,
    top_n: int | None = None,
) -> pd.DataFrame:
    """Return the most frequent unordered off-diagonal class-pair confusions."""
    pairs = confusion_pair_summary(
        frame,
        true_column=true_column,
        predicted_column=predicted_column,
        participant_column=participant_column,
        group_columns=group_columns,
        metadata=metadata,
        category_columns=category_columns,
    )
    if top_n is None:
        return pairs
    return pairs.head(int(top_n)).reset_index(drop=True)


def confusion_pair_summary(
    frame: pd.DataFrame,
    true_column: str = "true_label",
    predicted_column: str = "predicted_label",
    participant_column: str | None = None,
    group_columns: Sequence[str] = (),
    metadata: pd.DataFrame | Sequence[dict[str, Any]] | None = None,
    category_columns: Sequence[str] | str | None = None,
) -> pd.DataFrame:
    """Summarize off-diagonal errors as unordered, bidirectional class pairs."""
    group_columns = _normalize_columns(group_columns)
    required_columns = [true_column, predicted_column, *group_columns]
    if participant_column is not None:
        required_columns.append(participant_column)
    _require_columns(frame, required_columns)

    working_columns = [*group_columns, true_column, predicted_column]
    if participant_column is not None:
        working_columns.append(participant_column)
    working = frame[working_columns].rename(columns={true_column: "true_label", predicted_column: "predicted_label"})

    metadata_by_label = _metadata_by_label(metadata)
    category_columns = _normalize_category_columns(metadata, category_columns)
    rows: list[dict[str, object]] = []
    for group_key, group_frame in _iter_frame_groups(working, group_columns):
        rows.extend(
            _confusion_pair_rows_for_group(
                group_frame,
                _group_row(group_columns, group_key),
                participant_column=participant_column,
                metadata_by_label=metadata_by_label,
                category_columns=category_columns,
            )
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(
        ["total_confusions", "mean_directional_rate", "label_a", "label_b"],
        ascending=[False, False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)


def category_confusion_enrichment(
    frame: pd.DataFrame,
    *,
    metadata: pd.DataFrame | Sequence[dict[str, Any]],
    category_columns: Sequence[str] | str | None = None,
    true_column: str = "true_label",
    predicted_column: str = "predicted_label",
    participant_column: str | None = None,
    group_columns: Sequence[str] = (),
    n_permutations: int = 10_000,
    random_state: int | None = 0,
) -> pd.DataFrame:
    """Test whether off-diagonal errors stay within class metadata categories."""
    group_columns = _normalize_columns(group_columns)
    required_columns = [true_column, predicted_column, *group_columns]
    if participant_column is not None:
        required_columns.append(participant_column)
    _require_columns(frame, required_columns)

    metadata_by_label = _metadata_by_label(metadata)
    category_columns = _normalize_category_columns(metadata, category_columns)
    if not metadata_by_label or not category_columns:
        return pd.DataFrame()

    working_columns = [*group_columns, true_column, predicted_column]
    if participant_column is not None:
        working_columns.append(participant_column)
    working = frame[working_columns].rename(columns={true_column: "true_label", predicted_column: "predicted_label"})
    rows: list[dict[str, object]] = []
    for group_key, group_frame in _iter_frame_groups(working, group_columns):
        rows.extend(
            _category_enrichment_rows_for_group(
                group_frame,
                _group_row(group_columns, group_key),
                metadata_by_label,
                category_columns,
                participant_column=participant_column,
                n_permutations=n_permutations,
                random_state=random_state,
            )
        )
    return pd.DataFrame(rows).reset_index(drop=True)


def category_confusion_matrix(
    frame: pd.DataFrame,
    *,
    metadata: pd.DataFrame | Sequence[dict[str, Any]],
    category_columns: Sequence[str] | str | None = None,
    true_column: str = "true_label",
    predicted_column: str = "predicted_label",
    participant_column: str | None = None,
    group_columns: Sequence[str] = (),
) -> pd.DataFrame:
    """Summarize directional category-to-category error counts and lifts."""
    group_columns = _normalize_columns(group_columns)
    required_columns = [true_column, predicted_column, *group_columns]
    if participant_column is not None:
        required_columns.append(participant_column)
    _require_columns(frame, required_columns)

    metadata_by_label = _metadata_by_label(metadata)
    category_columns = _normalize_category_columns(metadata, category_columns)
    if not metadata_by_label or not category_columns:
        return pd.DataFrame()

    working_columns = [*group_columns, true_column, predicted_column]
    if participant_column is not None:
        working_columns.append(participant_column)
    working = frame[working_columns].rename(columns={true_column: "true_label", predicted_column: "predicted_label"})
    rows: list[dict[str, object]] = []
    for group_key, group_frame in _iter_frame_groups(working, group_columns):
        rows.extend(
            _category_matrix_rows_for_group(
                group_frame,
                _group_row(group_columns, group_key),
                metadata_by_label,
                category_columns,
                participant_column=participant_column,
            )
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(
        ["category_confusion_lift", "count", "category_column", "true_category", "predicted_category"],
        ascending=[False, False, True, True, True],
        kind="mergesort",
    ).reset_index(drop=True)


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


def _iter_frame_groups(frame: pd.DataFrame, group_columns: Sequence[str]):
    if not group_columns:
        yield (), frame
        return
    for group_key, group_frame in frame.groupby(list(group_columns), dropna=False, sort=True):
        if len(group_columns) == 1 and not isinstance(group_key, tuple):
            group_key = (group_key,)
        yield tuple(group_key), group_frame


def _confusion_pair_rows_for_group(
    group_frame: pd.DataFrame,
    group_values: dict[str, object],
    *,
    participant_column: str | None,
    metadata_by_label: dict[object, dict[str, object]],
    category_columns: Sequence[str],
) -> list[dict[str, object]]:
    true_counts = group_frame["true_label"].value_counts(dropna=False).to_dict()
    error_frame = group_frame[group_frame["true_label"] != group_frame["predicted_label"]]
    if error_frame.empty:
        return []

    pair_counts: dict[tuple[object, object], Counter] = {}
    pair_participants: dict[tuple[object, object], set] = {}
    directional_participants: dict[tuple[object, object, object, object], set] = {}
    for row in error_frame.itertuples(index=False):
        true_label = getattr(row, "true_label")
        predicted_label = getattr(row, "predicted_label")
        label_a, label_b = _ordered_label_pair(true_label, predicted_label)
        pair = (label_a, label_b)
        pair_counts.setdefault(pair, Counter())
        pair_counts[pair][(true_label, predicted_label)] += 1
        if participant_column is not None:
            participant = getattr(row, participant_column)
            if participant not in (None, "") and not pd.isna(participant):
                pair_participants.setdefault(pair, set()).add(participant)
                directional_participants.setdefault((label_a, label_b, true_label, predicted_label), set()).add(participant)

    error_marginals = _confusion_error_marginals(error_frame)
    rows = []
    for (label_a, label_b), counts in pair_counts.items():
        rows.append(
            _confusion_pair_summary_row(
                group_values,
                label_a,
                label_b,
                counts,
                true_counts,
                error_marginals,
                pair_participants,
                directional_participants,
                metadata_by_label,
                category_columns,
            )
        )
    return rows


def _confusion_pair_summary_row(
    group_values: dict[str, object],
    label_a: object,
    label_b: object,
    counts: Counter,
    true_counts: dict[object, int],
    error_marginals: dict[str, object],
    pair_participants: dict[tuple[object, object], set],
    directional_participants: dict[tuple[object, object, object, object], set],
    metadata_by_label: dict[object, dict[str, object]],
    category_columns: Sequence[str],
) -> dict[str, object]:
    a_to_b_count = int(counts[(label_a, label_b)])
    b_to_a_count = int(counts[(label_b, label_a)])
    true_a_trials = int(true_counts.get(label_a, 0))
    true_b_trials = int(true_counts.get(label_b, 0))
    a_to_b_rate = _safe_rate(a_to_b_count, true_a_trials)
    b_to_a_rate = _safe_rate(b_to_a_count, true_b_trials)
    total_confusions = a_to_b_count + b_to_a_count
    result = {
        **group_values,
        "label_a": label_a,
        "label_b": label_b,
        "a_to_b_count": a_to_b_count,
        "b_to_a_count": b_to_a_count,
        "total_confusions": total_confusions,
        "true_a_trials": true_a_trials,
        "true_b_trials": true_b_trials,
        "a_to_b_rate": a_to_b_rate,
        "b_to_a_rate": b_to_a_rate,
        "mean_directional_rate": _nanmean_or_nan((a_to_b_rate, b_to_a_rate)),
        "max_directional_rate": _nanmax_or_nan((a_to_b_rate, b_to_a_rate)),
        "min_directional_rate": _nanmin_or_nan((a_to_b_rate, b_to_a_rate)),
        "absolute_rate_asymmetry": _absolute_difference_or_nan(a_to_b_rate, b_to_a_rate),
        "total_pair_error_rate": _safe_rate(total_confusions, true_a_trials + true_b_trials),
        **_confusion_pair_bias_metrics(label_a, label_b, a_to_b_count, b_to_a_count, error_marginals),
        "symmetric_confusion_count": min(a_to_b_count, b_to_a_count),
        "n_confused_participants": len(pair_participants.get((label_a, label_b), set())),
        "a_to_b_participants": len(directional_participants.get((label_a, label_b, label_a, label_b), set())),
        "b_to_a_participants": len(directional_participants.get((label_a, label_b, label_b, label_a), set())),
    }
    _add_label_metadata(result, label_a, label_b, metadata_by_label, category_columns)
    return result


def _confusion_error_marginals(error_frame: pd.DataFrame) -> dict[str, object]:
    return {
        "true_counts": error_frame["true_label"].value_counts(dropna=False).to_dict(),
        "predicted_counts": error_frame["predicted_label"].value_counts(dropna=False).to_dict(),
        "total": int(len(error_frame)),
    }


def _confusion_pair_bias_metrics(label_a: object, label_b: object, a_to_b_count: int, b_to_a_count: int, error_marginals: dict[str, object]) -> dict[str, float | int]:
    true_error_counts = error_marginals["true_counts"]
    predicted_error_counts = error_marginals["predicted_counts"]
    total_errors = error_marginals["total"]
    true_a_error_count = int(true_error_counts.get(label_a, 0))
    true_b_error_count = int(true_error_counts.get(label_b, 0))
    predicted_a_error_count = int(predicted_error_counts.get(label_a, 0))
    predicted_b_error_count = int(predicted_error_counts.get(label_b, 0))
    expected_a_to_b_count = _expected_confusion_count(true_a_error_count, predicted_b_error_count, total_errors)
    expected_b_to_a_count = _expected_confusion_count(true_b_error_count, predicted_a_error_count, total_errors)
    expected_total_confusions = expected_a_to_b_count + expected_b_to_a_count
    total_confusions = a_to_b_count + b_to_a_count
    return {
        "true_a_error_count": true_a_error_count,
        "true_b_error_count": true_b_error_count,
        "predicted_a_error_count": predicted_a_error_count,
        "predicted_b_error_count": predicted_b_error_count,
        "expected_a_to_b_count": expected_a_to_b_count,
        "expected_b_to_a_count": expected_b_to_a_count,
        "expected_total_confusions": expected_total_confusions,
        "a_to_b_lift": _safe_rate(a_to_b_count, expected_a_to_b_count),
        "b_to_a_lift": _safe_rate(b_to_a_count, expected_b_to_a_count),
        "pair_confusion_lift": _safe_rate(total_confusions, expected_total_confusions),
        "total_confusion_excess": _difference_or_nan(total_confusions, expected_total_confusions),
        "pair_standardized_residual": _standardized_residual(total_confusions, expected_total_confusions),
    }


def _category_enrichment_rows_for_group(
    group_frame: pd.DataFrame,
    group_values: dict[str, object],
    metadata_by_label: dict[object, dict[str, object]],
    category_columns: Sequence[str],
    *,
    participant_column: str | None,
    n_permutations: int,
    random_state: int | None,
) -> list[dict[str, object]]:
    error_frame = group_frame[group_frame["true_label"] != group_frame["predicted_label"]]
    if error_frame.empty:
        return []

    rows: list[dict[str, object]] = []
    for category_column in category_columns:
        category_errors = _category_error_rows(error_frame, metadata_by_label, category_column, participant_column=participant_column)
        if not category_errors:
            continue
        true_categories = [row["true_category"] for row in category_errors]
        predicted_categories = [row["predicted_category"] for row in category_errors]
        same_category = [true_category == predicted_category for true_category, predicted_category in zip(true_categories, predicted_categories)]
        observed = int(sum(same_category))
        total = int(len(category_errors))
        expected = _expected_same_category_count(true_categories, predicted_categories)
        same_participants = {
            row["participant"]
            for row, is_same in zip(category_errors, same_category)
            if is_same and row.get("participant") not in (None, "")
        }
        error_participants = {row["participant"] for row in category_errors if row.get("participant") not in (None, "")}
        rows.append(
            {
                **group_values,
                "category_column": category_column,
                "category_values": ";".join(sorted(set(true_categories) | set(predicted_categories))),
                "n_errors_with_category": total,
                "same_category_errors": observed,
                "expected_same_category_errors": expected,
                "same_category_error_rate": _safe_rate(observed, total),
                "expected_same_category_error_rate": _safe_rate(expected, total),
                "same_category_lift": _safe_rate(observed, expected),
                "same_category_excess": _difference_or_nan(observed, expected),
                "same_category_standardized_residual": _standardized_residual(observed, expected),
                "n_participants_with_category_errors": len(error_participants),
                "n_participants_with_same_category_errors": len(same_participants),
                "same_category_permutation_p_value": _same_category_permutation_p_value(
                    true_categories,
                    predicted_categories,
                    observed=observed,
                    n_permutations=n_permutations,
                    random_state=_category_random_state(random_state, group_values, category_column),
                ),
            }
        )
    return rows


def _category_matrix_rows_for_group(
    group_frame: pd.DataFrame,
    group_values: dict[str, object],
    metadata_by_label: dict[object, dict[str, object]],
    category_columns: Sequence[str],
    *,
    participant_column: str | None,
) -> list[dict[str, object]]:
    error_frame = group_frame[group_frame["true_label"] != group_frame["predicted_label"]]
    if error_frame.empty:
        return []

    rows: list[dict[str, object]] = []
    for category_column in category_columns:
        category_errors = _category_error_rows(error_frame, metadata_by_label, category_column, participant_column=participant_column)
        if not category_errors:
            continue
        total = int(len(category_errors))
        true_counts = Counter(row["true_category"] for row in category_errors)
        predicted_counts = Counter(row["predicted_category"] for row in category_errors)
        pair_counts = Counter((row["true_category"], row["predicted_category"]) for row in category_errors)
        pair_participants: dict[tuple[str, str], set] = {}
        for row in category_errors:
            participant = row.get("participant")
            if participant in (None, ""):
                continue
            pair_participants.setdefault((row["true_category"], row["predicted_category"]), set()).add(participant)

        for (true_category, predicted_category), count in pair_counts.items():
            expected = _expected_confusion_count(true_counts[true_category], predicted_counts[predicted_category], total)
            rows.append(
                {
                    **group_values,
                    "category_column": category_column,
                    "true_category": true_category,
                    "predicted_category": predicted_category,
                    "same_category": bool(true_category == predicted_category),
                    "count": int(count),
                    "expected_count": expected,
                    "rate": _safe_rate(count, total),
                    "expected_rate": _safe_rate(expected, total),
                    "category_confusion_lift": _safe_rate(count, expected),
                    "category_confusion_excess": _difference_or_nan(count, expected),
                    "category_standardized_residual": _standardized_residual(count, expected),
                    "true_category_error_count": int(true_counts[true_category]),
                    "predicted_category_error_count": int(predicted_counts[predicted_category]),
                    "n_errors_with_category": total,
                    "n_participants": len(pair_participants.get((true_category, predicted_category), set())),
                }
            )
    return rows


def _category_error_rows(
    error_frame: pd.DataFrame,
    metadata_by_label: dict[object, dict[str, object]],
    category_column: str,
    *,
    participant_column: str | None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in error_frame.itertuples(index=False):
        true_category = _metadata_category_value(metadata_by_label, getattr(row, "true_label"), category_column)
        predicted_category = _metadata_category_value(metadata_by_label, getattr(row, "predicted_label"), category_column)
        if true_category == "" or predicted_category == "":
            continue
        rows.append(
            {
                "true_category": true_category,
                "predicted_category": predicted_category,
                "participant": getattr(row, participant_column) if participant_column is not None else "",
            }
        )
    return rows


def _metadata_by_label(
    metadata: pd.DataFrame | Sequence[dict[str, Any]] | None,
    *,
    label_column: str | None = None,
) -> dict[object, dict[str, object]]:
    rows = _metadata_rows(metadata)
    if not rows:
        return {}
    metadata_by_label: dict[object, dict[str, object]] = {}
    for row in rows:
        label = _metadata_label_id(row, label_column=label_column)
        if label is not None:
            metadata_by_label[label] = dict(row)
    return metadata_by_label


def _metadata_rows(metadata: pd.DataFrame | Sequence[dict[str, Any]] | None) -> list[dict[str, object]]:
    if metadata is None:
        return []
    if isinstance(metadata, pd.DataFrame):
        return metadata.to_dict(orient="records")
    return [dict(row) for row in metadata]


def _metadata_label_id(row: dict[str, object], *, label_column: str | None = None) -> object | None:
    columns = [label_column] if label_column is not None else ["label", "class", "stimulus", "stimulus_id", "id"]
    for column in columns:
        value = row.get(column)
        if value not in (None, ""):
            return value
    return None


def _normalize_category_columns(metadata: pd.DataFrame | Sequence[dict[str, Any]] | None, category_columns: Sequence[str] | str | None) -> tuple[str, ...]:
    if category_columns is None:
        return _infer_category_columns(metadata)
    if isinstance(category_columns, str):
        category_columns = [column.strip() for column in category_columns.split(",") if column.strip()]
    return tuple(str(column).strip() for column in category_columns if str(column).strip())


def _infer_category_columns(metadata: pd.DataFrame | Sequence[dict[str, Any]] | None) -> tuple[str, ...]:
    rows = _metadata_rows(metadata)
    if not rows:
        return tuple()
    excluded = {"label", "class", "stimulus", "stimulus_id", "id", "name", "stimulus_name", "image", "image_name", "filename", "file", "path"}
    columns = sorted({column for row in rows for column in row if column.lower() not in excluded})
    inferred: list[str] = []
    for column in columns:
        values = [str(row.get(column, "")).strip() for row in rows]
        values = [value for value in values if value]
        if len(set(values)) < len(values):
            inferred.append(column)
    return tuple(inferred)


def _lookup_metadata(metadata_by_label: dict[object, dict[str, object]], label: object) -> dict[str, object]:
    if label in metadata_by_label:
        return metadata_by_label[label]
    label_text = str(label)
    if label_text in metadata_by_label:
        return metadata_by_label[label_text]
    try:
        label_int = int(label)
    except (TypeError, ValueError):
        return {}
    return metadata_by_label.get(label_int, {})


def _metadata_category_value(metadata_by_label: dict[object, dict[str, object]], label: object, category_column: str) -> str:
    value = _lookup_metadata(metadata_by_label, label).get(category_column, "")
    if value in (None, ""):
        return ""
    return str(value).strip()


def _add_label_metadata(
    row: dict[str, object],
    label_a: object,
    label_b: object,
    metadata_by_label: dict[object, dict[str, object]],
    category_columns: Sequence[str],
) -> None:
    metadata_a = _lookup_metadata(metadata_by_label, label_a)
    metadata_b = _lookup_metadata(metadata_by_label, label_b)
    if not metadata_a and not metadata_b:
        return
    metadata_keys = sorted((set(metadata_a) | set(metadata_b)) - {"label", "class", "stimulus", "stimulus_id", "id"})
    for key in metadata_keys:
        value_a = metadata_a.get(key, "")
        value_b = metadata_b.get(key, "")
        row[f"label_a_{key}"] = value_a
        row[f"label_b_{key}"] = value_b
        if key in category_columns and value_a != "" and value_b != "":
            row[f"same_{key}"] = bool(value_a == value_b)


def _ordered_label_pair(first: object, second: object) -> tuple[object, object]:
    return tuple(sorted((first, second), key=_label_sort_key))


def _label_sort_key(value: object) -> tuple[int, object]:
    if pd.isna(value):
        return (2, "")
    try:
        return (0, int(value))
    except (TypeError, ValueError):
        return (1, str(value))


def _expected_same_category_count(true_categories: Sequence[str], predicted_categories: Sequence[str]) -> float:
    total = len(true_categories)
    if total == 0:
        return np.nan
    true_counts = Counter(true_categories)
    predicted_counts = Counter(predicted_categories)
    categories = set(true_counts) | set(predicted_counts)
    return float(sum(true_counts[category] * predicted_counts[category] for category in categories) / total)


def _same_category_permutation_p_value(
    true_categories: Sequence[str],
    predicted_categories: Sequence[str],
    *,
    observed: int,
    n_permutations: int,
    random_state: int | np.random.SeedSequence | None,
) -> float:
    if n_permutations is None or int(n_permutations) <= 0:
        return np.nan
    true_categories = np.asarray(true_categories, dtype=object)
    predicted_categories = np.asarray(predicted_categories, dtype=object)
    if true_categories.size == 0 or predicted_categories.size == 0:
        return np.nan
    rng = np.random.default_rng(random_state)
    exceedances = 0
    for _ in range(int(n_permutations)):
        shuffled = rng.permutation(predicted_categories)
        exceedances += int(np.sum(true_categories == shuffled) >= observed)
    return float((exceedances + 1.0) / (int(n_permutations) + 1.0))


def _category_random_state(random_state: int | None, group_values: dict[str, object], category_column: str) -> np.random.SeedSequence:
    seed_values = [0 if random_state is None else int(random_state), sum(ord(character) for character in str(category_column))]
    for key, value in sorted(group_values.items()):
        seed_values.append(sum(ord(character) for character in f"{key}={value}"))
    return np.random.SeedSequence(seed_values)


def _safe_rate(numerator: float, denominator: float) -> float:
    denominator = float(denominator)
    if denominator <= 0.0 or not np.isfinite(denominator):
        return np.nan
    return float(numerator) / denominator


def _expected_confusion_count(true_error_count: int, predicted_error_count: int, total_errors: int) -> float:
    total_errors = float(total_errors)
    if total_errors <= 0.0:
        return np.nan
    return float(true_error_count) * float(predicted_error_count) / total_errors


def _difference_or_nan(first: float, second: float) -> float:
    if not np.isfinite(first) or not np.isfinite(second):
        return np.nan
    return float(first - second)


def _standardized_residual(observed: float, expected: float) -> float:
    if not np.isfinite(expected) or expected <= 0.0:
        return np.nan
    return float(observed - expected) / float(np.sqrt(expected))


def _absolute_difference_or_nan(first: float, second: float) -> float:
    if not np.isfinite(first) or not np.isfinite(second):
        return np.nan
    return float(abs(first - second))


def _nanmean_or_nan(values: Sequence[float]) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan
    return float(np.mean(values))


def _nanmax_or_nan(values: Sequence[float]) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan
    return float(np.max(values))


def _nanmin_or_nan(values: Sequence[float]) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan
    return float(np.min(values))
