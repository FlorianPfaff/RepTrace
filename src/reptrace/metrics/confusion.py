from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

import numpy as np
import pandas as pd

DEFAULT_METADATA_LABEL_COLUMNS = ("label", "label_id", "class", "class_id", "true_label", "stimulus", "stimulus_id", "image_id")
_TRUE_LABEL = "__reptrace_true_label"
_PREDICTED_LABEL = "__reptrace_predicted_label"
_PARTICIPANT = "__reptrace_participant"


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


def confusion_pair_summary(
    frame: pd.DataFrame,
    true_column: str = "true_label",
    predicted_column: str = "predicted_label",
    group_columns: Sequence[str] = (),
    participant_column: str | None = None,
    metadata_frame: pd.DataFrame | None = None,
    metadata_label_columns: Sequence[str] = DEFAULT_METADATA_LABEL_COLUMNS,
    label_prefix: str = "label",
) -> pd.DataFrame:
    """Summarize off-diagonal errors as unordered, bidirectional label pairs.

    Expected counts preserve the true-label and predicted-label error marginals.
    Metadata columns, when supplied, are copied for both labels and get an
    additional ``same_<metadata_column>`` flag when both sides are known.
    """
    group_columns = _normalize_columns(group_columns)
    working = _prediction_frame(
        frame,
        true_column=true_column,
        predicted_column=predicted_column,
        group_columns=group_columns,
        participant_column=participant_column,
    )
    metadata_by_label = _metadata_by_label(metadata_frame, metadata_label_columns)

    rows: list[dict[str, object]] = []
    for group_key, group_frame in _iter_frame_groups(working, group_columns):
        rows.extend(
            _summarize_confusion_pairs_for_group(
                group_frame,
                _group_row(group_columns, group_key),
                metadata_by_label,
                metadata_label_columns,
                label_prefix,
            )
        )

    label_a_column = f"{label_prefix}_a"
    label_b_column = f"{label_prefix}_b"
    sorted_rows = sorted(
        rows,
        key=lambda row: (
            -int(row["total_confusions"]),
            -float(row["mean_directional_rate"]) if np.isfinite(float(row["mean_directional_rate"])) else np.inf,
            _label_sort_key(row[label_a_column]),
            _label_sort_key(row[label_b_column]),
        ),
    )
    return pd.DataFrame(sorted_rows).reset_index(drop=True)


def confusion_category_enrichment(
    frame: pd.DataFrame,
    *,
    metadata_frame: pd.DataFrame,
    true_column: str = "true_label",
    predicted_column: str = "predicted_label",
    category_columns: Sequence[str] | str | None = None,
    group_columns: Sequence[str] = (),
    participant_column: str | None = None,
    metadata_label_columns: Sequence[str] = DEFAULT_METADATA_LABEL_COLUMNS,
    n_permutations: int | None = 10_000,
    seed: int | None = 0,
) -> pd.DataFrame:
    """Test whether off-diagonal errors stay within label metadata categories."""
    group_columns = _normalize_columns(group_columns)
    working = _prediction_frame(
        frame,
        true_column=true_column,
        predicted_column=predicted_column,
        group_columns=group_columns,
        participant_column=participant_column,
    )
    metadata_by_label = _metadata_by_label(metadata_frame, metadata_label_columns)
    category_columns = _normalize_category_columns(metadata_frame, category_columns, metadata_label_columns)
    if not metadata_by_label or not category_columns:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    for group_key, group_frame in _iter_frame_groups(working, group_columns):
        rows.extend(
            _summarize_category_enrichment_for_group(
                group_frame,
                _group_row(group_columns, group_key),
                metadata_by_label,
                category_columns,
                n_permutations=n_permutations,
                seed=seed,
            )
        )
    return pd.DataFrame(rows).reset_index(drop=True)


def confusion_category_matrix(
    frame: pd.DataFrame,
    *,
    metadata_frame: pd.DataFrame,
    true_column: str = "true_label",
    predicted_column: str = "predicted_label",
    category_columns: Sequence[str] | str | None = None,
    group_columns: Sequence[str] = (),
    participant_column: str | None = None,
    metadata_label_columns: Sequence[str] = DEFAULT_METADATA_LABEL_COLUMNS,
) -> pd.DataFrame:
    """Summarize directional category-to-category error counts and lifts."""
    group_columns = _normalize_columns(group_columns)
    working = _prediction_frame(
        frame,
        true_column=true_column,
        predicted_column=predicted_column,
        group_columns=group_columns,
        participant_column=participant_column,
    )
    metadata_by_label = _metadata_by_label(metadata_frame, metadata_label_columns)
    category_columns = _normalize_category_columns(metadata_frame, category_columns, metadata_label_columns)
    if not metadata_by_label or not category_columns:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    for group_key, group_frame in _iter_frame_groups(working, group_columns):
        rows.extend(
            _summarize_category_matrix_for_group(
                group_frame,
                _group_row(group_columns, group_key),
                metadata_by_label,
                category_columns,
            )
        )

    sorted_rows = sorted(
        rows,
        key=lambda row: (
            -float(row["category_confusion_lift"]) if np.isfinite(float(row["category_confusion_lift"])) else np.inf,
            -int(row["count"]),
            str(row["category_column"]),
            str(row["true_category"]),
            str(row["predicted_category"]),
        ),
    )
    return pd.DataFrame(sorted_rows).reset_index(drop=True)


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


def _prediction_frame(
    frame: pd.DataFrame,
    *,
    true_column: str,
    predicted_column: str,
    group_columns: Sequence[str],
    participant_column: str | None,
) -> pd.DataFrame:
    required_columns = [true_column, predicted_column, *group_columns]
    if participant_column is not None:
        required_columns.append(participant_column)
    _require_columns(frame, required_columns)

    columns = list(dict.fromkeys(required_columns))
    rename_columns = {true_column: _TRUE_LABEL, predicted_column: _PREDICTED_LABEL}
    if participant_column is not None:
        rename_columns[participant_column] = _PARTICIPANT
    return frame[columns].rename(columns=rename_columns).copy()


def _iter_frame_groups(frame: pd.DataFrame, group_columns: Sequence[str]):
    if not group_columns:
        yield (), frame
        return
    for group_key, group_frame in frame.groupby(list(group_columns), dropna=False, sort=True):
        if len(group_columns) == 1:
            group_key = (group_key,)
        yield tuple(group_key), group_frame


def _summarize_confusion_pairs_for_group(
    group_frame: pd.DataFrame,
    group_values: dict[str, object],
    metadata_by_label: dict[object, dict[str, object]],
    metadata_label_columns: Sequence[str],
    label_prefix: str,
) -> list[dict[str, object]]:
    true_counts = group_frame[_TRUE_LABEL].value_counts(dropna=False).to_dict()
    error_frame = group_frame[group_frame[_TRUE_LABEL] != group_frame[_PREDICTED_LABEL]]
    if error_frame.empty:
        return []

    pair_counts, pair_participants, directional_participants = _confusion_pair_maps(error_frame)
    error_marginals = _confusion_error_marginals(error_frame)
    return [
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
            metadata_label_columns,
            label_prefix,
        )
        for (label_a, label_b), counts in pair_counts.items()
    ]


def _confusion_pair_maps(error_frame: pd.DataFrame):
    pair_counts: dict[tuple[object, object], Counter] = {}
    pair_participants: dict[tuple[object, object], set] = {}
    directional_participants: dict[tuple[object, object, object, object], set] = {}
    for _, row in error_frame.iterrows():
        true_label = row[_TRUE_LABEL]
        predicted_label = row[_PREDICTED_LABEL]
        label_a, label_b = _ordered_label_pair(true_label, predicted_label)
        pair_counts.setdefault((label_a, label_b), Counter())
        pair_participants.setdefault((label_a, label_b), set())
        directional_participants.setdefault((label_a, label_b, true_label, predicted_label), set())
        pair_counts[(label_a, label_b)][(true_label, predicted_label)] += 1
        if _PARTICIPANT in row and not _is_blank(row[_PARTICIPANT]):
            participant = row[_PARTICIPANT]
            pair_participants[(label_a, label_b)].add(participant)
            directional_participants[(label_a, label_b, true_label, predicted_label)].add(participant)
    return pair_counts, pair_participants, directional_participants


def _confusion_error_marginals(error_frame: pd.DataFrame) -> dict[str, object]:
    return {
        "true_counts": error_frame[_TRUE_LABEL].value_counts(dropna=False).to_dict(),
        "predicted_counts": error_frame[_PREDICTED_LABEL].value_counts(dropna=False).to_dict(),
        "total": int(len(error_frame)),
    }


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
    metadata_label_columns: Sequence[str],
    label_prefix: str,
) -> dict[str, object]:
    a_to_b_count = int(counts[(label_a, label_b)])
    b_to_a_count = int(counts[(label_b, label_a)])
    true_a_trials = int(true_counts.get(label_a, 0))
    true_b_trials = int(true_counts.get(label_b, 0))
    a_to_b_rate = _safe_rate(a_to_b_count, true_a_trials)
    b_to_a_rate = _safe_rate(b_to_a_count, true_b_trials)
    total_confusions = a_to_b_count + b_to_a_count
    bias_metrics = _confusion_pair_bias_metrics(label_a, label_b, a_to_b_count, b_to_a_count, error_marginals)
    result: dict[str, object] = {
        **group_values,
        f"{label_prefix}_a": label_a,
        f"{label_prefix}_b": label_b,
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
        **bias_metrics,
        "symmetric_confusion_count": min(a_to_b_count, b_to_a_count),
        "n_confused_participants": len(pair_participants.get((label_a, label_b), set())),
        "a_to_b_participants": len(directional_participants.get((label_a, label_b, label_a, label_b), set())),
        "b_to_a_participants": len(directional_participants.get((label_a, label_b, label_b, label_a), set())),
    }
    _add_label_metadata(result, label_a, label_b, metadata_by_label, metadata_label_columns, label_prefix)
    return result


def _confusion_pair_bias_metrics(label_a: object, label_b: object, a_to_b_count: int, b_to_a_count: int, error_marginals: dict[str, object]) -> dict[str, object]:
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


def _summarize_category_enrichment_for_group(
    group_frame: pd.DataFrame,
    group_values: dict[str, object],
    metadata_by_label: dict[object, dict[str, object]],
    category_columns: Sequence[str],
    *,
    n_permutations: int | None,
    seed: int | None,
) -> list[dict[str, object]]:
    error_frame = group_frame[group_frame[_TRUE_LABEL] != group_frame[_PREDICTED_LABEL]]
    if error_frame.empty:
        return []

    rows: list[dict[str, object]] = []
    for category_column in category_columns:
        category_errors = _category_error_rows(error_frame, metadata_by_label, category_column)
        if not category_errors:
            continue
        true_categories = [row["true_category"] for row in category_errors]
        predicted_categories = [row["predicted_category"] for row in category_errors]
        same_category = [true_category == predicted_category for true_category, predicted_category in zip(true_categories, predicted_categories, strict=True)]
        observed = int(sum(same_category))
        total = int(len(category_errors))
        expected = _expected_same_category_count(true_categories, predicted_categories)
        same_participants = {row["participant"] for row, is_same in zip(category_errors, same_category, strict=True) if is_same and not _is_blank(row.get("participant"))}
        error_participants = {row["participant"] for row in category_errors if not _is_blank(row.get("participant"))}
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
                    seed=_category_seed(seed, group_values, category_column),
                ),
            }
        )
    return rows


def _summarize_category_matrix_for_group(
    group_frame: pd.DataFrame,
    group_values: dict[str, object],
    metadata_by_label: dict[object, dict[str, object]],
    category_columns: Sequence[str],
) -> list[dict[str, object]]:
    error_frame = group_frame[group_frame[_TRUE_LABEL] != group_frame[_PREDICTED_LABEL]]
    if error_frame.empty:
        return []

    rows: list[dict[str, object]] = []
    for category_column in category_columns:
        category_errors = _category_error_rows(error_frame, metadata_by_label, category_column)
        if not category_errors:
            continue
        total = int(len(category_errors))
        true_counts = Counter(row["true_category"] for row in category_errors)
        predicted_counts = Counter(row["predicted_category"] for row in category_errors)
        category_pair_counts = Counter((row["true_category"], row["predicted_category"]) for row in category_errors)
        category_pair_participants: dict[tuple[str, str], set] = {}
        for row in category_errors:
            participant = row.get("participant")
            if _is_blank(participant):
                continue
            category_pair_participants.setdefault((row["true_category"], row["predicted_category"]), set()).add(participant)

        for (true_category, predicted_category), count in category_pair_counts.items():
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
                    "n_participants": len(category_pair_participants.get((true_category, predicted_category), set())),
                }
            )
    return rows


def _category_error_rows(error_frame: pd.DataFrame, metadata_by_label: dict[object, dict[str, object]], category_column: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for _, row in error_frame.iterrows():
        true_category = _metadata_category_value(metadata_by_label, row[_TRUE_LABEL], category_column)
        predicted_category = _metadata_category_value(metadata_by_label, row[_PREDICTED_LABEL], category_column)
        if true_category == "" or predicted_category == "":
            continue
        rows.append(
            {
                "true_category": true_category,
                "predicted_category": predicted_category,
                "participant": row.get(_PARTICIPANT, ""),
            }
        )
    return rows


def _expected_same_category_count(true_categories: Sequence[str], predicted_categories: Sequence[str]) -> float:
    total = len(true_categories)
    if total == 0:
        return np.nan
    true_counts = Counter(true_categories)
    predicted_counts = Counter(predicted_categories)
    return float(sum(true_counts[category] * predicted_counts[category] for category in set(true_counts) | set(predicted_counts)) / total)


def _same_category_permutation_p_value(true_categories: Sequence[str], predicted_categories: Sequence[str], *, observed: int, n_permutations: int | None, seed) -> float:
    if n_permutations is None or int(n_permutations) <= 0:
        return np.nan
    true_categories_array = np.asarray(true_categories, dtype=object)
    predicted_categories_array = np.asarray(predicted_categories, dtype=object)
    if true_categories_array.size == 0 or predicted_categories_array.size == 0:
        return np.nan
    rng = np.random.default_rng(seed)
    exceedances = 0
    for _ in range(int(n_permutations)):
        shuffled = rng.permutation(predicted_categories_array)
        exceedances += int(np.sum(true_categories_array == shuffled) >= observed)
    return float((exceedances + 1) / (int(n_permutations) + 1))


def _category_seed(seed: int | None, group_values: dict[str, object], category_column: str):
    seed_values = [0 if seed is None else int(seed), sum(ord(character) for character in str(category_column))]
    for key, value in sorted(group_values.items()):
        seed_values.append(sum(ord(character) for character in f"{key}={value}"))
    return np.random.SeedSequence(seed_values)


def _ordered_label_pair(first: object, second: object) -> tuple[object, object]:
    return tuple(sorted((first, second), key=_label_sort_key))


def _label_sort_key(value: object) -> tuple[int, float | str, str]:
    try:
        return 0, float(value), str(value)
    except (TypeError, ValueError):
        return 1, str(value), str(value)


def _metadata_by_label(metadata_frame: pd.DataFrame | None, metadata_label_columns: Sequence[str]) -> dict[object, dict[str, object]]:
    metadata_by_label: dict[object, dict[str, object]] = {}
    if metadata_frame is None or metadata_frame.empty:
        return metadata_by_label
    label_columns = _normalize_columns(metadata_label_columns)
    for _, metadata_row in metadata_frame.iterrows():
        label = _metadata_label_id(metadata_row, label_columns)
        if label is not None:
            metadata_by_label[label] = metadata_row.to_dict()
    return metadata_by_label


def _metadata_label_id(metadata_row: pd.Series, label_columns: Sequence[str]) -> object | None:
    for column in label_columns:
        if column not in metadata_row:
            continue
        value = metadata_row.get(column)
        if not _is_blank(value):
            return value
    return None


def _normalize_category_columns(metadata_frame: pd.DataFrame, category_columns: Sequence[str] | str | None, metadata_label_columns: Sequence[str]) -> tuple[str, ...]:
    if category_columns is None:
        return _infer_category_columns(metadata_frame, metadata_label_columns)
    if isinstance(category_columns, str):
        category_columns = [column.strip() for column in category_columns.split(",") if column.strip()]
    return tuple(str(column).strip() for column in category_columns if str(column).strip())


def _infer_category_columns(metadata_frame: pd.DataFrame, metadata_label_columns: Sequence[str]) -> tuple[str, ...]:
    if metadata_frame.empty:
        return tuple()
    excluded = {column.lower() for column in metadata_label_columns}
    excluded.update({"name", "label_name", "class_name", "stimulus_name", "image", "image_name", "filename", "file", "path"})
    inferred = []
    for column in sorted(metadata_frame.columns):
        if column.lower() in excluded:
            continue
        values = [str(value).strip() for value in metadata_frame[column].tolist() if not _is_blank(value)]
        if len(set(values)) < len(values):
            inferred.append(str(column))
    return tuple(inferred)


def _metadata_category_value(metadata_by_label: dict[object, dict[str, object]], label: object, category_column: str) -> str:
    metadata = _lookup_label_metadata(metadata_by_label, label)
    value = metadata.get(category_column, "")
    if _is_blank(value):
        return ""
    return str(value).strip()


def _add_label_metadata(
    row: dict[str, object],
    label_a: object,
    label_b: object,
    metadata_by_label: dict[object, dict[str, object]],
    metadata_label_columns: Sequence[str],
    label_prefix: str,
) -> None:
    metadata_a = _lookup_label_metadata(metadata_by_label, label_a)
    metadata_b = _lookup_label_metadata(metadata_by_label, label_b)
    if not metadata_a and not metadata_b:
        return

    metadata_keys = sorted((set(metadata_a) | set(metadata_b)) - set(metadata_label_columns))
    for key in metadata_keys:
        value_a = _clean_metadata_value(metadata_a.get(key, ""))
        value_b = _clean_metadata_value(metadata_b.get(key, ""))
        row[f"{label_prefix}_a_{key}"] = value_a
        row[f"{label_prefix}_b_{key}"] = value_b
        if value_a != "" and value_b != "":
            row[f"same_{key}"] = bool(value_a == value_b)


def _lookup_label_metadata(metadata_by_label: dict[object, dict[str, object]], label: object) -> dict[str, object]:
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


def _clean_metadata_value(value: object) -> object:
    if _is_blank(value):
        return ""
    return value


def _is_blank(value: object) -> bool:
    if value is None:
        return True
    try:
        if bool(pd.isna(value)):
            return True
    except (TypeError, ValueError):
        pass
    return isinstance(value, str) and value.strip() == ""


def _nanmean_or_nan(values: Sequence[float]) -> float:
    finite_values = np.asarray(values, dtype=float)
    finite_values = finite_values[np.isfinite(finite_values)]
    if finite_values.size == 0:
        return np.nan
    return float(np.mean(finite_values))


def _nanmax_or_nan(values: Sequence[float]) -> float:
    finite_values = np.asarray(values, dtype=float)
    finite_values = finite_values[np.isfinite(finite_values)]
    if finite_values.size == 0:
        return np.nan
    return float(np.max(finite_values))


def _nanmin_or_nan(values: Sequence[float]) -> float:
    finite_values = np.asarray(values, dtype=float)
    finite_values = finite_values[np.isfinite(finite_values)]
    if finite_values.size == 0:
        return np.nan
    return float(np.min(finite_values))


def _safe_rate(numerator: float, denominator: float) -> float:
    denominator = float(denominator)
    if denominator <= 0:
        return np.nan
    return float(numerator) / denominator


def _expected_confusion_count(true_error_count: int, predicted_error_count: int, total_errors: int) -> float:
    total_errors = float(total_errors)
    if total_errors <= 0:
        return np.nan
    return float(true_error_count) * float(predicted_error_count) / total_errors


def _difference_or_nan(first: float, second: float) -> float:
    if not np.isfinite(first) or not np.isfinite(second):
        return np.nan
    return float(first - second)


def _standardized_residual(observed: float, expected: float) -> float:
    if not np.isfinite(expected) or expected <= 0:
        return np.nan
    return float(observed - expected) / float(np.sqrt(expected))


def _absolute_difference_or_nan(first: float, second: float) -> float:
    if not np.isfinite(first) or not np.isfinite(second):
        return np.nan
    return float(abs(first - second))
