from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from reptrace.metrics import expected_calibration_error
from reptrace.observations import probability_columns
from reptrace.results.tables import peak_metric_rows, summarize_metric_table

__all__ = [
    "METRIC_COLUMNS",
    "SUMMARY_GROUP_COLUMNS",
    "WEIGHT_COLUMN",
    "aggregate_time_decode_csvs",
    "aggregate_time_decode_results",
    "mean_across_folds",
    "peak_metric_rows",
    "read_time_decode_observations",
    "read_time_decode_results",
    "summarize_metric_table",
]

METRIC_COLUMNS = ("accuracy", "log_loss", "brier", "ece")
SUMMARY_GROUP_COLUMNS = ("decoder", "emission_mode")
WEIGHT_COLUMN = "n_test"


def _assign_subject(frame: pd.DataFrame, csv_path: Path, subject_column: str | None) -> None:
    if subject_column is not None:
        if subject_column not in frame.columns:
            raise ValueError(f"{csv_path} is missing subject column '{subject_column}'.")
        frame["subject"] = frame[subject_column].astype(str)
    elif "subject" not in frame.columns:
        frame["subject"] = csv_path.stem
    else:
        frame["subject"] = frame["subject"].astype(str)


def read_time_decode_results(csv_paths: list[Path], *, subject_column: str | None = None) -> pd.DataFrame:
    if not csv_paths:
        raise ValueError("At least one CSV path is required.")
    frames = []
    for csv_path in csv_paths:
        frame = pd.read_csv(csv_path)
        missing = [column for column in ("time", *METRIC_COLUMNS) if column not in frame.columns]
        if missing:
            raise ValueError(f"{csv_path} is missing required columns: {missing}")
        _assign_subject(frame, csv_path, subject_column)
        if "emission_mode" not in frame.columns:
            frame["emission_mode"] = "calibrated"
        frame["source_file"] = csv_path.name
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def read_time_decode_observations(csv_paths: list[Path], *, subject_column: str | None = None) -> pd.DataFrame:
    if not csv_paths:
        raise ValueError("At least one observation CSV path is required.")
    frames = []
    for csv_path in csv_paths:
        frame = pd.read_csv(csv_path)
        missing = [column for column in ("time", "true_label") if column not in frame.columns]
        if missing:
            raise ValueError(f"{csv_path} is missing required observation columns: {missing}")
        if not probability_columns(frame):
            raise ValueError(f"{csv_path} does not contain any prob_class_* columns.")
        _assign_subject(frame, csv_path, subject_column)
        if "emission_mode" not in frame.columns:
            frame["emission_mode"] = "calibrated"
        frame["source_file"] = csv_path.name
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def mean_across_folds(results: pd.DataFrame, group_columns: Sequence[str], *, metric_columns: Sequence[str] = METRIC_COLUMNS) -> pd.DataFrame:
    group_columns = list(group_columns)
    metric_columns = list(metric_columns)
    missing = [column for column in (*group_columns, *metric_columns) if column not in results.columns]
    if missing:
        raise ValueError(f"Results are missing required columns: {missing}")
    if WEIGHT_COLUMN not in results.columns:
        return results.groupby(group_columns, as_index=False)[metric_columns].mean()
    weighted = results.copy()
    weights = pd.to_numeric(weighted[WEIGHT_COLUMN], errors="coerce")
    if weights.isna().any() or not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError(f"Column '{WEIGHT_COLUMN}' must contain positive finite fold sizes.")
    weighted["__fold_weight"] = weights.astype(float)
    weighted_columns = []
    denominator_columns = []
    for metric in metric_columns:
        values = pd.to_numeric(weighted[metric], errors="coerce")
        weighted_column = f"__weighted_{metric}"
        denominator_column = f"__weight_{metric}"
        weighted[weighted_column] = values * weighted["__fold_weight"]
        weighted[denominator_column] = weighted["__fold_weight"].where(values.notna())
        weighted_columns.append(weighted_column)
        denominator_columns.append(denominator_column)
    grouped = weighted.groupby(group_columns, as_index=False)[[*weighted_columns, *denominator_columns]].sum()
    for metric, weighted_column, denominator_column in zip(metric_columns, weighted_columns, denominator_columns, strict=True):
        grouped[metric] = grouped[weighted_column] / grouped[denominator_column]
    return grouped[[*group_columns, *metric_columns]]


_mean_across_folds = mean_across_folds


def _as_key_row(columns: Sequence[str], key: object) -> dict[str, object]:
    if len(columns) == 1 and not isinstance(key, tuple):
        key = (key,)
    return dict(zip(columns, key))


def _probabilities_and_labels(group: pd.DataFrame, prob_columns: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
    probabilities = group.loc[:, prob_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(probabilities).all():
        raise ValueError("Observation probability columns must contain finite numeric values.")
    label_values = pd.to_numeric(group["true_label"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(label_values).all():
        raise ValueError("Observation column 'true_label' must contain finite integer labels.")
    labels = np.rint(label_values).astype(int)
    if not np.allclose(label_values, labels):
        raise ValueError("Observation column 'true_label' must contain integer labels.")
    if (labels < 0).any() or (labels >= probabilities.shape[1]).any():
        raise ValueError("Observation labels must be valid class indices for the probability columns.")
    return probabilities, labels


def _observation_ece(observations: pd.DataFrame, group_columns: Sequence[str], *, ece_bins: int) -> pd.DataFrame:
    required_columns = [*group_columns, "subject", "time", "true_label"]
    missing = [column for column in required_columns if column not in observations.columns]
    if missing:
        raise ValueError(f"Observation results are missing required columns for exact ECE: {missing}")
    prob_columns = probability_columns(observations)
    if not prob_columns:
        raise ValueError("Observation results do not contain any prob_class_* columns.")
    working = observations.copy()
    if "emission_mode" in working.columns:
        working["emission_mode"] = working["emission_mode"].fillna("calibrated")
    keys = [*group_columns, "subject", "time"]
    rows = []
    for group_key, group in working.groupby(keys, dropna=False, sort=True):
        probabilities, labels = _probabilities_and_labels(group, prob_columns)
        rows.append({**_as_key_row(keys, group_key), "ece": expected_calibration_error(probabilities, labels, n_bins=ece_bins)})
    return pd.DataFrame(rows)


def _replace_ece_with_observation_ece(subject_time: pd.DataFrame, observations: pd.DataFrame, group_columns: Sequence[str], *, ece_bins: int) -> pd.DataFrame:
    subject_time_keys = [*group_columns, "subject", "time"]
    exact = _observation_ece(observations, group_columns, ece_bins=ece_bins).rename(columns={"ece": "__observation_ece"})
    merged = subject_time.drop(columns=["ece"]).merge(exact, on=subject_time_keys, how="left", validate="one_to_one")
    missing_mask = merged["__observation_ece"].isna()
    if bool(missing_mask.any()):
        missing_keys = merged.loc[missing_mask, subject_time_keys].drop_duplicates().head(5).to_dict("records")
        raise ValueError(f"Observation CSVs do not cover all result subject/time cells needed for exact ECE. Missing examples: {missing_keys}")
    return merged.rename(columns={"__observation_ece": "ece"})


def aggregate_time_decode_results(results: pd.DataFrame, *, observations: pd.DataFrame | None = None, ece_bins: int = 10) -> pd.DataFrame:
    missing = [column for column in ("subject", "time", *METRIC_COLUMNS) if column not in results.columns]
    if missing:
        raise ValueError(f"Results are missing required columns: {missing}")
    if ece_bins < 1:
        raise ValueError("ece_bins must be positive.")
    if "emission_mode" in results.columns:
        results = results.copy()
        results["emission_mode"] = results["emission_mode"].fillna("calibrated")
    group_columns = [column for column in SUMMARY_GROUP_COLUMNS if column in results.columns]
    subject_time_keys = [*group_columns, "subject", "time"]
    aggregate_keys = [*group_columns, "time"]
    subject_time = mean_across_folds(results, subject_time_keys).sort_values(subject_time_keys)
    if observations is not None:
        subject_time = _replace_ece_with_observation_ece(subject_time, observations, group_columns, ece_bins=ece_bins).sort_values(subject_time_keys)
    grouped = subject_time.groupby(aggregate_keys, as_index=False)
    aggregated = grouped[list(METRIC_COLUMNS)].mean()
    n_subjects = grouped["subject"].nunique().rename(columns={"subject": "n_subjects"})
    aggregated = aggregated.merge(n_subjects, on=aggregate_keys)
    for metric in METRIC_COLUMNS:
        sem = grouped[metric].sem().rename(columns={metric: f"{metric}_sem"})
        aggregated = aggregated.merge(sem, on=aggregate_keys).rename(columns={metric: f"{metric}_mean"})
    return aggregated.sort_values(aggregate_keys).reset_index(drop=True)


def aggregate_time_decode_csvs(csv_paths: list[Path], out_path: Path, *, subject_column: str | None = None, observation_paths: list[Path] | None = None, ece_bins: int = 10) -> pd.DataFrame:
    results = read_time_decode_results(csv_paths, subject_column=subject_column)
    observations = read_time_decode_observations(observation_paths, subject_column=subject_column) if observation_paths else None
    aggregated = aggregate_time_decode_results(results, observations=observations, ece_bins=ece_bins)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    aggregated.to_csv(out_path, index=False)
    return aggregated


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate time-resolved decoding CSV files across folds and subjects.")
    parser.add_argument("csv", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--subject-column")
    parser.add_argument("--observations", nargs="+", type=Path, help="Optional held-out probability observation CSVs used to recompute exact pooled ECE.")
    parser.add_argument("--ece-bins", type=int, default=10)
    args = parser.parse_args()
    aggregated = aggregate_time_decode_csvs(args.csv, out_path=args.out, subject_column=args.subject_column, observation_paths=args.observations, ece_bins=args.ece_bins)
    print(f"Wrote {args.out}")
    print(f"Aggregated {len(args.csv)} file(s) across {int(aggregated['n_subjects'].max())} subject(s).")


if __name__ == "__main__":
    main()
