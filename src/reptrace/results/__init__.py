from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
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
    "read_probability_observations",
    "read_time_decode_observations",
    "read_time_decode_results",
    "summarize_metric_table",
]

METRIC_COLUMNS = ("accuracy", "log_loss", "brier", "ece")
SUMMARY_GROUP_COLUMNS = ("decoder", "emission_mode")
WEIGHT_COLUMN = "n_test"
DEFAULT_ECE_BINS = 10


def read_time_decode_results(
    csv_paths: list[Path],
    *,
    subject_column: str | None = None,
) -> pd.DataFrame:
    """Read one or more time-resolved decoding result CSV files."""
    if not csv_paths:
        raise ValueError("At least one CSV path is required.")

    frames = []
    for csv_path in csv_paths:
        frame = pd.read_csv(csv_path)
        missing = [column for column in ("time", *METRIC_COLUMNS) if column not in frame.columns]
        if missing:
            raise ValueError(f"{csv_path} is missing required columns: {missing}")
        if subject_column is not None:
            if subject_column not in frame.columns:
                raise ValueError(f"{csv_path} is missing subject column '{subject_column}'.")
            frame["subject"] = frame[subject_column].astype(str)
        elif "subject" not in frame.columns:
            frame["subject"] = csv_path.stem
        else:
            frame["subject"] = frame["subject"].astype(str)
        if "emission_mode" not in frame.columns:
            frame["emission_mode"] = "calibrated"
        frame["source_file"] = csv_path.name
        frames.append(frame)

    return pd.concat(frames, ignore_index=True)


def read_probability_observations(
    csv_paths: list[Path],
    *,
    subject_column: str | None = None,
    fallback_subjects_by_file: Mapping[str, object] | None = None,
) -> pd.DataFrame:
    """Read held-out probability observation CSVs for exact calibration aggregation."""
    if not csv_paths:
        raise ValueError("At least one observation CSV path is required.")

    fallback_subjects_by_file = dict(fallback_subjects_by_file or {})
    frames = []
    expected_probability_columns: tuple[str, ...] | None = None
    for csv_path in csv_paths:
        frame = pd.read_csv(csv_path)
        prob_columns = probability_columns(frame)
        missing = [column for column in ("time", "true_label") if column not in frame.columns]
        if not prob_columns:
            missing.append("prob_class_*")
        if missing:
            raise ValueError(f"{csv_path} is missing required probability-observation columns: {missing}")
        if expected_probability_columns is None:
            expected_probability_columns = prob_columns
        elif prob_columns != expected_probability_columns:
            raise ValueError(
                f"{csv_path} probability columns {list(prob_columns)} do not match "
                f"the first observation file {list(expected_probability_columns)}."
            )

        fallback_subject = str(fallback_subjects_by_file.get(csv_path.name, csv_path.stem))
        if subject_column is not None:
            if subject_column not in frame.columns:
                raise ValueError(f"{csv_path} is missing subject column '{subject_column}'.")
            frame["subject"] = frame[subject_column]
        elif "subject" not in frame.columns:
            frame["subject"] = fallback_subject

        frame["subject"] = frame["subject"].where(pd.notna(frame["subject"]), fallback_subject).astype(str)
        frame.loc[frame["subject"].str.len() == 0, "subject"] = fallback_subject
        if "emission_mode" not in frame.columns:
            frame["emission_mode"] = "calibrated"
        else:
            frame["emission_mode"] = frame["emission_mode"].where(pd.notna(frame["emission_mode"]), "calibrated").astype(str)
            frame.loc[frame["emission_mode"].str.len() == 0, "emission_mode"] = "calibrated"
        frame["source_file"] = csv_path.name
        frames.append(frame)

    return pd.concat(frames, ignore_index=True)


read_time_decode_observations = read_probability_observations


def _mean_across_folds(
    results: pd.DataFrame,
    group_columns: list[str],
    *,
    metric_columns: Sequence[str] | str | None = None,
) -> pd.DataFrame:
    """Average fold metrics, weighting by held-out sample count when available."""
    if metric_columns is None:
        selected_metric_columns = list(METRIC_COLUMNS)
    elif isinstance(metric_columns, str):
        selected_metric_columns = [metric_columns]
    else:
        selected_metric_columns = list(metric_columns)
    if not selected_metric_columns:
        raise ValueError("At least one metric column is required.")
    missing = [column for column in selected_metric_columns if column not in results.columns]
    if missing:
        raise ValueError(f"Results are missing metric columns: {missing}")

    if WEIGHT_COLUMN not in results.columns:
        return results.groupby(group_columns, as_index=False)[selected_metric_columns].mean()

    weighted = results.copy()
    weights = pd.to_numeric(weighted[WEIGHT_COLUMN], errors="coerce")
    if weights.isna().any() or not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError(f"Column '{WEIGHT_COLUMN}' must contain positive finite fold sizes.")
    weighted["__fold_weight"] = weights.astype(float)

    weighted_columns = []
    denominator_columns = []
    for metric in selected_metric_columns:
        values = pd.to_numeric(weighted[metric], errors="coerce")
        weighted_column = f"__weighted_{metric}"
        denominator_column = f"__weight_{metric}"
        weighted[weighted_column] = values * weighted["__fold_weight"]
        weighted[denominator_column] = weighted["__fold_weight"].where(values.notna())
        weighted_columns.append(weighted_column)
        denominator_columns.append(denominator_column)

    aggregate_columns = [*weighted_columns, *denominator_columns]
    grouped = weighted.groupby(group_columns, as_index=False)[aggregate_columns].sum()
    for metric, weighted_column, denominator_column in zip(
        selected_metric_columns, weighted_columns, denominator_columns
    ):
        grouped[metric] = grouped[weighted_column] / grouped[denominator_column]

    return grouped[[*group_columns, *selected_metric_columns]]


mean_across_folds = _mean_across_folds


def _normalize_emission_mode(frame: pd.DataFrame) -> pd.DataFrame:
    if "emission_mode" not in frame.columns:
        return frame
    normalized = frame.copy()
    normalized["emission_mode"] = normalized["emission_mode"].where(pd.notna(normalized["emission_mode"]), "calibrated").astype(str)
    normalized.loc[normalized["emission_mode"].str.len() == 0, "emission_mode"] = "calibrated"
    return normalized


def _probability_ece_by_group(observations: pd.DataFrame, group_columns: list[str], *, n_bins: int) -> pd.DataFrame:
    """Compute ECE from pooled probability observations within each subject/time group."""
    prob_columns = list(probability_columns(observations))
    missing = [column for column in (*group_columns, "true_label") if column not in observations.columns]
    if not prob_columns:
        missing.append("prob_class_*")
    if missing:
        raise ValueError(f"Probability observations are missing required columns: {missing}")

    working = observations.copy()
    working["time"] = pd.to_numeric(working["time"], errors="coerce")
    if working["time"].isna().any():
        raise ValueError("Probability-observation column 'time' must be numeric and non-missing.")
    working["true_label"] = pd.to_numeric(working["true_label"], errors="coerce")
    if working["true_label"].isna().any():
        raise ValueError("Probability-observation column 'true_label' must be numeric and non-missing.")
    for column in prob_columns:
        working[column] = pd.to_numeric(working[column], errors="coerce")
    if working[prob_columns].isna().any().any():
        raise ValueError("Probability-observation columns must be numeric and non-missing.")

    labels = working["true_label"].to_numpy(dtype=int)
    if bool(((labels < 0) | (labels >= len(prob_columns))).any()):
        raise ValueError("Probability-observation true_label values must index prob_class_* columns.")
    probabilities = working[prob_columns].to_numpy(dtype=float)
    if not np.isfinite(probabilities).all():
        raise ValueError("Probability-observation columns must be finite.")

    rows: list[dict[str, object]] = []
    for group_key, group in working.groupby(group_columns, dropna=False, sort=True):
        if len(group_columns) == 1 and not isinstance(group_key, tuple):
            group_key = (group_key,)
        row = dict(zip(group_columns, group_key))
        probabilities = group[prob_columns].to_numpy(dtype=float)
        labels = group["true_label"].to_numpy(dtype=int)
        row["n_observations"] = int(len(group))
        row["ece"] = expected_calibration_error(probabilities, labels, n_bins=n_bins)
        rows.append(row)

    return pd.DataFrame(rows)


def _expected_observation_counts(results: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame | None:
    if WEIGHT_COLUMN not in results.columns:
        return None

    weights = pd.to_numeric(results[WEIGHT_COLUMN], errors="coerce")
    if weights.isna().any() or not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError(f"Column '{WEIGHT_COLUMN}' must contain positive finite fold sizes.")
    weighted = results.copy()
    weighted["__expected_observations"] = weights.astype(float)
    return weighted.groupby(group_columns, as_index=False)["__expected_observations"].sum()


def _replace_ece_from_observations(
    subject_time: pd.DataFrame,
    observations: pd.DataFrame,
    subject_time_keys: list[str],
    *,
    n_bins: int,
    expected_counts: pd.DataFrame | None = None,
) -> pd.DataFrame:
    observation_ece = _probability_ece_by_group(
        _normalize_emission_mode(observations),
        subject_time_keys,
        n_bins=n_bins,
    )

    merged = subject_time.drop(columns=["ece"]).merge(
        observation_ece,
        on=subject_time_keys,
        how="left",
        validate="one_to_one",
    )
    if merged["ece"].isna().any():
        missing = merged.loc[merged["ece"].isna(), subject_time_keys].drop_duplicates().head(5).to_dict("records")
        raise ValueError(
            "Probability observations do not cover all result subject/time groups "
            f"needed for exact ECE aggregation. Missing examples: {missing}"
        )

    if expected_counts is not None:
        count_check = expected_counts.merge(
            observation_ece[[*subject_time_keys, "n_observations"]],
            on=subject_time_keys,
            how="left",
            validate="one_to_one",
        )
        count_check["__count_matches"] = np.isclose(count_check["__expected_observations"], count_check["n_observations"])
        if count_check["__count_matches"].isna().any() or not bool(count_check["__count_matches"].all()):
            examples = count_check.loc[
                ~count_check["__count_matches"].fillna(False),
                [*subject_time_keys, "__expected_observations", "n_observations"],
            ]
            raise ValueError(
                "Probability-observation row counts do not match fold n_test totals for exact ECE aggregation. "
                f"Mismatch examples: {examples.head(5).to_dict('records')}"
            )

    subject_key_frame = subject_time[subject_time_keys].drop_duplicates()
    extras = observation_ece.merge(subject_key_frame, on=subject_time_keys, how="left", indicator=True)
    extras = extras.loc[extras["_merge"] == "left_only", subject_time_keys]
    if not extras.empty:
        examples = extras.drop_duplicates().head(5).to_dict("records")
        raise ValueError(
            "Probability observations contain subject/time groups that are absent from result metrics. "
            f"Extra examples: {examples}"
        )

    return merged.drop(columns=["n_observations"], errors="ignore")


def _observation_subject_fallbacks(results: pd.DataFrame, csv_paths: list[Path], observation_csv_paths: list[Path] | None) -> dict[str, object]:
    if not observation_csv_paths or len(csv_paths) != len(observation_csv_paths) or "source_file" not in results.columns:
        return {}

    fallbacks: dict[str, object] = {}
    for result_path, observation_path in zip(csv_paths, observation_csv_paths):
        subjects = results.loc[results["source_file"] == result_path.name, "subject"].dropna().astype(str).unique()
        if len(subjects) == 1:
            fallbacks[observation_path.name] = subjects[0]
    return fallbacks


def aggregate_time_decode_results(
    results: pd.DataFrame,
    *,
    observations: pd.DataFrame | None = None,
    ece_bins: int = DEFAULT_ECE_BINS,
) -> pd.DataFrame:
    """Aggregate fold-level decoding results into time-level summary statistics.

    Fold-linear metrics are averaged within subject/time after weighting by
    ``n_test`` when available. When probability observations are provided,
    ECE is recomputed from pooled held-out probabilities within each
    subject/time group instead of averaging fold-level ECE values.
    """
    missing = [column for column in ("subject", "time", *METRIC_COLUMNS) if column not in results.columns]
    if missing:
        raise ValueError(f"Results are missing required columns: {missing}")
    if ece_bins < 1:
        raise ValueError("ece_bins must be positive")

    results = _normalize_emission_mode(results)
    group_columns = [column for column in SUMMARY_GROUP_COLUMNS if column in results.columns]
    subject_time_keys = [*group_columns, "subject", "time"]
    aggregate_keys = [*group_columns, "time"]
    subject_time = _mean_across_folds(results, subject_time_keys).sort_values(subject_time_keys)
    if observations is not None:
        subject_time = _replace_ece_from_observations(
            subject_time,
            observations,
            subject_time_keys,
            n_bins=ece_bins,
            expected_counts=_expected_observation_counts(results, subject_time_keys),
        )

    grouped = subject_time.groupby(aggregate_keys, as_index=False)
    aggregated = grouped[list(METRIC_COLUMNS)].mean()
    n_subjects = grouped["subject"].nunique().rename(columns={"subject": "n_subjects"})
    aggregated = aggregated.merge(n_subjects, on=aggregate_keys)

    for metric in METRIC_COLUMNS:
        sem = grouped[metric].sem().rename(columns={metric: f"{metric}_sem"})
        aggregated = aggregated.merge(sem, on=aggregate_keys)
        aggregated = aggregated.rename(columns={metric: f"{metric}_mean"})

    return aggregated.sort_values(aggregate_keys).reset_index(drop=True)


def aggregate_time_decode_csvs(
    csv_paths: list[Path],
    out_path: Path,
    *,
    subject_column: str | None = None,
    observation_csv_paths: list[Path] | None = None,
    observation_subject_column: str | None = None,
    ece_bins: int = DEFAULT_ECE_BINS,
) -> pd.DataFrame:
    """Aggregate time-resolved decoding CSV files and write a summary CSV."""
    results = read_time_decode_results(csv_paths, subject_column=subject_column)
    observations = None
    if observation_csv_paths is not None:
        observations = read_probability_observations(
            observation_csv_paths,
            subject_column=observation_subject_column,
            fallback_subjects_by_file=_observation_subject_fallbacks(results, csv_paths, observation_csv_paths),
        )
    aggregated = aggregate_time_decode_results(results, observations=observations, ece_bins=ece_bins)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    aggregated.to_csv(out_path, index=False)
    return aggregated


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate time-resolved decoding CSV files across folds and subjects.")
    parser.add_argument("csv", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--subject-column")
    parser.add_argument("--observations", nargs="+", type=Path, help="Optional probability-observation CSVs used to recompute ECE exactly.")
    parser.add_argument("--observation-subject-column", help="Subject column for --observations; omitted values fall back to subject or filename mapping.")
    parser.add_argument("--ece-bins", type=int, default=DEFAULT_ECE_BINS, help="Number of bins for exact observation-level ECE.")
    args = parser.parse_args()

    aggregated = aggregate_time_decode_csvs(
        args.csv,
        out_path=args.out,
        subject_column=args.subject_column,
        observation_csv_paths=args.observations,
        observation_subject_column=args.observation_subject_column,
        ece_bins=args.ece_bins,
    )
    print(f"Wrote {args.out}")
    print(f"Aggregated {len(args.csv)} file(s) across {int(aggregated['n_subjects'].max())} subject(s).")


if __name__ == "__main__":
    main()
