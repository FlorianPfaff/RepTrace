from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

METRIC_COLUMNS = ("accuracy", "log_loss", "brier", "ece")
SUMMARY_GROUP_COLUMNS = ("decoder", "emission_mode")


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


def aggregate_time_decode_results(results: pd.DataFrame) -> pd.DataFrame:
    """Aggregate fold-level decoding results into time-level summary statistics."""
    missing = [column for column in ("subject", "time", *METRIC_COLUMNS) if column not in results.columns]
    if missing:
        raise ValueError(f"Results are missing required columns: {missing}")
    if "emission_mode" in results.columns:
        results = results.copy()
        results["emission_mode"] = results["emission_mode"].fillna("calibrated")

    group_columns = [column for column in SUMMARY_GROUP_COLUMNS if column in results.columns]
    subject_time_keys = [*group_columns, "subject", "time"]
    aggregate_keys = [*group_columns, "time"]
    subject_time = (
        results.groupby(subject_time_keys, as_index=False)[list(METRIC_COLUMNS)]
        .mean()
        .sort_values(subject_time_keys)
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
) -> pd.DataFrame:
    """Aggregate time-resolved decoding CSV files and write a summary CSV."""
    results = read_time_decode_results(csv_paths, subject_column=subject_column)
    aggregated = aggregate_time_decode_results(results)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    aggregated.to_csv(out_path, index=False)
    return aggregated


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate time-resolved decoding CSV files across folds and subjects."
    )
    parser.add_argument("csv", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--subject-column")
    args = parser.parse_args()

    aggregated = aggregate_time_decode_csvs(
        args.csv,
        out_path=args.out,
        subject_column=args.subject_column,
    )
    print(f"Wrote {args.out}")
    print(f"Aggregated {len(args.csv)} file(s) across {int(aggregated['n_subjects'].max())} subject(s).")


if __name__ == "__main__":
    main()
