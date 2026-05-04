from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from reptrace.metadata import prepare_binary_metadata
from reptrace.mne_time_decode import run_time_resolved_decode
from reptrace.plot_time_decode import plot_time_decode_results
from reptrace.results import aggregate_time_decode_csvs
from reptrace.decoding import DECODER_CHOICES, normalize_decoder_name


@dataclass(frozen=True)
class BenchmarkRun:
    """Paths created by a benchmark manifest run."""

    result_csvs: list[Path]
    aggregate_csv: Path | None
    plot_path: Path | None
    calibration_csvs: list[Path]


def _missing(value: Any) -> bool:
    return value is None or pd.isna(value) or str(value).strip() == ""


def _string_value(row: pd.Series, column: str, default: str | None = None) -> str | None:
    if column not in row or _missing(row[column]):
        return default
    return str(row[column])


def _float_value(row: pd.Series, column: str, default: float | None = None) -> float | None:
    value = _string_value(row, column)
    return default if value is None else float(value)


def _int_value(row: pd.Series, column: str, default: int) -> int:
    value = _string_value(row, column)
    return default if value is None else int(float(value))


def _bool_value(row: pd.Series, column: str, default: bool = False) -> bool:
    value = _string_value(row, column)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "y"}


def _resolve_path(value: str | None, base_dir: Path) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = base_dir / path
    return path


def _required_path(row: pd.Series, column: str, base_dir: Path) -> Path:
    value = _string_value(row, column)
    if value is None:
        raise ValueError(f"Manifest row is missing required column '{column}'.")
    path = _resolve_path(value, base_dir)
    if path is None:
        raise ValueError(f"Manifest row is missing required column '{column}'.")
    return path


def _safe_name(value: str) -> str:
    return value.lower().replace("-", "_").replace(" ", "_")


def _decoder_output_stem(subject: str, decoder: str, has_decoder_column: bool) -> str:
    return f"{subject}_{_safe_name(decoder)}" if has_decoder_column else subject


def _prepare_or_resolve_metadata(row: pd.Series, manifest_dir: Path, out_dir: Path, subject: str) -> Path | None:
    events_csv = _resolve_path(_string_value(row, "events_csv"), manifest_dir)
    metadata_csv = _resolve_path(_string_value(row, "metadata_csv"), manifest_dir)
    source_column = _string_value(row, "source_column")
    positive_pattern = _string_value(row, "positive_pattern")

    if events_csv is None:
        return metadata_csv
    if source_column is None or positive_pattern is None:
        raise ValueError(
            f"Subject '{subject}' has events_csv but lacks source_column or positive_pattern."
        )

    metadata_out = _resolve_path(_string_value(row, "metadata_out"), manifest_dir)
    if metadata_out is None:
        metadata_out = out_dir / "metadata" / f"{subject}_metadata.csv"

    prepare_binary_metadata(
        events_csv=events_csv,
        out_path=metadata_out,
        source_column=source_column,
        positive_pattern=positive_pattern,
        negative_pattern=_string_value(row, "negative_pattern"),
        label_column=_string_value(row, "label_column", "condition") or "condition",
        positive_label=_string_value(row, "positive_label", "positive") or "positive",
        negative_label=_string_value(row, "negative_label", "negative") or "negative",
        case_sensitive=_bool_value(row, "case_sensitive"),
    )
    return metadata_out


def run_benchmark_manifest(
    manifest_csv: Path,
    *,
    out_dir: Path,
    aggregate_out: Path | None = None,
    plot_out: Path | None = None,
    chance: float | None = None,
    default_label_column: str | None = None,
    default_group_column: str | None = None,
    default_picks: str = "data",
    default_tmin: float | None = None,
    default_tmax: float | None = None,
    default_window_ms: float = 20.0,
    default_step_ms: float = 10.0,
    default_n_splits: int = 5,
    default_max_iter: int = 1000,
    default_decoder: str = "logistic",
    calibration_dir: Path | None = None,
    calibration_bins: int = 10,
) -> BenchmarkRun:
    """Run a manifest-defined benchmark and optionally aggregate and plot results."""
    manifest = pd.read_csv(manifest_csv)
    if "subject" not in manifest.columns or "epochs" not in manifest.columns:
        raise ValueError("Manifest must contain 'subject' and 'epochs' columns.")

    manifest_dir = manifest_csv.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    result_csvs: list[Path] = []
    calibration_csvs: list[Path] = []

    for _, row in manifest.iterrows():
        subject = _string_value(row, "subject")
        if subject is None:
            raise ValueError("Manifest contains a row with an empty subject.")

        label_column = _string_value(row, "label_column", default_label_column)
        if label_column is None:
            raise ValueError(f"Subject '{subject}' has no label column.")
        decoder = normalize_decoder_name(_string_value(row, "decoder", default_decoder) or default_decoder)

        output_csv = _resolve_path(_string_value(row, "out_csv"), manifest_dir)
        if output_csv is None:
            output_stem = f"{_decoder_output_stem(subject, decoder, 'decoder' in manifest.columns)}_time_decode"
            output_csv = out_dir / f"{output_stem}.csv"
        calibration_out_csv = _resolve_path(_string_value(row, "calibration_out_csv"), manifest_dir)
        if calibration_out_csv is None and calibration_dir is not None:
            calibration_out_csv = calibration_dir / f"{_decoder_output_stem(subject, decoder, 'decoder' in manifest.columns)}_calibration_bins.csv"

        metadata_csv = _prepare_or_resolve_metadata(row, manifest_dir, out_dir, subject)
        results = run_time_resolved_decode(
            epochs_path=_required_path(row, "epochs", manifest_dir),
            metadata_csv=metadata_csv,
            label_column=label_column,
            group_column=_string_value(row, "group_column", default_group_column),
            out_path=output_csv,
            picks=_string_value(row, "picks", default_picks) or default_picks,
            tmin=_float_value(row, "tmin", default_tmin),
            tmax=_float_value(row, "tmax", default_tmax),
            window_ms=_float_value(row, "window_ms", default_window_ms) or default_window_ms,
            step_ms=_float_value(row, "step_ms", default_step_ms) or default_step_ms,
            n_splits=_int_value(row, "n_splits", default_n_splits),
            max_iter=_int_value(row, "max_iter", default_max_iter),
            decoder=decoder,
            calibration_out_path=calibration_out_csv,
            calibration_bins=_int_value(row, "calibration_bins", calibration_bins),
        )
        if calibration_out_csv is not None:
            calibration_csvs.append(calibration_out_csv)
        if "subject" not in results.columns:
            results.insert(0, "subject", subject)
        else:
            results["subject"] = subject
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(output_csv, index=False)
        result_csvs.append(output_csv)

    if aggregate_out is None:
        aggregate_out = out_dir / "summary.csv"
    aggregate = aggregate_time_decode_csvs(result_csvs, out_path=aggregate_out)
    aggregate_path: Path | None = aggregate_out

    plot_path: Path | None = None
    if plot_out is not None:
        plot_time_decode_results(
            aggregate_out,
            out_path=plot_out,
            chance=chance,
            title=f"RepTrace benchmark ({int(aggregate['n_subjects'].max())} subject(s))",
        )
        plot_path = plot_out

    return BenchmarkRun(result_csvs=result_csvs, aggregate_csv=aggregate_path, plot_path=plot_path, calibration_csvs=calibration_csvs)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a manifest-defined RepTrace benchmark."
    )
    parser.add_argument("manifest_csv", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--aggregate-out", type=Path)
    parser.add_argument("--plot-out", type=Path)
    parser.add_argument("--chance", type=float)
    parser.add_argument("--label-column")
    parser.add_argument("--group-column")
    parser.add_argument("--picks", default="data")
    parser.add_argument("--tmin", type=float)
    parser.add_argument("--tmax", type=float)
    parser.add_argument("--window-ms", type=float, default=20.0)
    parser.add_argument("--step-ms", type=float, default=10.0)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--decoder", choices=DECODER_CHOICES, default="logistic")
    parser.add_argument("--calibration-dir", type=Path)
    parser.add_argument("--calibration-bins", type=int, default=10)
    args = parser.parse_args()

    run = run_benchmark_manifest(
        args.manifest_csv,
        out_dir=args.out_dir,
        aggregate_out=args.aggregate_out,
        plot_out=args.plot_out,
        chance=args.chance,
        default_label_column=args.label_column,
        default_group_column=args.group_column,
        default_picks=args.picks,
        default_tmin=args.tmin,
        default_tmax=args.tmax,
        default_window_ms=args.window_ms,
        default_step_ms=args.step_ms,
        default_n_splits=args.n_splits,
        default_max_iter=args.max_iter,
        default_decoder=args.decoder,
        calibration_dir=args.calibration_dir,
        calibration_bins=args.calibration_bins,
    )
    print(f"Wrote {len(run.result_csvs)} subject result file(s).")
    if run.aggregate_csv is not None:
        print(f"Wrote aggregate CSV: {run.aggregate_csv}")
    if run.plot_path is not None:
        print(f"Wrote plot: {run.plot_path}")
    if run.calibration_csvs:
        print(f"Wrote {len(run.calibration_csvs)} calibration bin file(s).")


if __name__ == "__main__":
    main()
