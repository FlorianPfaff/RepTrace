from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from reptrace.metadata import prepare_binary_metadata
from reptrace.mne_time_decode import run_time_resolved_decode
from reptrace.plot_time_decode import plot_time_decode_results
from reptrace.results import aggregate_time_decode_csvs, write_provenance_table
from reptrace.temporal_smoothing import DEFAULT_EMISSION_SUFFIX, DEFAULT_FIT_WINDOW, smooth_probability_observations
from reptrace.decoding import (
    DECODER_CLI_CHOICES,
    EMISSION_MODE_CHOICES,
    FEATURE_PREPROCESSOR_CHOICES,
    TUNING_SCORING_CHOICES,
    normalize_decoder_name,
    normalize_emission_mode,
    normalize_feature_preprocessor,
    normalize_tuning_scoring,
    parse_c_grid,
)

EMISSION_RUN_CHOICES = (*EMISSION_MODE_CHOICES, "both")
TemporalTrainWindow = tuple[float, float]


@dataclass(frozen=True)
class BenchmarkRun:
    """Paths created by a benchmark manifest run."""

    result_csvs: list[Path]
    aggregate_csv: Path | None
    plot_path: Path | None
    calibration_csvs: list[Path]
    observation_csvs: list[Path]
    provenance_csv: Path | None = None
    skipped_existing: int = 0
    smoothed_observation_csv: Path | None = None
    smoothed_metric_csv: Path | None = None


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


def _temporal_train_window_value(
    row: pd.Series,
    default: TemporalTrainWindow | None = None,
) -> TemporalTrainWindow | None:
    start = _float_value(row, "temporal_train_window_start")
    stop = _float_value(row, "temporal_train_window_stop")
    if start is not None or stop is not None:
        if start is None or stop is None:
            raise ValueError("Manifest must set both temporal_train_window_start and temporal_train_window_stop.")
        return (start, stop)

    value = _string_value(row, "temporal_train_window")
    if value is None:
        return default
    parts = value.replace(",", " ").replace("|", " ").split()
    if len(parts) != 2:
        raise ValueError("temporal_train_window must contain exactly two values: START STOP.")
    return (float(parts[0]), float(parts[1]))


def _resolve_path(value: str | None, base_dir: Path) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = base_dir / path
    return path


def _usable_file(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def _required_path(row: pd.Series, column: str, base_dir: Path) -> Path:
    value = _string_value(row, column)
    if value is None:
        raise ValueError(f"Manifest row is missing required column '{column}'.")
    path = _resolve_path(value, base_dir)
    if path is None:
        raise ValueError(f"Manifest row is missing required column '{column}'.")
    return path


def _safe_name(value: str) -> str:
    return value.lower().replace("-", "_").replace(" ", "_").replace(".", "p").replace("|", "_")


def _safe_window_name(window: TemporalTrainWindow) -> str:
    return f"trainwin{_safe_name(f'{window[0]:g}')}_{_safe_name(f'{window[1]:g}')}"


def _decoder_output_stem(subject: str, decoder: str, has_decoder_column: bool) -> str:
    return f"{subject}_{_safe_name(decoder)}" if has_decoder_column else subject


def _output_stem(
    subject: str,
    decoder: str,
    emission_mode: str,
    *,
    has_decoder_column: bool,
    has_emission_mode_column: bool,
    variant: str | None = None,
    feature_preprocessor: str = "none",
    pca_components: str | None = None,
    tune_hyperparameters: bool = False,
    tuning_scoring: str = "accuracy",
    temporal_train_window: TemporalTrainWindow | None = None,
    has_feature_preprocessor_column: bool = False,
    has_pca_components_column: bool = False,
    has_tune_hyperparameters_column: bool = False,
    has_tuning_scoring_column: bool = False,
    has_temporal_train_window_column: bool = False,
) -> str:
    parts = [subject]
    if variant is not None:
        parts.append(_safe_name(variant))
        return "_".join(parts)
    if has_decoder_column:
        parts.append(_safe_name(decoder))
    if has_emission_mode_column:
        parts.append(_safe_name(emission_mode))
    if has_feature_preprocessor_column:
        parts.append(_safe_name(feature_preprocessor))
    if has_pca_components_column and pca_components is not None:
        parts.append(f"pca{_safe_name(str(pca_components))}")
    if has_tune_hyperparameters_column:
        parts.append("tuned" if tune_hyperparameters else "untuned")
    if has_tuning_scoring_column and tune_hyperparameters:
        parts.append(_safe_name(tuning_scoring))
    if has_temporal_train_window_column and temporal_train_window is not None:
        parts.append(_safe_window_name(temporal_train_window))
    return "_".join(parts)


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
    provenance_out: Path | None = None,
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
    default_emission_mode: str = "calibrated",
    default_feature_preprocessor: str = "none",
    default_pca_components: str | None = None,
    default_tune_hyperparameters: bool = False,
    default_tuning_cv_splits: int = 3,
    default_tuning_scoring: str = "accuracy",
    default_tuning_c_grid: str | None = None,
    default_temporal_train_window: TemporalTrainWindow | None = None,
    calibration_dir: Path | None = None,
    calibration_bins: int = 10,
    observation_dir: Path | None = None,
    temporal_smoothing_dir: Path | None = None,
    temporal_smoothing_fit_window: tuple[float, float] | None = DEFAULT_FIT_WINDOW,
    temporal_smoothing_stay_grid_size: int = 200,
    temporal_smoothing_emission_suffix: str = DEFAULT_EMISSION_SUFFIX,
    resume: bool = False,
) -> BenchmarkRun:
    """Run a manifest-defined benchmark and optionally aggregate and plot results."""
    manifest = pd.read_csv(manifest_csv)
    if "subject" not in manifest.columns or "epochs" not in manifest.columns:
        raise ValueError("Manifest must contain 'subject' and 'epochs' columns.")

    manifest_dir = manifest_csv.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    if temporal_smoothing_dir is not None and observation_dir is None:
        observation_dir = out_dir / "observations"
    result_csvs: list[Path] = []
    calibration_csvs: list[Path] = []
    observation_csvs: list[Path] = []
    skipped_existing = 0

    for _, row in manifest.iterrows():
        subject = _string_value(row, "subject")
        if subject is None:
            raise ValueError("Manifest contains a row with an empty subject.")

        label_column = _string_value(row, "label_column", default_label_column)
        if label_column is None:
            raise ValueError(f"Subject '{subject}' has no label column.")
        decoder = normalize_decoder_name(_string_value(row, "decoder", default_decoder) or default_decoder)
        emission_mode = _string_value(row, "emission_mode", default_emission_mode) or default_emission_mode
        if emission_mode != "both":
            emission_mode = normalize_emission_mode(emission_mode)
        feature_preprocessor = normalize_feature_preprocessor(
            _string_value(row, "feature_preprocessor", default_feature_preprocessor) or default_feature_preprocessor
        )
        pca_components = _string_value(row, "pca_components", default_pca_components)
        tune_hyperparameters = _bool_value(row, "tune_hyperparameters", default_tune_hyperparameters)
        tuning_cv_splits = _int_value(row, "tuning_cv_splits", default_tuning_cv_splits)
        tuning_scoring = normalize_tuning_scoring(
            _string_value(row, "tuning_scoring", default_tuning_scoring) or default_tuning_scoring
        )
        tuning_c_grid = _string_value(row, "tuning_c_grid", default_tuning_c_grid)
        temporal_train_window = _temporal_train_window_value(row, default_temporal_train_window)
        output_stem = _output_stem(
            subject,
            decoder,
            emission_mode,
            has_decoder_column="decoder" in manifest.columns,
            has_emission_mode_column="emission_mode" in manifest.columns,
            variant=_string_value(row, "variant"),
            feature_preprocessor=feature_preprocessor,
            pca_components=pca_components,
            tune_hyperparameters=tune_hyperparameters,
            tuning_scoring=tuning_scoring,
            temporal_train_window=temporal_train_window,
            has_feature_preprocessor_column="feature_preprocessor" in manifest.columns,
            has_pca_components_column="pca_components" in manifest.columns,
            has_tune_hyperparameters_column="tune_hyperparameters" in manifest.columns,
            has_tuning_scoring_column="tuning_scoring" in manifest.columns,
            has_temporal_train_window_column=bool(
                {"temporal_train_window", "temporal_train_window_start", "temporal_train_window_stop"}.intersection(
                    manifest.columns
                )
            ),
        )

        output_csv = _resolve_path(_string_value(row, "out_csv"), manifest_dir)
        if output_csv is None:
            output_csv = out_dir / f"{output_stem}_time_decode.csv"
        calibration_out_csv = _resolve_path(_string_value(row, "calibration_out_csv"), manifest_dir)
        if calibration_out_csv is None and calibration_dir is not None:
            calibration_out_csv = calibration_dir / f"{output_stem}_calibration_bins.csv"
        observation_out_csv = _resolve_path(_string_value(row, "observation_out_csv"), manifest_dir)
        if observation_out_csv is None and observation_dir is not None:
            observation_out_csv = observation_dir / f"{output_stem}_observations.csv"

        if (
            resume
            and _usable_file(output_csv)
            and (calibration_out_csv is None or _usable_file(calibration_out_csv))
            and (observation_out_csv is None or _usable_file(observation_out_csv))
        ):
            result_csvs.append(output_csv)
            if calibration_out_csv is not None:
                calibration_csvs.append(calibration_out_csv)
            if observation_out_csv is not None:
                observation_csvs.append(observation_out_csv)
            skipped_existing += 1
            continue

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
            emission_mode=emission_mode,
            feature_preprocessor=feature_preprocessor,
            pca_components=pca_components,
            tune_hyperparameters=tune_hyperparameters,
            tuning_cv_splits=tuning_cv_splits,
            tuning_scoring=tuning_scoring,
            tuning_c_grid=tuning_c_grid,
            temporal_train_window=temporal_train_window,
            calibration_out_path=calibration_out_csv,
            calibration_bins=_int_value(row, "calibration_bins", calibration_bins),
            observation_out_path=observation_out_csv,
            subject=subject,
        )
        if calibration_out_csv is not None:
            calibration_csvs.append(calibration_out_csv)
        if observation_out_csv is not None:
            observation_csvs.append(observation_out_csv)
        if "subject" not in results.columns:
            results.insert(0, "subject", subject)
        else:
            results["subject"] = subject
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(output_csv, index=False)
        result_csvs.append(output_csv)

    aggregate_result_csvs = list(result_csvs)
    aggregate_observation_csvs = list(observation_csvs)
    smoothed_observation_csv: Path | None = None
    smoothed_metric_csv: Path | None = None
    if temporal_smoothing_dir is not None:
        if not observation_csvs:
            raise ValueError("Temporal smoothing requires probability observations; pass --observation-dir.")
        smoothed_observation_csv = temporal_smoothing_dir / "smoothed_observations.csv"
        smoothed_metric_csv = temporal_smoothing_dir / "smoothed_metrics.csv"
        if not (resume and _usable_file(smoothed_observation_csv) and _usable_file(smoothed_metric_csv)):
            smooth_probability_observations(
                observation_csvs,
                fit_window=temporal_smoothing_fit_window,
                stay_grid_size=temporal_smoothing_stay_grid_size,
                emission_suffix=temporal_smoothing_emission_suffix,
                ece_bins=calibration_bins,
                out_observations=smoothed_observation_csv,
                out_metrics=smoothed_metric_csv,
            )
        aggregate_result_csvs.append(smoothed_metric_csv)
        aggregate_observation_csvs.append(smoothed_observation_csv)

    if aggregate_out is None:
        aggregate_out = out_dir / "summary.csv"
    aggregate = aggregate_time_decode_csvs(
        aggregate_result_csvs,
        out_path=aggregate_out,
        observation_csv_paths=aggregate_observation_csvs or None,
    )
    aggregate_path: Path | None = aggregate_out
    if provenance_out is None:
        provenance_out = out_dir / "provenance.csv"
    provenance = write_provenance_table(
        aggregate,
        aggregate_result_csvs,
        provenance_out,
    )
    provenance_path: Path | None = provenance_out

    plot_path: Path | None = None
    if plot_out is not None:
        plot_time_decode_results(
            aggregate_out,
            out_path=plot_out,
            chance=chance,
            title=f"RepTrace benchmark ({int(provenance['n_subjects'].max())} subject(s))",
        )
        plot_path = plot_out

    return BenchmarkRun(
        result_csvs=result_csvs,
        aggregate_csv=aggregate_path,
        provenance_csv=provenance_path,
        plot_path=plot_path,
        calibration_csvs=calibration_csvs,
        observation_csvs=observation_csvs,
        skipped_existing=skipped_existing,
        smoothed_observation_csv=smoothed_observation_csv,
        smoothed_metric_csv=smoothed_metric_csv,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a manifest-defined RepTrace benchmark."
    )
    parser.add_argument("manifest_csv", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--aggregate-out", type=Path)
    parser.add_argument("--provenance-out", type=Path)
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
    parser.add_argument("--decoder", choices=DECODER_CLI_CHOICES, default="logistic")
    parser.add_argument("--emission-mode", choices=EMISSION_RUN_CHOICES, default="calibrated")
    parser.add_argument("--feature-preprocessor", choices=(*FEATURE_PREPROCESSOR_CHOICES, "pca-whiten"), default="none")
    parser.add_argument("--pca-components")
    parser.add_argument("--tune-hyperparameters", action="store_true")
    parser.add_argument("--tuning-cv-splits", type=int, default=3)
    parser.add_argument("--tuning-scoring", choices=TUNING_SCORING_CHOICES, default="accuracy")
    parser.add_argument("--tuning-c-grid", default=",".join(str(value) for value in parse_c_grid(None)))
    parser.add_argument("--temporal-train-window", type=float, nargs=2, metavar=("START", "STOP"))
    parser.add_argument("--calibration-dir", type=Path)
    parser.add_argument("--calibration-bins", type=int, default=10)
    parser.add_argument("--observation-dir", type=Path, help="Optional directory for held-out trial/time probability observation CSVs.")
    parser.add_argument("--temporal-smoothing-dir", type=Path, help="Optional directory for smoothed observations and metrics.")
    parser.add_argument("--temporal-smoothing-fit-window", type=float, nargs=2, default=DEFAULT_FIT_WINDOW, metavar=("START", "STOP"))
    parser.add_argument(
        "--temporal-smoothing-full-sequence-fit",
        action="store_true",
        help="Fit temporal smoothing on every available time bin instead of --temporal-smoothing-fit-window.",
    )
    parser.add_argument("--temporal-smoothing-stay-grid-size", type=int, default=200)
    parser.add_argument("--temporal-smoothing-emission-suffix", default=DEFAULT_EMISSION_SUFFIX)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip manifest rows whose result CSV and requested calibration-bin CSV already exist.",
    )
    args = parser.parse_args()

    run = run_benchmark_manifest(
        args.manifest_csv,
        out_dir=args.out_dir,
        aggregate_out=args.aggregate_out,
        provenance_out=args.provenance_out,
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
        default_emission_mode=args.emission_mode,
        default_feature_preprocessor=args.feature_preprocessor,
        default_pca_components=args.pca_components,
        default_tune_hyperparameters=args.tune_hyperparameters,
        default_tuning_cv_splits=args.tuning_cv_splits,
        default_tuning_scoring=args.tuning_scoring,
        default_tuning_c_grid=args.tuning_c_grid,
        default_temporal_train_window=tuple(args.temporal_train_window) if args.temporal_train_window is not None else None,
        calibration_dir=args.calibration_dir,
        calibration_bins=args.calibration_bins,
        observation_dir=args.observation_dir,
        temporal_smoothing_dir=args.temporal_smoothing_dir,
        temporal_smoothing_fit_window=None
        if args.temporal_smoothing_full_sequence_fit
        else tuple(args.temporal_smoothing_fit_window),
        temporal_smoothing_stay_grid_size=args.temporal_smoothing_stay_grid_size,
        temporal_smoothing_emission_suffix=args.temporal_smoothing_emission_suffix,
        resume=args.resume,
    )
    if run.skipped_existing:
        print(f"Skipped {run.skipped_existing} complete existing row(s).")
    print(f"Available {len(run.result_csvs)} subject result file(s).")
    if run.aggregate_csv is not None:
        print(f"Wrote aggregate CSV: {run.aggregate_csv}")
    if run.provenance_csv is not None:
        print(f"Wrote provenance CSV: {run.provenance_csv}")
    if run.plot_path is not None:
        print(f"Wrote plot: {run.plot_path}")
    if run.calibration_csvs:
        print(f"Wrote {len(run.calibration_csvs)} calibration bin file(s).")
    if run.observation_csvs:
        print(f"Wrote {len(run.observation_csvs)} probability observation file(s).")
    if run.smoothed_observation_csv is not None:
        print(f"Wrote smoothed observations: {run.smoothed_observation_csv}")
    if run.smoothed_metric_csv is not None:
        print(f"Wrote smoothed metrics: {run.smoothed_metric_csv}")


if __name__ == "__main__":
    main()
