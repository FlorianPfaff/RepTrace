from __future__ import annotations

import argparse
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import LabelEncoder

from reptrace.decoding import (
    DECODER_CHOICES,
    EMISSION_MODE_CHOICES,
    make_cross_validator,
    make_decoder_backend,
    normalize_decoder_name,
    normalize_emission_mode,
    time_windows,
)
from reptrace.metrics import brier_score_multiclass, expected_calibration_error, reliability_bins
from reptrace.observations import ProbabilityObservationTable

EMISSION_RUN_CHOICES = (*EMISSION_MODE_CHOICES, "both")


def _add_subject(row: dict, subject: str | None) -> dict:
    if subject is not None:
        row = {"subject": subject, **row}
    return row


def _load_epochs_and_metadata(
    epochs_path: Path,
    metadata_csv: Path | None,
) -> tuple[mne.Epochs, pd.DataFrame]:
    epochs = mne.read_epochs(epochs_path, preload=True, verbose="error")
    metadata = epochs.metadata.copy() if epochs.metadata is not None else None
    if metadata_csv is not None:
        metadata = pd.read_csv(metadata_csv)
    if metadata is None:
        raise ValueError("No metadata found. Provide --metadata-csv or use epochs with metadata.")
    if len(metadata) != len(epochs):
        raise ValueError(
            f"Metadata row count ({len(metadata)}) does not match epochs ({len(epochs)})."
        )
    return epochs, metadata.reset_index(drop=True)


def run_time_resolved_decode(
    epochs_path: Path,
    label_column: str,
    out_path: Path,
    *,
    metadata_csv: Path | None = None,
    group_column: str | None = None,
    picks: str = "data",
    tmin: float | None = None,
    tmax: float | None = None,
    window_ms: float = 20.0,
    step_ms: float = 10.0,
    n_splits: int = 5,
    max_iter: int = 1000,
    decoder: str = "logistic",
    emission_mode: str = "calibrated",
    calibration_out_path: Path | None = None,
    calibration_bins: int = 10,
    observation_out_path: Path | None = None,
    subject: str | None = None,
) -> pd.DataFrame:
    """Run time-resolved decoding on an MNE epochs file and save metrics as CSV."""
    epochs, metadata = _load_epochs_and_metadata(epochs_path, metadata_csv)
    decoder_name = normalize_decoder_name(decoder)
    emission_modes = list(EMISSION_MODE_CHOICES) if emission_mode == "both" else [normalize_emission_mode(emission_mode)]

    if label_column not in metadata.columns:
        raise ValueError(f"Label column '{label_column}' not found in metadata.")
    if group_column is not None and group_column not in metadata.columns:
        raise ValueError(f"Group column '{group_column}' not found in metadata.")

    epochs = epochs.copy().pick(picks)
    if tmin is not None or tmax is not None:
        epochs.crop(tmin=tmin, tmax=tmax)

    raw_labels = metadata[label_column].to_numpy()
    keep = pd.notna(raw_labels)
    original_indices = np.arange(len(raw_labels))[keep]
    epochs = epochs[keep]
    raw_labels = raw_labels[keep]
    metadata = metadata.loc[keep].reset_index(drop=True)

    encoder = LabelEncoder()
    labels = encoder.fit_transform(raw_labels)
    groups = metadata[group_column].to_numpy() if group_column else None

    data = epochs.get_data(copy=False)
    classes = np.arange(len(encoder.classes_))
    rows = []
    calibration_rows = []
    observation_tables: list[ProbabilityObservationTable] = []

    for start, stop, center in time_windows(epochs.times, window_ms=window_ms, step_ms=step_ms):
        features = data[:, :, start:stop].reshape(len(labels), -1)
        for fold, (train_idx, test_idx) in enumerate(make_cross_validator(labels, groups, n_splits)):
            for current_emission_mode in emission_modes:
                backend = make_decoder_backend(decoder_name, max_iter=max_iter, emission_mode=current_emission_mode)
                fit_result = backend.fit_predict(features, labels, train_idx, test_idx)
                probabilities = fit_result.probabilities
                predictions = fit_result.predictions
                test_labels = fit_result.test_labels

                rows.append(
                    _add_subject(
                        {
                            "fold": fold,
                            "decoder": decoder_name,
                            "emission_mode": current_emission_mode,
                            "time": center,
                            "window_start": float(epochs.times[start]),
                            "window_stop": float(epochs.times[stop - 1]),
                            "accuracy": accuracy_score(test_labels, predictions),
                            "log_loss": log_loss(test_labels, probabilities, labels=classes),
                            "brier": brier_score_multiclass(probabilities, test_labels),
                            "ece": expected_calibration_error(probabilities, test_labels),
                            "n_train": len(train_idx),
                            "n_test": len(test_idx),
                            "n_classes": len(classes),
                            "class_names": "|".join(map(str, encoder.classes_)),
                        },
                        subject,
                    )
                )
                if calibration_out_path is not None:
                    for bin_row in reliability_bins(probabilities, test_labels, n_bins=calibration_bins):
                        calibration_rows.append(
                            _add_subject(
                                {
                                    "fold": fold,
                                    "decoder": decoder_name,
                                    "emission_mode": current_emission_mode,
                                    "time": center,
                                    "window_start": float(epochs.times[start]),
                                    "window_stop": float(epochs.times[stop - 1]),
                                    **bin_row,
                                },
                                subject,
                            )
                        )
                if observation_out_path is not None:
                    observation_tables.append(
                        backend.build_observation_table(
                            fit_result,
                            class_names=encoder.classes_,
                            test_indices=test_idx,
                            fold=fold,
                            time=center,
                            original_indices=original_indices,
                            subject=subject,
                            group_values=groups,
                            seed=13,
                            train_time=center,
                            window_start=float(epochs.times[start]),
                            window_stop=float(epochs.times[stop - 1]),
                            calibration_fold="",
                        )
                    )

    results = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_path, index=False)
    if calibration_out_path is not None:
        calibration_out_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(calibration_rows).to_csv(calibration_out_path, index=False)
    if observation_out_path is not None:
        observation_out_path.parent.mkdir(parents=True, exist_ok=True)
        ProbabilityObservationTable.concat(observation_tables).standardized(defaults={"backend": "sklearn"}).frame.to_csv(
            observation_out_path,
            index=False,
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run calibrated time-resolved decoding on an MNE Epochs FIF file."
    )
    parser.add_argument("--epochs", type=Path, required=True)
    parser.add_argument("--label-column", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--metadata-csv", type=Path)
    parser.add_argument("--group-column")
    parser.add_argument("--picks", default="data")
    parser.add_argument("--tmin", type=float)
    parser.add_argument("--tmax", type=float)
    parser.add_argument("--window-ms", type=float, default=20.0)
    parser.add_argument("--step-ms", type=float, default=10.0)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--decoder", choices=DECODER_CHOICES, default="logistic")
    parser.add_argument("--emission-mode", choices=EMISSION_RUN_CHOICES, default="calibrated")
    parser.add_argument("--calibration-out", type=Path)
    parser.add_argument("--calibration-bins", type=int, default=10)
    parser.add_argument("--observations-out", type=Path, help="Optional held-out trial/time probability observation CSV.")
    parser.add_argument("--subject", help="Optional subject identifier to include in output CSVs.")
    args = parser.parse_args()

    results = run_time_resolved_decode(
        epochs_path=args.epochs,
        metadata_csv=args.metadata_csv,
        label_column=args.label_column,
        group_column=args.group_column,
        out_path=args.out,
        picks=args.picks,
        tmin=args.tmin,
        tmax=args.tmax,
        window_ms=args.window_ms,
        step_ms=args.step_ms,
        n_splits=args.n_splits,
        max_iter=args.max_iter,
        decoder=args.decoder,
        emission_mode=args.emission_mode,
        calibration_out_path=args.calibration_out,
        calibration_bins=args.calibration_bins,
        observation_out_path=args.observations_out,
        subject=args.subject,
    )
    print(f"Wrote {args.out}")
    if args.observations_out is not None:
        print(f"Wrote probability observations: {args.observations_out}")
    for emission_mode_name, summary in results.groupby("emission_mode", sort=True):
        time_summary = summary.groupby("time")[["accuracy", "log_loss", "brier", "ece"]].mean()
        best_time = time_summary["accuracy"].idxmax()
        print(f"Best {emission_mode_name} mean accuracy: {time_summary.loc[best_time, 'accuracy']:.3f} at {best_time:.3f}s")


if __name__ == "__main__":
    main()
