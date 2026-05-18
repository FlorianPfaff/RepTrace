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
    FEATURE_PREPROCESSOR_CHOICES,
    make_cross_validator,
    make_decoder,
    normalize_decoder_name,
    normalize_emission_mode,
    normalize_feature_preprocessor,
    normalize_pca_components,
    predict_emission_probabilities,
    time_windows,
)
from reptrace.metrics import brier_score_multiclass, expected_calibration_error, reliability_bins
from reptrace.observations import ProbabilityObservationTable, stable_hash

EMISSION_RUN_CHOICES = (*EMISSION_MODE_CHOICES, "both")
FEATURE_PREPROCESSOR_RUN_CHOICES = (*FEATURE_PREPROCESSOR_CHOICES, "pca-whiten")
TimeWindow = tuple[int, int, float]
TemporalTrainWindow = tuple[float, float]


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


def _normalize_temporal_train_window(
    temporal_train_window: tuple[float, float] | list[float] | None,
) -> TemporalTrainWindow | None:
    if temporal_train_window is None:
        return None
    if len(temporal_train_window) != 2:
        raise ValueError("temporal_train_window must contain exactly two times: start and stop.")
    start, stop = map(float, temporal_train_window)
    if stop < start:
        raise ValueError("temporal_train_window stop must be greater than or equal to start.")
    return start, stop


def _select_temporal_train_windows(
    windows: list[TimeWindow],
    temporal_train_window: tuple[float, float] | list[float] | None,
) -> list[TimeWindow] | None:
    normalized = _normalize_temporal_train_window(temporal_train_window)
    if normalized is None:
        return None
    train_start, train_stop = normalized
    selected = [window for window in windows if train_start <= window[2] <= train_stop]
    if selected:
        return selected

    available_centers = [window[2] for window in windows]
    if not available_centers:
        raise ValueError("No time windows are available for temporal train-window selection.")
    raise ValueError(
        "No time-window centers fall inside temporal_train_window "
        f"[{train_start}, {train_stop}]. Available centers span "
        f"[{min(available_centers)}, {max(available_centers)}]."
    )


def _features_for_window(data: np.ndarray, window: TimeWindow) -> np.ndarray:
    start, stop, _center = window
    return data[:, :, start:stop].reshape(data.shape[0], -1)


def _probability_average(probability_sum: np.ndarray, n_models: int) -> np.ndarray:
    probabilities = probability_sum / float(n_models)
    row_sums = probabilities.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0.0):
        raise ValueError("Averaged probabilities must have positive row sums.")
    return probabilities / row_sums


def _train_window_summary(
    epochs: mne.Epochs,
    train_windows: list[TimeWindow],
) -> tuple[float, float, float]:
    return (
        float(np.mean([window[2] for window in train_windows])),
        float(min(epochs.times[window[0]] for window in train_windows)),
        float(max(epochs.times[window[1] - 1] for window in train_windows)),
    )


def _model_hash(
    *,
    decoder_name: str,
    emission_mode: str,
    max_iter: int,
    feature_preprocessor: str,
    pca_components: int | float | None,
    temporal_mode: str,
    temporal_train_window: TemporalTrainWindow | None,
    train_window_centers: list[float] | None = None,
) -> str:
    return stable_hash(
        {
            "backend": "sklearn",
            "decoder": decoder_name,
            "emission_mode": emission_mode,
            "max_iter": max_iter,
            "feature_preprocessor": feature_preprocessor,
            "pca_components": pca_components,
            "temporal_mode": temporal_mode,
            "temporal_train_window": temporal_train_window,
            "train_window_centers": train_window_centers,
        }
    )


def _append_decoded_outputs(
    *,
    rows: list[dict],
    calibration_rows: list[dict],
    observation_rows: list[dict],
    probabilities: np.ndarray,
    test_labels: np.ndarray,
    test_idx: np.ndarray,
    original_indices: np.ndarray,
    session_values: np.ndarray | None,
    groups: np.ndarray | None,
    group_column: str | None,
    classes: np.ndarray,
    class_names: np.ndarray,
    fold: int,
    n_train: int,
    decoder_name: str,
    emission_mode: str,
    feature_preprocessor_name: str,
    pca_components_value: int | float | None,
    time_window: TimeWindow,
    epochs: mne.Epochs,
    split_id: str,
    preprocessing_hash: str,
    model_hash: str,
    temporal_mode: str,
    train_time: float,
    train_window_start: float,
    train_window_stop: float,
    n_train_windows: int,
    calibration_out_path: Path | None,
    calibration_bins: int,
    observation_out_path: Path | None,
    subject: str | None,
) -> None:
    start, stop, center = time_window
    predictions = probabilities.argmax(axis=1)
    row = {
        "fold": fold,
        "decoder": decoder_name,
        "emission_mode": emission_mode,
        "feature_preprocessor": feature_preprocessor_name,
        "pca_components": "" if pca_components_value is None else pca_components_value,
        "temporal_mode": temporal_mode,
        "train_time": train_time,
        "time": center,
        "test_time": center,
        "train_window_start": train_window_start,
        "train_window_stop": train_window_stop,
        "n_train_windows": n_train_windows,
        "window_start": float(epochs.times[start]),
        "window_stop": float(epochs.times[stop - 1]),
        "accuracy": accuracy_score(test_labels, predictions),
        "log_loss": log_loss(test_labels, probabilities, labels=classes),
        "brier": brier_score_multiclass(probabilities, test_labels),
        "ece": expected_calibration_error(probabilities, test_labels),
        "n_train": n_train,
        "n_test": len(test_idx),
        "n_classes": len(classes),
        "class_names": "|".join(map(str, class_names)),
    }
    rows.append(_add_subject(row, subject))

    if calibration_out_path is not None:
        for bin_row in reliability_bins(probabilities, test_labels, n_bins=calibration_bins):
            calibration_rows.append(
                _add_subject(
                    {
                        "fold": fold,
                        "decoder": decoder_name,
                        "emission_mode": emission_mode,
                        "feature_preprocessor": feature_preprocessor_name,
                        "pca_components": "" if pca_components_value is None else pca_components_value,
                        "temporal_mode": temporal_mode,
                        "train_time": train_time,
                        "time": center,
                        "test_time": center,
                        "train_window_start": train_window_start,
                        "train_window_stop": train_window_stop,
                        "n_train_windows": n_train_windows,
                        "window_start": float(epochs.times[start]),
                        "window_stop": float(epochs.times[stop - 1]),
                        **bin_row,
                    },
                    subject,
                )
            )
    if observation_out_path is not None:
        for local_position, filtered_index in enumerate(test_idx):
            true_label = int(test_labels[local_position])
            predicted_label = int(predictions[local_position])
            observation = {
                "fold": fold,
                "split_id": split_id,
                "seed": 13,
                "decoder": decoder_name,
                "backend": "sklearn",
                "emission_mode": emission_mode,
                "feature_preprocessor": feature_preprocessor_name,
                "pca_components": "" if pca_components_value is None else pca_components_value,
                "temporal_mode": temporal_mode,
                "train_time": train_time,
                "test_time": center,
                "time": center,
                "train_window_start": train_window_start,
                "train_window_stop": train_window_stop,
                "n_train_windows": n_train_windows,
                "window_start": float(epochs.times[start]),
                "window_stop": float(epochs.times[stop - 1]),
                "sample_index": int(original_indices[filtered_index]),
                "sequence_id": int(original_indices[filtered_index]),
                "session": "" if session_values is None else session_values[filtered_index],
                "true_label": true_label,
                "true_class": str(class_names[true_label]),
                "predicted_label": predicted_label,
                "predicted_class": str(class_names[predicted_label]),
                "probability_true_class": float(probabilities[local_position, true_label]),
                "confidence": float(probabilities[local_position].max()),
                "is_correct": bool(predicted_label == true_label),
                "calibration_fold": "",
                "preprocessing_hash": preprocessing_hash,
                "model_hash": model_hash,
            }
            if group_column is not None:
                observation["group"] = groups[filtered_index] if groups is not None else ""
            for class_index, class_name in enumerate(class_names):
                observation[f"class_{class_index}"] = str(class_name)
                observation[f"prob_class_{class_index}"] = float(probabilities[local_position, class_index])
            observation_rows.append(_add_subject(observation, subject))


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
    feature_preprocessor: str = "none",
    pca_components: int | float | str | None = None,
    calibration_out_path: Path | None = None,
    calibration_bins: int = 10,
    observation_out_path: Path | None = None,
    subject: str | None = None,
    temporal_train_window: tuple[float, float] | None = None,
) -> pd.DataFrame:
    """Run time-resolved decoding on an MNE epochs file and save metrics as CSV.

    If ``temporal_train_window`` is set, models are trained on every decoding
    window whose center lies in that interval and are evaluated at every test
    time. The per-test-time probabilities are averaged across those train-time
    models, turning temporal generalization into a direct result-improving
    train-window ensemble. Without the option, the historical diagonal
    train-time == test-time decoding path is used.
    """
    epochs, metadata = _load_epochs_and_metadata(epochs_path, metadata_csv)
    decoder_name = normalize_decoder_name(decoder)
    emission_modes = list(EMISSION_MODE_CHOICES) if emission_mode == "both" else [normalize_emission_mode(emission_mode)]
    feature_preprocessor_name = normalize_feature_preprocessor(feature_preprocessor)
    if feature_preprocessor_name == "none" and pca_components is not None:
        raise ValueError("pca_components can only be set when feature_preprocessor is 'pca' or 'pca_whiten'.")
    pca_components_value = normalize_pca_components(pca_components) if feature_preprocessor_name != "none" else None
    normalized_temporal_train_window = _normalize_temporal_train_window(temporal_train_window)

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
    session_values = metadata["session"].to_numpy() if "session" in metadata.columns else groups
    splitter_name = "stratified-group-kfold" if groups is not None else "stratified-kfold"
    split_id = f"{splitter_name}-{n_splits}"
    temporal_mode = "same_time" if normalized_temporal_train_window is None else "train_window_ensemble"
    preprocessing_hash = stable_hash(
        {
            "picks": picks,
            "tmin": tmin,
            "tmax": tmax,
            "window_ms": window_ms,
            "step_ms": step_ms,
            "feature_preprocessor": feature_preprocessor_name,
            "pca_components": pca_components_value,
            "temporal_train_window": normalized_temporal_train_window,
        }
    )
    default_model_hash = _model_hash(
        decoder_name=decoder_name,
        emission_mode=emission_mode,
        max_iter=max_iter,
        feature_preprocessor=feature_preprocessor_name,
        pca_components=pca_components_value,
        temporal_mode=temporal_mode,
        temporal_train_window=normalized_temporal_train_window,
    )

    data = epochs.get_data(copy=False)
    classes = np.arange(len(encoder.classes_))
    rows = []
    calibration_rows = []
    observation_rows = []
    windows = time_windows(epochs.times, window_ms=window_ms, step_ms=step_ms)
    selected_train_windows = _select_temporal_train_windows(windows, normalized_temporal_train_window)
    splits = list(make_cross_validator(labels, groups, n_splits))

    if selected_train_windows is None:
        for time_window in windows:
            features = _features_for_window(data, time_window)
            start, stop, center = time_window
            for fold, (train_idx, test_idx) in enumerate(splits):
                test_labels = labels[test_idx]
                for current_emission_mode in emission_modes:
                    model = make_decoder(
                        decoder_name,
                        max_iter=max_iter,
                        emission_mode=current_emission_mode,
                        feature_preprocessor=feature_preprocessor_name,
                        pca_components=pca_components_value,
                    )
                    model.fit(features[train_idx], labels[train_idx])

                    probabilities = predict_emission_probabilities(
                        model,
                        features[test_idx],
                        emission_mode=current_emission_mode,
                    )
                    current_model_hash = _model_hash(
                        decoder_name=decoder_name,
                        emission_mode=current_emission_mode,
                        max_iter=max_iter,
                        feature_preprocessor=feature_preprocessor_name,
                        pca_components=pca_components_value,
                        temporal_mode=temporal_mode,
                        temporal_train_window=None,
                        train_window_centers=[center],
                    )
                    _append_decoded_outputs(
                        rows=rows,
                        calibration_rows=calibration_rows,
                        observation_rows=observation_rows,
                        probabilities=probabilities,
                        test_labels=test_labels,
                        test_idx=test_idx,
                        original_indices=original_indices,
                        session_values=session_values,
                        groups=groups,
                        group_column=group_column,
                        classes=classes,
                        class_names=encoder.classes_,
                        fold=fold,
                        n_train=len(train_idx),
                        decoder_name=decoder_name,
                        emission_mode=current_emission_mode,
                        feature_preprocessor_name=feature_preprocessor_name,
                        pca_components_value=pca_components_value,
                        time_window=time_window,
                        epochs=epochs,
                        split_id=split_id,
                        preprocessing_hash=preprocessing_hash,
                        model_hash=current_model_hash,
                        temporal_mode=temporal_mode,
                        train_time=center,
                        train_window_start=float(epochs.times[start]),
                        train_window_stop=float(epochs.times[stop - 1]),
                        n_train_windows=1,
                        calibration_out_path=calibration_out_path,
                        calibration_bins=calibration_bins,
                        observation_out_path=observation_out_path,
                        subject=subject,
                    )
    else:
        feature_cache = {time_window: _features_for_window(data, time_window) for time_window in windows}
        train_time, train_window_start, train_window_stop = _train_window_summary(epochs, selected_train_windows)
        train_window_centers = [window[2] for window in selected_train_windows]
        for fold, (train_idx, test_idx) in enumerate(splits):
            test_labels = labels[test_idx]
            for current_emission_mode in emission_modes:
                probability_sums = {
                    time_window: np.zeros((len(test_idx), len(classes)), dtype=float)
                    for time_window in windows
                }
                for train_window in selected_train_windows:
                    train_features = feature_cache[train_window]
                    model = make_decoder(
                        decoder_name,
                        max_iter=max_iter,
                        emission_mode=current_emission_mode,
                        feature_preprocessor=feature_preprocessor_name,
                        pca_components=pca_components_value,
                    )
                    model.fit(train_features[train_idx], labels[train_idx])
                    for test_window in windows:
                        probability_sums[test_window] += predict_emission_probabilities(
                            model,
                            feature_cache[test_window][test_idx],
                            emission_mode=current_emission_mode,
                        )

                current_model_hash = _model_hash(
                    decoder_name=decoder_name,
                    emission_mode=current_emission_mode,
                    max_iter=max_iter,
                    feature_preprocessor=feature_preprocessor_name,
                    pca_components=pca_components_value,
                    temporal_mode=temporal_mode,
                    temporal_train_window=normalized_temporal_train_window,
                    train_window_centers=train_window_centers,
                )
                for test_window in windows:
                    probabilities = _probability_average(probability_sums[test_window], len(selected_train_windows))
                    _append_decoded_outputs(
                        rows=rows,
                        calibration_rows=calibration_rows,
                        observation_rows=observation_rows,
                        probabilities=probabilities,
                        test_labels=test_labels,
                        test_idx=test_idx,
                        original_indices=original_indices,
                        session_values=session_values,
                        groups=groups,
                        group_column=group_column,
                        classes=classes,
                        class_names=encoder.classes_,
                        fold=fold,
                        n_train=len(train_idx),
                        decoder_name=decoder_name,
                        emission_mode=current_emission_mode,
                        feature_preprocessor_name=feature_preprocessor_name,
                        pca_components_value=pca_components_value,
                        time_window=test_window,
                        epochs=epochs,
                        split_id=split_id,
                        preprocessing_hash=preprocessing_hash,
                        model_hash=current_model_hash,
                        temporal_mode=temporal_mode,
                        train_time=train_time,
                        train_window_start=train_window_start,
                        train_window_stop=train_window_stop,
                        n_train_windows=len(selected_train_windows),
                        calibration_out_path=calibration_out_path,
                        calibration_bins=calibration_bins,
                        observation_out_path=observation_out_path,
                        subject=subject,
                    )

    results = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_path, index=False)
    if calibration_out_path is not None:
        calibration_out_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(calibration_rows).to_csv(calibration_out_path, index=False)
    if observation_out_path is not None:
        ProbabilityObservationTable(pd.DataFrame(observation_rows)).standardized(
            defaults={
                "backend": "sklearn",
                "split_id": split_id,
                "seed": 13,
                "calibration_fold": "",
                "preprocessing_hash": preprocessing_hash,
                "model_hash": default_model_hash,
            }
        ).to_csv(observation_out_path)
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
    parser.add_argument("--feature-preprocessor", choices=FEATURE_PREPROCESSOR_RUN_CHOICES, default="none")
    parser.add_argument(
        "--pca-components",
        help="PCA component count or explained-variance fraction. Only valid with --feature-preprocessor pca or pca-whiten.",
    )
    parser.add_argument("--calibration-out", type=Path)
    parser.add_argument("--calibration-bins", type=int, default=10)
    parser.add_argument("--observations-out", type=Path, help="Optional held-out trial/time probability observation CSV.")
    parser.add_argument("--subject", help="Optional subject identifier to include in output CSVs.")
    parser.add_argument(
        "--temporal-train-window",
        nargs=2,
        type=float,
        metavar=("START", "STOP"),
        help=(
            "Train one model per time-window center in START..STOP seconds, "
            "evaluate each model at every test time, and average probabilities."
        ),
    )
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
        feature_preprocessor=args.feature_preprocessor,
        pca_components=args.pca_components,
        calibration_out_path=args.calibration_out,
        calibration_bins=args.calibration_bins,
        observation_out_path=args.observations_out,
        subject=args.subject,
        temporal_train_window=tuple(args.temporal_train_window) if args.temporal_train_window is not None else None,
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
