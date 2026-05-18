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
MODEL_SELECTION_CHOICES = ("none", "regularized")
SELECTION_METRIC_CHOICES = ("log_loss", "brier", "accuracy")
REGULARIZATION_C_GRID = (0.01, 0.1, 1.0, 10.0)
LDA_SHRINKAGE_GRID = (None, "auto", 0.1, 0.5, 0.9)
PCA_COMPONENT_GRID = (None, 0.8, 0.9, 0.95)


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


def _normalize_model_selection(name: str) -> str:
    normalized = name.lower().replace("-", "_")
    if normalized not in MODEL_SELECTION_CHOICES:
        raise ValueError(f"Unknown model selection mode '{name}'. Available modes: {', '.join(MODEL_SELECTION_CHOICES)}.")
    return normalized


def _normalize_selection_metric(name: str) -> str:
    normalized = name.lower().replace("-", "_")
    if normalized not in SELECTION_METRIC_CHOICES:
        raise ValueError(f"Unknown selection metric '{name}'. Available metrics: {', '.join(SELECTION_METRIC_CHOICES)}.")
    return normalized


def _format_optional(value: object) -> object:
    return "" if value is None else value


def _unique_values(values: list[object]) -> list[object]:
    unique = []
    keys = set()
    for value in values:
        key = repr(value)
        if key not in keys:
            unique.append(value)
            keys.add(key)
    return unique


def _pca_candidates(feature_preprocessor: str, pca_components: int | float | None) -> list[int | float | None]:
    if feature_preprocessor == "none":
        return [None]
    return _unique_values([pca_components, *PCA_COMPONENT_GRID])


def _decoder_config(
    *,
    decoder: str,
    emission_mode: str,
    feature_preprocessor: str,
    pca_components: int | float | None,
    regularization_c: float = 1.0,
    lda_shrinkage: str | float | None = None,
    calibration_method: str = "sigmoid",
    model_selection: str = "none",
    selection_metric: str = "",
    selection_score: float = float("nan"),
    inner_splits: int | str = "",
) -> dict[str, object]:
    return {
        "decoder": decoder,
        "emission_mode": emission_mode,
        "feature_preprocessor": feature_preprocessor,
        "pca_components": pca_components,
        "regularization_c": float(regularization_c),
        "lda_shrinkage": lda_shrinkage,
        "calibration_method": calibration_method,
        "model_selection": model_selection,
        "selection_metric": selection_metric,
        "selection_score": selection_score,
        "inner_splits": inner_splits,
    }


def _model_selection_candidates(
    *,
    decoder: str,
    emission_mode: str,
    feature_preprocessor: str,
    pca_components: int | float | None,
) -> list[dict[str, object]]:
    pca_values = _pca_candidates(feature_preprocessor, pca_components)
    candidates = []
    if decoder == "logistic":
        for regularization_c in REGULARIZATION_C_GRID:
            for pca_value in pca_values:
                candidates.append(
                    _decoder_config(
                        decoder=decoder,
                        emission_mode=emission_mode,
                        feature_preprocessor=feature_preprocessor,
                        pca_components=pca_value,
                        regularization_c=regularization_c,
                    )
                )
    elif decoder == "lda":
        for lda_shrinkage in LDA_SHRINKAGE_GRID:
            for pca_value in pca_values:
                candidates.append(
                    _decoder_config(
                        decoder=decoder,
                        emission_mode=emission_mode,
                        feature_preprocessor=feature_preprocessor,
                        pca_components=pca_value,
                        lda_shrinkage=lda_shrinkage,
                    )
                )
    elif decoder == "linear_svm":
        calibration_methods = ("sigmoid", "isotonic") if emission_mode == "calibrated" else ("sigmoid",)
        for regularization_c in REGULARIZATION_C_GRID:
            for calibration_method in calibration_methods:
                for pca_value in pca_values:
                    candidates.append(
                        _decoder_config(
                            decoder=decoder,
                            emission_mode=emission_mode,
                            feature_preprocessor=feature_preprocessor,
                            pca_components=pca_value,
                            regularization_c=regularization_c,
                            calibration_method=calibration_method,
                        )
                    )
    else:  # Defensive guard after normalize_decoder_name.
        raise ValueError(f"Unknown decoder '{decoder}'.")
    return candidates


def _effective_inner_splits(labels: np.ndarray, groups: np.ndarray | None, requested_splits: int) -> int:
    if requested_splits < 2:
        raise ValueError("inner_splits must be at least 2 when model selection is enabled.")
    _, class_counts = np.unique(labels, return_counts=True)
    effective = min(int(requested_splits), int(np.min(class_counts)))
    if groups is not None:
        effective = min(effective, len(np.unique(groups)))
    return effective if effective >= 2 else 0


def _score_candidate_probabilities(
    probabilities: np.ndarray,
    labels: np.ndarray,
    *,
    classes: np.ndarray,
    selection_metric: str,
) -> float:
    if selection_metric == "accuracy":
        return float(accuracy_score(labels, probabilities.argmax(axis=1)))
    if selection_metric == "brier":
        return -brier_score_multiclass(probabilities, labels)
    if selection_metric == "log_loss":
        return -float(log_loss(labels, probabilities, labels=classes))
    raise ValueError(f"Unknown selection metric '{selection_metric}'.")


def _fit_decoder_from_config(config: dict[str, object], features: np.ndarray, labels: np.ndarray, *, max_iter: int):
    model = make_decoder(
        str(config["decoder"]),
        max_iter=max_iter,
        emission_mode=str(config["emission_mode"]),
        feature_preprocessor=str(config["feature_preprocessor"]),
        pca_components=config["pca_components"],
        regularization_c=config["regularization_c"],
        lda_shrinkage=config["lda_shrinkage"],
        calibration_method=str(config["calibration_method"]),
    )
    model.fit(features, labels)
    return model


def _select_decoder_config(
    features: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray | None,
    *,
    decoder: str,
    emission_mode: str,
    feature_preprocessor: str,
    pca_components: int | float | None,
    max_iter: int,
    model_selection: str,
    selection_metric: str,
    inner_splits: int,
) -> dict[str, object]:
    base_config = _decoder_config(
        decoder=decoder,
        emission_mode=emission_mode,
        feature_preprocessor=feature_preprocessor,
        pca_components=pca_components,
        model_selection=model_selection,
        selection_metric="" if model_selection == "none" else selection_metric,
    )
    if model_selection == "none":
        return base_config

    effective_splits = _effective_inner_splits(labels, groups, inner_splits)
    if effective_splits < 2:
        return {**base_config, "inner_splits": 0}

    classes = np.arange(len(np.unique(labels)))
    best_config: dict[str, object] | None = None
    best_score = -np.inf
    for candidate in _model_selection_candidates(
        decoder=decoder,
        emission_mode=emission_mode,
        feature_preprocessor=feature_preprocessor,
        pca_components=pca_components,
    ):
        fold_scores = []
        try:
            inner_cv = make_cross_validator(labels, groups, effective_splits)
            for inner_train, inner_valid in inner_cv:
                model = _fit_decoder_from_config(candidate, features[inner_train], labels[inner_train], max_iter=max_iter)
                probabilities = predict_emission_probabilities(
                    model,
                    features[inner_valid],
                    emission_mode=emission_mode,
                )
                fold_scores.append(
                    _score_candidate_probabilities(
                        probabilities,
                        labels[inner_valid],
                        classes=classes,
                        selection_metric=selection_metric,
                    )
                )
        except (FloatingPointError, ValueError, np.linalg.LinAlgError):
            continue
        if not fold_scores:
            continue
        mean_score = float(np.mean(fold_scores))
        if mean_score > best_score:
            best_score = mean_score
            best_config = candidate
    if best_config is None:
        return {**base_config, "inner_splits": effective_splits}
    return {
        **best_config,
        "model_selection": model_selection,
        "selection_metric": selection_metric,
        "selection_score": best_score,
        "inner_splits": effective_splits,
    }


def _preprocessing_hash(
    *,
    picks: str,
    tmin: float | None,
    tmax: float | None,
    window_ms: float,
    step_ms: float,
    config: dict[str, object],
) -> str:
    return stable_hash(
        {
            "picks": picks,
            "tmin": tmin,
            "tmax": tmax,
            "window_ms": window_ms,
            "step_ms": step_ms,
            "feature_preprocessor": config["feature_preprocessor"],
            "pca_components": config["pca_components"],
        }
    )


def _model_hash(*, config: dict[str, object], max_iter: int) -> str:
    return stable_hash(
        {
            "backend": "sklearn",
            "decoder": config["decoder"],
            "emission_mode": config["emission_mode"],
            "max_iter": max_iter,
            "feature_preprocessor": config["feature_preprocessor"],
            "pca_components": config["pca_components"],
            "regularization_c": config["regularization_c"],
            "lda_shrinkage": config["lda_shrinkage"],
            "calibration_method": config["calibration_method"],
            "model_selection": config["model_selection"],
            "selection_metric": config["selection_metric"],
            "inner_splits": config["inner_splits"],
        }
    )


def _config_result_columns(config: dict[str, object]) -> dict[str, object]:
    return {
        "decoder": config["decoder"],
        "emission_mode": config["emission_mode"],
        "feature_preprocessor": config["feature_preprocessor"],
        "pca_components": _format_optional(config["pca_components"]),
        "regularization_c": config["regularization_c"],
        "lda_shrinkage": _format_optional(config["lda_shrinkage"]),
        "calibration_method": config["calibration_method"],
        "model_selection": config["model_selection"],
        "selection_metric": config["selection_metric"],
        "selection_score": config["selection_score"],
        "inner_splits": config["inner_splits"],
    }


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
    model_selection: str = "none",
    inner_splits: int = 3,
    selection_metric: str = "log_loss",
    calibration_out_path: Path | None = None,
    calibration_bins: int = 10,
    observation_out_path: Path | None = None,
    subject: str | None = None,
) -> pd.DataFrame:
    """Run time-resolved decoding on an MNE epochs file and save metrics as CSV."""
    epochs, metadata = _load_epochs_and_metadata(epochs_path, metadata_csv)
    decoder_name = normalize_decoder_name(decoder)
    emission_modes = list(EMISSION_MODE_CHOICES) if emission_mode == "both" else [normalize_emission_mode(emission_mode)]
    feature_preprocessor_name = normalize_feature_preprocessor(feature_preprocessor)
    if feature_preprocessor_name == "none" and pca_components is not None:
        raise ValueError("pca_components can only be set when feature_preprocessor is 'pca' or 'pca_whiten'.")
    pca_components_value = normalize_pca_components(pca_components) if feature_preprocessor_name != "none" else None
    model_selection_name = _normalize_model_selection(model_selection)
    selection_metric_name = _normalize_selection_metric(selection_metric)
    if model_selection_name != "none" and inner_splits < 2:
        raise ValueError("inner_splits must be at least 2 when model selection is enabled.")

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

    data = epochs.get_data(copy=False)
    classes = np.arange(len(encoder.classes_))
    rows = []
    calibration_rows = []
    observation_rows = []

    for start, stop, center in time_windows(epochs.times, window_ms=window_ms, step_ms=step_ms):
        features = data[:, :, start:stop].reshape(len(labels), -1)
        for fold, (train_idx, test_idx) in enumerate(make_cross_validator(labels, groups, n_splits)):
            train_groups = None if groups is None else groups[train_idx]
            for current_emission_mode in emission_modes:
                selected_config = _select_decoder_config(
                    features[train_idx],
                    labels[train_idx],
                    train_groups,
                    decoder=decoder_name,
                    emission_mode=current_emission_mode,
                    feature_preprocessor=feature_preprocessor_name,
                    pca_components=pca_components_value,
                    max_iter=max_iter,
                    model_selection=model_selection_name,
                    selection_metric=selection_metric_name,
                    inner_splits=inner_splits,
                )
                model = _fit_decoder_from_config(selected_config, features[train_idx], labels[train_idx], max_iter=max_iter)

                probabilities = predict_emission_probabilities(
                    model,
                    features[test_idx],
                    emission_mode=current_emission_mode,
                )
                predictions = probabilities.argmax(axis=1)
                test_labels = labels[test_idx]
                current_preprocessing_hash = _preprocessing_hash(
                    picks=picks,
                    tmin=tmin,
                    tmax=tmax,
                    window_ms=window_ms,
                    step_ms=step_ms,
                    config=selected_config,
                )
                current_model_hash = _model_hash(config=selected_config, max_iter=max_iter)
                config_columns = _config_result_columns(selected_config)

                rows.append(
                    _add_subject(
                        {
                            "fold": fold,
                            **config_columns,
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
                                    **config_columns,
                                    "time": center,
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
                            **config_columns,
                            "backend": "sklearn",
                            "train_time": center,
                            "test_time": center,
                            "time": center,
                            "window_start": float(epochs.times[start]),
                            "window_stop": float(epochs.times[stop - 1]),
                            "sample_index": int(original_indices[filtered_index]),
                            "sequence_id": int(original_indices[filtered_index]),
                            "session": "" if session_values is None else session_values[filtered_index],
                            "true_label": true_label,
                            "true_class": str(encoder.classes_[true_label]),
                            "predicted_label": predicted_label,
                            "predicted_class": str(encoder.classes_[predicted_label]),
                            "probability_true_class": float(probabilities[local_position, true_label]),
                            "confidence": float(probabilities[local_position].max()),
                            "is_correct": bool(predicted_label == true_label),
                            "calibration_fold": "",
                            "preprocessing_hash": current_preprocessing_hash,
                            "model_hash": current_model_hash,
                        }
                        if group_column is not None:
                            observation["group"] = groups[filtered_index]
                        for class_index, class_name in enumerate(encoder.classes_):
                            observation[f"class_{class_index}"] = str(class_name)
                            observation[f"prob_class_{class_index}"] = float(probabilities[local_position, class_index])
                        observation_rows.append(_add_subject(observation, subject))

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
    parser.add_argument(
        "--model-selection",
        choices=MODEL_SELECTION_CHOICES,
        default="none",
        help="Optional inner-CV hyperparameter selection within each outer train fold.",
    )
    parser.add_argument("--inner-splits", type=int, default=3, help="Number of inner CV splits used when --model-selection is enabled.")
    parser.add_argument(
        "--selection-metric",
        choices=SELECTION_METRIC_CHOICES,
        default="log_loss",
        help="Inner-CV metric to optimize when --model-selection is enabled.",
    )
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
        feature_preprocessor=args.feature_preprocessor,
        pca_components=args.pca_components,
        model_selection=args.model_selection,
        inner_splits=args.inner_splits,
        selection_metric=args.selection_metric,
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
