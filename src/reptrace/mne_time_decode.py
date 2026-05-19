from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import LabelEncoder

from reptrace.decoding import (
    DECODER_CLI_CHOICES,
    EMISSION_MODE_CHOICES,
    FEATURE_PREPROCESSOR_CHOICES,
    TUNING_SCORING_CHOICES,
    make_calibration_cross_validator,
    make_cross_validator,
    make_decoder,
    make_tuning_cross_validator,
    normalize_decoder_name,
    normalize_emission_mode,
    normalize_feature_preprocessor,
    normalize_pca_components,
    normalize_tuning_scoring,
    parse_c_grid,
    parse_pca_components_grid,
    predict_emission_probabilities,
    time_windows,
)
from reptrace.metrics import brier_score_multiclass, expected_calibration_error, reliability_bins
from reptrace.observations import ProbabilityObservationTable, stable_hash

EMISSION_RUN_CHOICES = (*EMISSION_MODE_CHOICES, "both")
FEATURE_PREPROCESSOR_RUN_CHOICES = (*FEATURE_PREPROCESSOR_CHOICES, "pca-whiten")
RESULT_SELECTION_METRIC_CHOICES = ("accuracy", "log_loss", "brier", "ece")
RESULT_SELECTION_MINIMIZE_METRICS = {"log_loss", "brier", "ece"}
TEMPORAL_SELECTION_SCORING_CHOICES = ("accuracy", "neg_log_loss", "neg_brier")
DEFAULT_TEMPORAL_SELECTION_N_WINDOWS = 3
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


def _best_params_json(models) -> str:
    if isinstance(models, Sequence) and not isinstance(models, (str, bytes)):
        best_params = [getattr(model, "best_params_", None) for model in models]
        best_params = [params for params in best_params if params is not None]
    else:
        best_params = getattr(models, "best_params_", None)
    return "" if not best_params else json.dumps(best_params, sort_keys=True, default=str, separators=(",", ":"))


def _best_scores(models) -> list[float]:
    if isinstance(models, Sequence) and not isinstance(models, (str, bytes)):
        return [float(model.best_score_) for model in models if hasattr(model, "best_score_")]
    if hasattr(models, "best_score_"):
        return [float(models.best_score_)]
    return []


def _tuning_metadata(
    models,
    *,
    tune_hyperparameters: bool,
    tuning_cv_splits: int,
    tuning_scoring: str,
    tuning_c_grid: Sequence[float],
    tuning_pca_components_grid: Sequence[int | float | None] | None = None,
) -> dict[str, object]:
    metadata: dict[str, object] = {
        "tuned_hyperparameters": bool(tune_hyperparameters),
        "tuning_cv_splits": int(tuning_cv_splits) if tune_hyperparameters else "",
        "tuning_scoring": tuning_scoring if tune_hyperparameters else "",
        "tuning_c_grid": "|".join(str(value) for value in tuning_c_grid) if tune_hyperparameters else "",
        "tuning_pca_components_grid": (
            "|".join("none" if value is None else str(value) for value in tuning_pca_components_grid)
            if tune_hyperparameters and tuning_pca_components_grid is not None
            else ""
        ),
        "best_params": "",
    }
    if not tune_hyperparameters:
        return metadata
    metadata["best_params"] = _best_params_json(models)
    scores = _best_scores(models)
    if len(scores) == 1:
        metadata["best_score"] = scores[0]
    elif scores:
        metadata["best_score"] = float(np.mean(scores))
        metadata["best_scores"] = json.dumps(scores, separators=(",", ":"))
    return metadata


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


def _normalize_temporal_selection_scoring(scoring: str) -> str:
    """Normalize the objective used for temporal train-window selection."""
    normalized = scoring.lower().replace("-", "_")
    aliases = {
        "logloss": "neg_log_loss",
        "log_loss": "neg_log_loss",
        "negative_log_loss": "neg_log_loss",
        "brier": "neg_brier",
        "brier_score": "neg_brier",
        "negative_brier": "neg_brier",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in TEMPORAL_SELECTION_SCORING_CHOICES:
        raise ValueError(
            f"Unknown temporal selection scoring '{scoring}'. Available values: "
            f"{', '.join(TEMPORAL_SELECTION_SCORING_CHOICES)}."
        )
    return normalized


def _score_temporal_selection(
    probabilities: np.ndarray,
    labels: np.ndarray,
    *,
    classes: np.ndarray,
    scoring: str,
) -> float:
    """Return a maximized score for temporal train-window selection."""
    if scoring == "accuracy":
        return float(accuracy_score(labels, probabilities.argmax(axis=1)))
    if scoring == "neg_log_loss":
        return float(-log_loss(labels, probabilities, labels=classes))
    if scoring == "neg_brier":
        return float(-brier_score_multiclass(probabilities, labels))
    raise ValueError(f"Unsupported temporal selection scoring '{scoring}'.")


def _neighboring_temporal_test_windows(
    windows: list[TimeWindow],
    center: float,
    radius: float,
) -> list[TimeWindow]:
    """Select test windows used to assess local temporal generalization."""
    if radius < 0.0:
        raise ValueError("temporal_selection_radius must be non-negative.")
    selected = [window for window in windows if abs(window[2] - center) <= radius + np.finfo(float).eps]
    if selected:
        return selected
    return [min(windows, key=lambda window: abs(window[2] - center))]


def _select_stable_temporal_train_windows(
    *,
    feature_cache: dict[TimeWindow, np.ndarray],
    windows: list[TimeWindow],
    labels: np.ndarray,
    groups: np.ndarray | None,
    train_idx: np.ndarray,
    classes: np.ndarray,
    decoder_name: str,
    max_iter: int,
    emission_mode: str,
    feature_preprocessor_name: str,
    pca_components_value: int | float | None,
    temporal_selection_cv_splits: int,
    temporal_selection_scoring: str,
    temporal_selection_radius: float,
    temporal_selection_n_windows: int,
) -> tuple[list[TimeWindow], float]:
    """Pick a contiguous train-time region using inner temporal generalization.

    Candidate train windows are scored only on the outer training split. For each
    candidate center, an inner CV decoder is trained at that time and evaluated
    on nearby test-time windows. A contiguous block with the highest mean score
    is then refit on the full outer training split by the caller.
    """
    if temporal_selection_n_windows < 1:
        raise ValueError("temporal_selection_n_windows must be at least 1.")
    if not windows:
        raise ValueError("No time windows are available for temporal train-window selection.")

    train_idx = np.asarray(train_idx)
    inner_splits = make_tuning_cross_validator(
        labels[train_idx],
        None if groups is None else groups[train_idx],
        temporal_selection_cv_splits,
    )
    window_scores: list[float] = []
    for train_window in windows:
        train_features = feature_cache[train_window]
        test_windows = _neighboring_temporal_test_windows(windows, train_window[2], temporal_selection_radius)
        candidate_scores: list[float] = []
        for inner_train_rel, inner_val_rel in inner_splits:
            inner_train_idx = train_idx[inner_train_rel]
            inner_val_idx = train_idx[inner_val_rel]
            model = make_decoder(
                decoder_name,
                max_iter=max_iter,
                emission_mode=emission_mode,
                feature_preprocessor=feature_preprocessor_name,
                pca_components=pca_components_value,
            )
            model.fit(train_features[inner_train_idx], labels[inner_train_idx])
            for test_window in test_windows:
                probabilities = predict_emission_probabilities(
                    model,
                    feature_cache[test_window][inner_val_idx],
                    emission_mode=emission_mode,
                )
                candidate_scores.append(
                    _score_temporal_selection(
                        probabilities,
                        labels[inner_val_idx],
                        classes=classes,
                        scoring=temporal_selection_scoring,
                    )
                )
        window_scores.append(float(np.mean(candidate_scores)))

    block_width = min(int(temporal_selection_n_windows), len(windows))
    block_scores = [
        float(np.mean(window_scores[start : start + block_width]))
        for start in range(0, len(windows) - block_width + 1)
    ]
    best_start = int(np.argmax(block_scores))
    return windows[best_start : best_start + block_width], block_scores[best_start]


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


def _best_time_by_metric(time_summary: pd.DataFrame, metric: str) -> float:
    """Return the best time index for a metric aggregated over folds."""
    if metric not in RESULT_SELECTION_METRIC_CHOICES:
        raise ValueError(f"Unknown selection metric '{metric}'.")
    if metric in RESULT_SELECTION_MINIMIZE_METRICS:
        return float(time_summary[metric].idxmin())
    return float(time_summary[metric].idxmax())


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
    select_temporal_train_window: bool = False,
    temporal_selection_cv_splits: int | None = None,
    temporal_selection_scoring: str | None = None,
    temporal_selection_radius: float | None = None,
    temporal_selection_n_windows: int | None = None,
    tune_hyperparameters: bool = False,
    tuning_cv_splits: int | None = None,
    tuning_scoring: str | None = None,
    tuning_c_grid: Sequence[float] | None = None,
    tuning_pca_components_grid: Sequence[int | float | None] | None = None,
    tuning_metadata: dict[str, object] | None = None,
    calibration_cv_splits: int | None = None,
    group_aware_calibration: bool = False,
) -> str:
    payload: dict[str, object] = {
        "backend": "sklearn",
        "decoder": decoder_name,
        "emission_mode": emission_mode,
        "max_iter": max_iter,
        "feature_preprocessor": feature_preprocessor,
        "pca_components": pca_components,
        "temporal_mode": temporal_mode,
        "temporal_train_window": temporal_train_window,
        "train_window_centers": train_window_centers,
        "select_temporal_train_window": select_temporal_train_window,
        "temporal_selection_cv_splits": temporal_selection_cv_splits,
        "temporal_selection_scoring": temporal_selection_scoring,
        "temporal_selection_radius": temporal_selection_radius,
        "temporal_selection_n_windows": temporal_selection_n_windows,
    }
    if decoder_name == "linear_svm" and emission_mode in {"calibrated", "both"}:
        payload["calibration_cv_splits"] = calibration_cv_splits
        payload["group_aware_calibration"] = group_aware_calibration
    if tune_hyperparameters:
        payload.update(
            {
                "tune_hyperparameters": True,
                "tuning_cv_splits": tuning_cv_splits,
                "tuning_scoring": tuning_scoring,
                "tuning_c_grid": tuple(tuning_c_grid or ()),
                "tuning_pca_components_grid": tuple(tuning_pca_components_grid or ()),
                "best_params": (tuning_metadata or {}).get("best_params", ""),
            }
        )
    return stable_hash(payload)


def _uses_group_aware_calibration(
    *,
    decoder_name: str,
    emission_mode: str,
    groups: np.ndarray | None,
    tune_hyperparameters: bool,
) -> bool:
    """Return whether the current fit can use group-disjoint SVM calibration."""
    return (
        decoder_name == "linear_svm"
        and emission_mode == "calibrated"
        and groups is not None
        and not tune_hyperparameters
    )


def _calibration_cv_for_decoder(
    *,
    decoder_name: str,
    emission_mode: str,
    labels: np.ndarray,
    groups: np.ndarray | None,
    n_splits: int,
    tune_hyperparameters: bool,
):
    """Return the calibration CV specification for a decoder fit."""
    if decoder_name != "linear_svm" or emission_mode != "calibrated":
        return 3
    if _uses_group_aware_calibration(
        decoder_name=decoder_name,
        emission_mode=emission_mode,
        groups=groups,
        tune_hyperparameters=tune_hyperparameters,
    ):
        return make_calibration_cross_validator(labels, groups, n_splits)
    return int(n_splits)


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
    temporal_train_window: TemporalTrainWindow | None,
    train_time: float,
    train_window_start: float,
    temporal_selection_score: float | None,
    train_window_stop: float,
    n_train_windows: int,
    calibration_out_path: Path | None,
    calibration_bins: int,
    observation_out_path: Path | None,
    subject: str | None,
    tuning_metadata: dict[str, object] | None = None,
) -> None:
    tuning_metadata = {} if tuning_metadata is None else tuning_metadata
    start, stop, center = time_window
    predictions = probabilities.argmax(axis=1)
    row = {
        "fold": fold,
        "decoder": decoder_name,
        "emission_mode": emission_mode,
        "feature_preprocessor": feature_preprocessor_name,
        "pca_components": "" if pca_components_value is None else pca_components_value,
        "temporal_mode": temporal_mode,
        "temporal_train_window_start": "" if temporal_train_window is None else temporal_train_window[0],
        "temporal_train_window_stop": "" if temporal_train_window is None else temporal_train_window[1],
        "train_time": train_time,
        "time": center,
        "test_time": center,
        "train_window_start": train_window_start,
        "train_window_stop": train_window_stop,
        "temporal_selection_score": "" if temporal_selection_score is None else temporal_selection_score,
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
    row.update(tuning_metadata)
    rows.append(_add_subject(row, subject))

    if calibration_out_path is not None:
        for bin_row in reliability_bins(probabilities, test_labels, n_bins=calibration_bins):
            calibration_row = {
                "fold": fold,
                "decoder": decoder_name,
                "emission_mode": emission_mode,
                "feature_preprocessor": feature_preprocessor_name,
                "pca_components": "" if pca_components_value is None else pca_components_value,
                "temporal_mode": temporal_mode,
                "temporal_train_window_start": "" if temporal_train_window is None else temporal_train_window[0],
                "temporal_train_window_stop": "" if temporal_train_window is None else temporal_train_window[1],
                "train_time": train_time,
                "time": center,
                "test_time": center,
                "train_window_start": train_window_start,
                "train_window_stop": train_window_stop,
                "temporal_selection_score": "" if temporal_selection_score is None else temporal_selection_score,
                "n_train_windows": n_train_windows,
                "window_start": float(epochs.times[start]),
                "window_stop": float(epochs.times[stop - 1]),
                **bin_row,
            }
            calibration_row.update(tuning_metadata)
            calibration_rows.append(_add_subject(calibration_row, subject))
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
                "temporal_train_window_start": "" if temporal_train_window is None else temporal_train_window[0],
                "temporal_train_window_stop": "" if temporal_train_window is None else temporal_train_window[1],
                "train_time": train_time,
                "test_time": center,
                "time": center,
                "train_window_start": train_window_start,
                "train_window_stop": train_window_stop,
                "temporal_selection_score": "" if temporal_selection_score is None else temporal_selection_score,
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
            observation.update(tuning_metadata)
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
    tune_hyperparameters: bool = False,
    tuning_cv_splits: int = 3,
    tuning_scoring: str = "accuracy",
    tuning_c_grid: Sequence[float] | str | None = None,
    tuning_pca_components_grid: Sequence[int | float | str | None] | str | None = None,
    calibration_cv_splits: int = 3,
    calibration_out_path: Path | None = None,
    calibration_bins: int = 10,
    observation_out_path: Path | None = None,
    subject: str | None = None,
    temporal_train_window: tuple[float, float] | None = None,
    select_temporal_train_window: bool = False,
    temporal_selection_cv_splits: int = 3,
    temporal_selection_scoring: str = "accuracy",
    temporal_selection_radius: float | None = None,
    temporal_selection_n_windows: int = DEFAULT_TEMPORAL_SELECTION_N_WINDOWS,
) -> pd.DataFrame:
    """Run time-resolved decoding on an MNE epochs file and save metrics as CSV.

    If ``temporal_train_window`` is set, models are trained on every decoding
    window whose center lies in that interval and are evaluated at every test
    time. The per-test-time probabilities are averaged across those train-time
    models, turning temporal generalization into a direct result-improving
    train-window ensemble. Without the option, the historical diagonal
    train-time == test-time decoding path is used. If
    ``select_temporal_train_window`` is enabled, that train-time ensemble is
    selected by inner-CV temporal-generalization scores inside each outer fold.
    """
    if temporal_train_window is not None and select_temporal_train_window:
        raise ValueError("Set either temporal_train_window or select_temporal_train_window, not both.")
    epochs, metadata = _load_epochs_and_metadata(epochs_path, metadata_csv)
    decoder_name = normalize_decoder_name(decoder)
    emission_modes = list(EMISSION_MODE_CHOICES) if emission_mode == "both" else [normalize_emission_mode(emission_mode)]
    feature_preprocessor_name = normalize_feature_preprocessor(feature_preprocessor)
    if feature_preprocessor_name == "none" and pca_components is not None:
        raise ValueError("pca_components can only be set when feature_preprocessor is 'pca' or 'pca_whiten'.")
    pca_components_value = normalize_pca_components(pca_components) if feature_preprocessor_name != "none" else None
    tuning_scoring = normalize_tuning_scoring(tuning_scoring)
    tuning_c_grid_values = parse_c_grid(tuning_c_grid)
    if feature_preprocessor_name == "none":
        if tuning_pca_components_grid is not None:
            raise ValueError(
                "tuning_pca_components_grid can only be set when feature_preprocessor is 'pca' or 'pca_whiten'."
            )
        tuning_pca_components_grid_values = None
    else:
        tuning_pca_components_grid_values = parse_pca_components_grid(tuning_pca_components_grid)
    normalized_temporal_train_window = _normalize_temporal_train_window(temporal_train_window)
    temporal_selection_scoring = _normalize_temporal_selection_scoring(temporal_selection_scoring)
    if temporal_selection_cv_splits < 2:
        raise ValueError("temporal_selection_cv_splits must be at least 2.")
    if temporal_selection_n_windows < 1:
        raise ValueError("temporal_selection_n_windows must be at least 1.")
    if calibration_cv_splits < 2:
        raise ValueError("calibration_cv_splits must be at least 2.")
    effective_temporal_selection_radius = window_ms / 1000.0 if temporal_selection_radius is None else float(temporal_selection_radius)

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
    if select_temporal_train_window:
        temporal_mode = "selected_train_window_ensemble"
    elif normalized_temporal_train_window is not None:
        temporal_mode = "train_window_ensemble"
    else:
        temporal_mode = "same_time"
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
            "select_temporal_train_window": select_temporal_train_window,
            "temporal_selection_cv_splits": temporal_selection_cv_splits if select_temporal_train_window else None,
            "temporal_selection_scoring": temporal_selection_scoring if select_temporal_train_window else None,
            "temporal_selection_radius": effective_temporal_selection_radius if select_temporal_train_window else None,
            "temporal_selection_n_windows": temporal_selection_n_windows if select_temporal_train_window else None,
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
        select_temporal_train_window=select_temporal_train_window,
        temporal_selection_cv_splits=temporal_selection_cv_splits if select_temporal_train_window else None,
        temporal_selection_scoring=temporal_selection_scoring if select_temporal_train_window else None,
        temporal_selection_radius=effective_temporal_selection_radius if select_temporal_train_window else None,
        temporal_selection_n_windows=temporal_selection_n_windows if select_temporal_train_window else None,
        tune_hyperparameters=tune_hyperparameters,
        tuning_cv_splits=tuning_cv_splits,
        tuning_scoring=tuning_scoring,
        tuning_c_grid=tuning_c_grid_values,
        tuning_pca_components_grid=tuning_pca_components_grid_values,
        calibration_cv_splits=calibration_cv_splits,
        group_aware_calibration=(
            groups is not None
            and decoder_name == "linear_svm"
            and emission_mode in {"calibrated", "both"}
            and not tune_hyperparameters
        ),
    )

    data = epochs.get_data(copy=False)
    classes = np.arange(len(encoder.classes_))
    rows = []
    calibration_rows = []
    observation_rows = []
    windows = time_windows(epochs.times, window_ms=window_ms, step_ms=step_ms)
    selected_train_windows = _select_temporal_train_windows(windows, normalized_temporal_train_window)
    splits = list(make_cross_validator(labels, groups, n_splits))

    if selected_train_windows is None and not select_temporal_train_window:
        for time_window in windows:
            features = _features_for_window(data, time_window)
            start, stop, center = time_window
            for fold, (train_idx, test_idx) in enumerate(splits):
                test_labels = labels[test_idx]
                for current_emission_mode in emission_modes:
                    tuning_cv = (
                        make_tuning_cross_validator(labels[train_idx], None if groups is None else groups[train_idx], tuning_cv_splits)
                        if tune_hyperparameters
                        else 3
                    )
                    calibration_cv = _calibration_cv_for_decoder(
                        decoder_name=decoder_name,
                        emission_mode=current_emission_mode,
                        labels=labels[train_idx],
                        groups=None if groups is None else groups[train_idx],
                        n_splits=calibration_cv_splits,
                        tune_hyperparameters=tune_hyperparameters,
                    )
                    model = make_decoder(
                        decoder_name,
                        max_iter=max_iter,
                        emission_mode=current_emission_mode,
                        feature_preprocessor=feature_preprocessor_name,
                        pca_components=pca_components_value,
                        tune_hyperparameters=tune_hyperparameters,
                        tuning_cv=tuning_cv,
                        tuning_scoring=tuning_scoring,
                        tuning_c_grid=tuning_c_grid_values,
                        tuning_pca_components_grid=tuning_pca_components_grid_values,
                        calibration_cv=calibration_cv,
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
                        select_temporal_train_window=False,
                        temporal_selection_cv_splits=None,
                        temporal_selection_scoring=None,
                        temporal_selection_radius=None,
                        temporal_selection_n_windows=None,
                        tune_hyperparameters=tune_hyperparameters,
                        tuning_cv_splits=tuning_cv_splits,
                        tuning_scoring=tuning_scoring,
                        tuning_c_grid=tuning_c_grid_values,
                        tuning_pca_components_grid=tuning_pca_components_grid_values,
                        calibration_cv_splits=calibration_cv_splits,
                        group_aware_calibration=_uses_group_aware_calibration(
                            decoder_name=decoder_name,
                            emission_mode=current_emission_mode,
                            groups=None if groups is None else groups[train_idx],
                            tune_hyperparameters=tune_hyperparameters,
                        ),
                        tuning_metadata=_tuning_metadata(
                            model,
                            tune_hyperparameters=tune_hyperparameters,
                            tuning_cv_splits=tuning_cv_splits,
                            tuning_scoring=tuning_scoring,
                            tuning_c_grid=tuning_c_grid_values,
                            tuning_pca_components_grid=tuning_pca_components_grid_values,
                        ),
                    )
                    tuning_metadata = _tuning_metadata(
                        model,
                        tune_hyperparameters=tune_hyperparameters,
                        tuning_cv_splits=tuning_cv_splits,
                        tuning_scoring=tuning_scoring,
                        tuning_c_grid=tuning_c_grid_values,
                        tuning_pca_components_grid=tuning_pca_components_grid_values,
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
                        temporal_train_window=normalized_temporal_train_window,
                        train_time=center,
                        train_window_start=float(epochs.times[start]),
                        temporal_selection_score=None,
                        train_window_stop=float(epochs.times[stop - 1]),
                        n_train_windows=1,
                        calibration_out_path=calibration_out_path,
                        calibration_bins=calibration_bins,
                        observation_out_path=observation_out_path,
                        subject=subject,
                        tuning_metadata=tuning_metadata,
                    )
    else:
        feature_cache = {time_window: _features_for_window(data, time_window) for time_window in windows}
        for fold, (train_idx, test_idx) in enumerate(splits):
            test_labels = labels[test_idx]
            for current_emission_mode in emission_modes:
                if select_temporal_train_window:
                    current_train_windows, temporal_selection_score = _select_stable_temporal_train_windows(
                        feature_cache=feature_cache,
                        windows=windows,
                        labels=labels,
                        groups=groups,
                        train_idx=np.asarray(train_idx),
                        classes=classes,
                        decoder_name=decoder_name,
                        max_iter=max_iter,
                        emission_mode=current_emission_mode,
                        feature_preprocessor_name=feature_preprocessor_name,
                        pca_components_value=pca_components_value,
                        temporal_selection_cv_splits=temporal_selection_cv_splits,
                        temporal_selection_scoring=temporal_selection_scoring,
                        temporal_selection_radius=effective_temporal_selection_radius,
                        temporal_selection_n_windows=temporal_selection_n_windows,
                    )
                    current_temporal_train_window = (
                        float(min(window[2] for window in current_train_windows)),
                        float(max(window[2] for window in current_train_windows)),
                    )
                else:
                    if selected_train_windows is None:
                        raise RuntimeError("Internal error: no temporal train windows selected.")
                    current_train_windows = selected_train_windows
                    current_temporal_train_window = normalized_temporal_train_window
                    temporal_selection_score = None
                train_time, train_window_start, train_window_stop = _train_window_summary(epochs, current_train_windows)
                train_window_centers = [window[2] for window in current_train_windows]
                tuning_cv = (
                    make_tuning_cross_validator(labels[train_idx], None if groups is None else groups[train_idx], tuning_cv_splits)
                    if tune_hyperparameters
                    else 3
                )
                calibration_cv = _calibration_cv_for_decoder(
                    decoder_name=decoder_name,
                    emission_mode=current_emission_mode,
                    labels=labels[train_idx],
                    groups=None if groups is None else groups[train_idx],
                    n_splits=calibration_cv_splits,
                    tune_hyperparameters=tune_hyperparameters,
                )
                fitted_models = []
                probability_sums = {
                    time_window: np.zeros((len(test_idx), len(classes)), dtype=float)
                    for time_window in windows
                }
                for train_window in current_train_windows:
                    train_features = feature_cache[train_window]
                    model = make_decoder(
                        decoder_name,
                        max_iter=max_iter,
                        emission_mode=current_emission_mode,
                        feature_preprocessor=feature_preprocessor_name,
                        pca_components=pca_components_value,
                        tune_hyperparameters=tune_hyperparameters,
                        tuning_cv=tuning_cv,
                        tuning_scoring=tuning_scoring,
                        tuning_c_grid=tuning_c_grid_values,
                        tuning_pca_components_grid=tuning_pca_components_grid_values,
                        calibration_cv=calibration_cv,
                    )
                    model.fit(train_features[train_idx], labels[train_idx])
                    fitted_models.append(model)
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
                    temporal_train_window=current_temporal_train_window,
                    train_window_centers=train_window_centers,
                    select_temporal_train_window=select_temporal_train_window,
                    temporal_selection_cv_splits=temporal_selection_cv_splits if select_temporal_train_window else None,
                    temporal_selection_scoring=temporal_selection_scoring if select_temporal_train_window else None,
                    temporal_selection_radius=effective_temporal_selection_radius if select_temporal_train_window else None,
                    temporal_selection_n_windows=temporal_selection_n_windows if select_temporal_train_window else None,
                    tune_hyperparameters=tune_hyperparameters,
                    tuning_cv_splits=tuning_cv_splits,
                    tuning_scoring=tuning_scoring,
                    tuning_c_grid=tuning_c_grid_values,
                    tuning_pca_components_grid=tuning_pca_components_grid_values,
                    calibration_cv_splits=calibration_cv_splits,
                    group_aware_calibration=_uses_group_aware_calibration(
                        decoder_name=decoder_name,
                        emission_mode=current_emission_mode,
                        groups=None if groups is None else groups[train_idx],
                        tune_hyperparameters=tune_hyperparameters,
                    ),
                    tuning_metadata=_tuning_metadata(
                        fitted_models,
                        tune_hyperparameters=tune_hyperparameters,
                        tuning_cv_splits=tuning_cv_splits,
                        tuning_scoring=tuning_scoring,
                        tuning_c_grid=tuning_c_grid_values,
                        tuning_pca_components_grid=tuning_pca_components_grid_values,
                    ),
                )
                tuning_metadata = _tuning_metadata(
                    fitted_models,
                    tune_hyperparameters=tune_hyperparameters,
                    tuning_cv_splits=tuning_cv_splits,
                    tuning_scoring=tuning_scoring,
                    tuning_c_grid=tuning_c_grid_values,
                    tuning_pca_components_grid=tuning_pca_components_grid_values,
                )
                for test_window in windows:
                    probabilities = _probability_average(probability_sums[test_window], len(current_train_windows))
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
                        temporal_train_window=current_temporal_train_window,
                        train_time=train_time,
                        train_window_start=train_window_start,
                        temporal_selection_score=temporal_selection_score,
                        train_window_stop=train_window_stop,
                        n_train_windows=len(current_train_windows),
                        calibration_out_path=calibration_out_path,
                        calibration_bins=calibration_bins,
                        observation_out_path=observation_out_path,
                        subject=subject,
                        tuning_metadata=tuning_metadata,
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
    parser.add_argument("--decoder", choices=DECODER_CLI_CHOICES, default="logistic")
    parser.add_argument("--emission-mode", choices=EMISSION_RUN_CHOICES, default="calibrated")
    parser.add_argument("--feature-preprocessor", choices=FEATURE_PREPROCESSOR_RUN_CHOICES, default="none")
    parser.add_argument(
        "--pca-components",
        help="PCA component count or explained-variance fraction. Only valid with --feature-preprocessor pca or pca-whiten.",
    )
    parser.add_argument("--tune-hyperparameters", action="store_true", help="Use nested inner-CV hyperparameter selection inside each outer train fold.")
    parser.add_argument("--tuning-cv-splits", type=int, default=3, help="Maximum number of inner CV folds for --tune-hyperparameters.")
    parser.add_argument("--tuning-scoring", choices=TUNING_SCORING_CHOICES, default="accuracy", help="Inner-CV objective for --tune-hyperparameters.")
    parser.add_argument("--selection-metric", choices=RESULT_SELECTION_METRIC_CHOICES, default="accuracy", help="Metric used only for the console 'best time' summary.")
    parser.add_argument("--calibration-cv-splits", type=int, default=3, help="Maximum number of inner calibration folds for calibrated linear SVM emissions.")
    parser.add_argument(
        "--tuning-c-grid",
        default=",".join(str(value) for value in parse_c_grid(None)),
        help="Comma-separated positive C values for tuned logistic regression and linear SVM.",
    )
    parser.add_argument(
        "--tuning-pca-components-grid",
        help=(
            "Comma-separated PCA component counts or explained-variance fractions for --tune-hyperparameters "
            "when PCA preprocessing is enabled. Use 'none' for PCA's full-rank default."
        ),
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
    parser.add_argument(
        "--select-temporal-train-window",
        action="store_true",
        help=(
            "Select a stable temporal train-window ensemble inside each outer fold by inner-CV temporal "
            "generalization, then evaluate the selected ensemble at every test time."
        ),
    )
    parser.add_argument(
        "--temporal-selection-cv-splits",
        type=int,
        default=3,
        help="Maximum number of inner CV folds for --select-temporal-train-window.",
    )
    parser.add_argument(
        "--temporal-selection-scoring",
        choices=TEMPORAL_SELECTION_SCORING_CHOICES,
        default="accuracy",
        help="Inner temporal-generalization objective for --select-temporal-train-window.",
    )
    parser.add_argument(
        "--temporal-selection-radius",
        type=float,
        help="Neighboring test-time radius in seconds used for temporal-generalization selection. Defaults to the window width.",
    )
    parser.add_argument(
        "--temporal-selection-n-windows",
        type=int,
        default=DEFAULT_TEMPORAL_SELECTION_N_WINDOWS,
        help="Number of contiguous train-time windows selected by --select-temporal-train-window.",
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
        tune_hyperparameters=args.tune_hyperparameters,
        tuning_cv_splits=args.tuning_cv_splits,
        tuning_scoring=args.tuning_scoring,
        tuning_c_grid=args.tuning_c_grid,
        tuning_pca_components_grid=args.tuning_pca_components_grid,
        calibration_cv_splits=args.calibration_cv_splits,
        calibration_out_path=args.calibration_out,
        calibration_bins=args.calibration_bins,
        observation_out_path=args.observations_out,
        subject=args.subject,
        temporal_train_window=tuple(args.temporal_train_window) if args.temporal_train_window is not None else None,
        select_temporal_train_window=args.select_temporal_train_window,
        temporal_selection_cv_splits=args.temporal_selection_cv_splits,
        temporal_selection_scoring=args.temporal_selection_scoring,
        temporal_selection_radius=args.temporal_selection_radius,
        temporal_selection_n_windows=args.temporal_selection_n_windows,
    )
    print(f"Wrote {args.out}")
    if args.observations_out is not None:
        print(f"Wrote probability observations: {args.observations_out}")
    for emission_mode_name, summary in results.groupby("emission_mode", sort=True):
        time_summary = summary.groupby("time")[["accuracy", "log_loss", "brier", "ece"]].mean()
        best_time = _best_time_by_metric(time_summary, args.selection_metric)
        best_value = time_summary.loc[best_time, args.selection_metric]
        direction = "lowest" if args.selection_metric in RESULT_SELECTION_MINIMIZE_METRICS else "highest"
        print(
            f"Best {emission_mode_name} mean {args.selection_metric} "
            f"({direction}): {best_value:.3f} at {best_time:.3f}s"
        )


if __name__ == "__main__":
    main()
