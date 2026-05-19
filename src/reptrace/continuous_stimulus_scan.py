from __future__ import annotations

import argparse
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import LabelEncoder

from reptrace.decoding import (
    FEATURE_PREPROCESSOR_CHOICES,
    TUNING_SCORING_CHOICES,
    make_decoder,
    make_tuning_cross_validator,
    normalize_decoder_name,
    normalize_emission_mode,
    normalize_feature_preprocessor,
    normalize_pca_components,
    normalize_tuning_scoring,
    parse_c_grid,
    predict_emission_probabilities,
)
from reptrace.observations import ProbabilityObservationTable, stable_hash
from reptrace.stimulus_detection import (
    CONFLICT_RESOLUTION_MODES,
    SCORE_MODES,
    THRESHOLD_METHODS,
    detect_stimulus_events,
    fit_stimulus_detection_thresholds,
    match_stimulus_annotations,
    summarize_stimulus_events,
)

FEATURE_PREPROCESSOR_RUN_CHOICES = (*FEATURE_PREPROCESSOR_CHOICES, "pca-whiten")


@dataclass(frozen=True)
class ScanSegment:
    """One continuous interval to scan as an independent probability stream."""

    stream_id: str
    start: float
    stop: float
    output_origin: float


@dataclass(frozen=True)
class ContinuousStimulusScanResult:
    """Tables emitted by the continuous stimulus-scan workflow."""

    observations: pd.DataFrame
    annotations: pd.DataFrame
    thresholds: pd.DataFrame
    events: pd.DataFrame
    summary: pd.DataFrame
    event_metrics: pd.DataFrame


def _read_table(path: Path, sep: str = "auto") -> pd.DataFrame:
    if sep == "auto":
        return pd.read_csv(path, sep=None, engine="python")
    if sep in {"tab", "\\t"}:
        sep = "\t"
    return pd.read_csv(path, sep=sep)


def _pattern_mask(values: pd.Series, pattern: str, *, case_sensitive: bool) -> pd.Series:
    flags = 0 if case_sensitive else re.IGNORECASE
    return values.astype("string").str.contains(pattern, flags=flags, regex=True, na=False)


def label_event_table(
    events: pd.DataFrame,
    *,
    onset_column: str = "onset",
    label_column: str = "stimulus_class",
    source_column: str | None = None,
    positive_pattern: str | None = None,
    negative_pattern: str | None = None,
    positive_label: str = "positive",
    negative_label: str = "negative",
    case_sensitive: bool = False,
) -> pd.DataFrame:
    """Return events with numeric onsets and string class labels.

    If ``source_column`` and ``positive_pattern`` are supplied, labels are built
    from regex matches. Otherwise ``label_column`` is used directly.
    """

    if onset_column not in events.columns:
        raise ValueError(f"Event table is missing onset column '{onset_column}'.")
    labeled = events.copy()
    if source_column is not None or positive_pattern is not None:
        if source_column is None or positive_pattern is None:
            raise ValueError("source_column and positive_pattern must be provided together.")
        if source_column not in labeled.columns:
            raise ValueError(f"Event table is missing source column '{source_column}'.")
        positive = _pattern_mask(labeled[source_column], positive_pattern, case_sensitive=case_sensitive)
        negative = (
            _pattern_mask(labeled[source_column], negative_pattern, case_sensitive=case_sensitive)
            if negative_pattern is not None
            else labeled[source_column].notna() & ~positive
        )
        labeled[label_column] = pd.NA
        labeled.loc[positive, label_column] = positive_label
        labeled.loc[negative, label_column] = negative_label
    elif label_column not in labeled.columns:
        raise ValueError(f"Event table is missing label column '{label_column}'.")

    labeled[onset_column] = pd.to_numeric(labeled[onset_column], errors="raise")
    labeled = labeled.loc[labeled[label_column].notna()].copy()
    labeled[label_column] = labeled[label_column].astype(str)
    return labeled.sort_values(onset_column).reset_index(drop=True)


def _limit_per_class(events: pd.DataFrame, *, onset_column: str, label_column: str, max_events_per_class: int | None) -> pd.DataFrame:
    if max_events_per_class is None:
        return events
    return (
        events.groupby(label_column, sort=True, group_keys=False)
        .head(max_events_per_class)
        .sort_values(onset_column)
        .reset_index(drop=True)
    )


def _mne_events(raw: mne.io.BaseRaw, events: pd.DataFrame, *, onset_column: str, label_column: str, event_id: dict[str, int]) -> np.ndarray:
    samples = raw.time_as_index(events[onset_column].to_numpy(dtype=float), use_rounding=True) + raw.first_samp
    codes = events[label_column].map(event_id).to_numpy(dtype=int)
    return np.column_stack([samples, np.zeros(len(samples), dtype=int), codes])


def _event_features(
    *,
    raw_path: Path,
    events: pd.DataFrame,
    onset_column: str,
    label_column: str,
    event_id: dict[str, int],
    window: tuple[float, float],
    picks: str | Sequence[str],
    baseline: tuple[float | None, float | None] | None,
    demean_window: bool,
) -> tuple[np.ndarray, np.ndarray, list[str], int]:
    raw = mne.io.read_raw_fif(raw_path, preload=False, verbose="error")
    epochs = mne.Epochs(
        raw,
        _mne_events(raw, events, onset_column=onset_column, label_column=label_column, event_id=event_id),
        event_id=event_id,
        tmin=window[0],
        tmax=window[1],
        baseline=baseline,
        metadata=events.reset_index(drop=True),
        picks=picks,
        preload=True,
        reject_by_annotation=True,
        verbose="error",
    )
    if epochs.metadata is None:
        raise ValueError("MNE dropped epoch metadata unexpectedly.")
    data = epochs.get_data(copy=False)
    if demean_window:
        data = data - data.mean(axis=2, keepdims=True)
    labels = epochs.metadata[label_column].astype(str).to_numpy()
    return data.reshape(len(labels), -1), labels, list(epochs.ch_names), data.shape[2]


def _best_params_json(model: object) -> str:
    """Return GridSearchCV best parameters as stable compact JSON."""

    best_params = getattr(model, "best_params_", None)
    if best_params is None:
        return ""
    return json.dumps(best_params, sort_keys=True, default=str, separators=(",", ":"))


def _tuning_metadata(
    model: object,
    *,
    tune_hyperparameters: bool,
    tuning_cv_splits: int,
    tuning_scoring: str,
    tuning_c_grid: Sequence[float],
) -> dict[str, object]:
    """Build result-table metadata for the decoder stack used during scanning."""

    metadata: dict[str, object] = {
        "tuned_hyperparameters": bool(tune_hyperparameters),
        "tuning_cv_splits": int(tuning_cv_splits) if tune_hyperparameters else "",
        "tuning_scoring": tuning_scoring if tune_hyperparameters else "",
        "tuning_c_grid": "|".join(str(value) for value in tuning_c_grid) if tune_hyperparameters else "",
        "best_params": "",
    }
    if tune_hyperparameters:
        metadata["best_params"] = _best_params_json(model)
        if hasattr(model, "best_score_"):
            metadata["best_score"] = float(model.best_score_)
    return metadata


def _fit_decoder(
    *,
    train_raw: Path,
    train_events: pd.DataFrame,
    onset_column: str,
    label_column: str,
    train_window: tuple[float, float],
    picks: str | Sequence[str],
    baseline: tuple[float | None, float | None] | None,
    decoder: str,
    emission_mode: str,
    max_iter: int,
    feature_preprocessor: str = "none",
    pca_components: int | float | None = None,
    tune_hyperparameters: bool = False,
    tuning_cv_splits: int = 3,
    tuning_scoring: str = "accuracy",
    tuning_c_grid: Sequence[float] | str | None = None,
    demean_window: bool = False,
) -> tuple[object, LabelEncoder, list[str], int, pd.DataFrame, dict[str, object]]:
    classes = sorted(train_events[label_column].dropna().astype(str).unique())
    if len(classes) < 2:
        raise ValueError("Training events must contain at least two classes.")
    event_id = {class_name: index + 1 for index, class_name in enumerate(classes)}
    features, raw_labels, channel_names, n_window = _event_features(
        raw_path=train_raw,
        events=train_events,
        onset_column=onset_column,
        label_column=label_column,
        event_id=event_id,
        window=train_window,
        picks=picks,
        baseline=baseline,
        demean_window=demean_window,
    )
    encoder = LabelEncoder()
    labels = encoder.fit_transform(raw_labels)
    tuning_c_grid_values = parse_c_grid(tuning_c_grid)
    tuning_cv = (
        make_tuning_cross_validator(labels, None, tuning_cv_splits)
        if tune_hyperparameters
        else 3
    )
    model = make_decoder(
        decoder,
        max_iter=max_iter,
        emission_mode=emission_mode,
        feature_preprocessor=feature_preprocessor,
        pca_components=pca_components,
        tune_hyperparameters=tune_hyperparameters,
        tuning_cv=tuning_cv,
        tuning_scoring=tuning_scoring,
        tuning_c_grid=tuning_c_grid_values,
    )
    model.fit(features, labels)
    counts = pd.Series(raw_labels, name=label_column).value_counts().rename_axis(label_column).reset_index(name="n_train_events")
    tuning_metadata = _tuning_metadata(
        model,
        tune_hyperparameters=tune_hyperparameters,
        tuning_cv_splits=tuning_cv_splits,
        tuning_scoring=tuning_scoring,
        tuning_c_grid=tuning_c_grid_values,
    )
    return model, encoder, channel_names, n_window, counts, tuning_metadata


def _safe_stream_id(path: Path) -> str:
    return path.stem.replace("_meg", "").replace(" ", "_")


def _continuous_split_id(*, train_raw: Path, scan_raw: Path, slice_seed: int) -> str:
    return f"continuous-train-{train_raw.stem}-scan-{scan_raw.stem}-seed-{slice_seed}"


def _continuous_preprocessing_hash(
    *,
    train_window: tuple[float, float],
    picks: str | Sequence[str],
    baseline: tuple[float | None, float | None] | None,
    demean_window: bool,
    scan_step: float,
    n_window_samples: int,
    channel_names: Sequence[str],
) -> str:
    return stable_hash(
        {
            "train_window": train_window,
            "picks": picks,
            "baseline": baseline,
            "demean_window": demean_window,
            "scan_step": scan_step,
            "n_window_samples": n_window_samples,
            "channel_names": list(channel_names),
        }
    )


def _continuous_model_hash(
    *,
    decoder: str,
    emission_mode: str,
    max_iter: int,
    train_window: tuple[float, float],
    feature_preprocessor: str,
    pca_components: int | float | None,
    tune_hyperparameters: bool,
    tuning_cv_splits: int,
    tuning_scoring: str,
    tuning_c_grid: Sequence[float],
    tuning_metadata: dict[str, object] | None = None,
) -> str:
    payload: dict[str, object] = {
        "backend": "sklearn",
        "decoder": decoder,
        "emission_mode": emission_mode,
        "max_iter": max_iter,
        "train_window": train_window,
        "feature_preprocessor": feature_preprocessor,
        "pca_components": pca_components,
    }
    if tune_hyperparameters:
        payload.update(
            {
                "tune_hyperparameters": True,
                "tuning_cv_splits": tuning_cv_splits,
                "tuning_scoring": tuning_scoring,
                "tuning_c_grid": tuple(tuning_c_grid),
                "best_params": (tuning_metadata or {}).get("best_params", ""),
            }
        )
    return stable_hash(payload)


def _baseline_sample_mask(
    *,
    train_window: tuple[float, float],
    n_window_samples: int,
    baseline: tuple[float | None, float | None],
) -> np.ndarray:
    """Return samples in a scan window that correspond to the epoch baseline.

    MNE applies ``baseline`` relative to the event-aligned epoch time axis.  The
    continuous scanner uses the same feature geometry, but the window is indexed
    around a candidate center time.  Mapping the baseline interval onto the
    training epoch time axis keeps streamed windows numerically consistent with
    the event epochs used to fit and validate the decoder.
    """

    if n_window_samples < 1:
        raise ValueError("n_window_samples must be positive.")
    epoch_start, epoch_stop = map(float, train_window)
    baseline_start = epoch_start if baseline[0] is None else float(baseline[0])
    baseline_stop = epoch_stop if baseline[1] is None else float(baseline[1])
    if baseline_stop < baseline_start:
        raise ValueError("Baseline stop must be greater than or equal to baseline start.")

    epoch_times = np.linspace(epoch_start, epoch_stop, n_window_samples)
    tolerance = np.finfo(float).eps * max(
        1.0,
        abs(epoch_start),
        abs(epoch_stop),
        abs(baseline_start),
        abs(baseline_stop),
    )
    baseline_samples = (epoch_times >= baseline_start - tolerance) & (epoch_times <= baseline_stop + tolerance)
    if not baseline_samples.any():
        raise ValueError("Epoch baseline interval does not include any samples in the scan window.")
    return baseline_samples


def _apply_epoch_baseline_to_window(
    data: np.ndarray,
    *,
    train_window: tuple[float, float],
    baseline: tuple[float | None, float | None] | None,
) -> np.ndarray:
    """Apply the same event-epoch baseline correction to one scan window."""

    if baseline is None:
        return data
    if data.ndim != 2:
        raise ValueError("Scan-window data must be a channels-by-samples matrix.")
    baseline_samples = _baseline_sample_mask(train_window=train_window, n_window_samples=data.shape[1], baseline=baseline)
    return data - data[:, baseline_samples].mean(axis=1, keepdims=True)


def _standardize_stream_observations(
    observations: pd.DataFrame,
    *,
    subject: str | None,
    split_id: str,
    slice_seed: int,
    decoder: str,
    emission_mode: str,
    train_time: float,
    preprocessing_hash: str,
    model_hash: str,
) -> pd.DataFrame:
    standardized = observations.copy()
    if "sequence_id" not in standardized.columns and {"stream_id", "sample_index"}.issubset(standardized.columns):
        standardized["sequence_id"] = standardized["stream_id"].astype(str) + ":" + standardized["sample_index"].astype(str)
    return ProbabilityObservationTable(standardized).standardized(
        defaults={
            "subject": "" if subject is None else subject,
            "fold": "",
            "split_id": split_id,
            "seed": slice_seed,
            "decoder": decoder,
            "backend": "sklearn",
            "emission_mode": emission_mode,
            "train_time": train_time,
            "calibration_fold": "",
            "preprocessing_hash": preprocessing_hash,
            "model_hash": model_hash,
        }
    ).frame


def _event_mask_in_window(events: pd.DataFrame, *, onset_column: str, start: float, stop: float, labels: set[str] | None = None, label_column: str = "stimulus_class") -> pd.Series:
    onsets = pd.to_numeric(events[onset_column], errors="coerce")
    mask = (onsets >= start) & (onsets <= stop)
    if labels is not None:
        mask &= events[label_column].astype(str).isin(labels)
    return mask


def build_scan_segments(
    *,
    scan_raw: Path,
    scan_start: float | None,
    scan_stop: float | None,
    slice_duration: float | None = None,
    slice_starts: Sequence[float] | None = None,
    slice_count: int | None = None,
    slice_seed: int = 13,
    scan_events: pd.DataFrame | None = None,
    onset_column: str = "onset",
    label_column: str = "stimulus_class",
    target_classes: Sequence[str] | None = None,
    threshold_window: tuple[float, float] | None = None,
    detection_window: tuple[float, float] | None = None,
    require_target_event: bool = False,
    exclude_events_from_threshold_window: bool = False,
    stream_id: str | None = None,
) -> list[ScanSegment]:
    """Build full-run, explicit-slice, or random-slice scan segments."""

    raw = mne.io.read_raw_fif(scan_raw, preload=False, verbose="error")
    raw_start = 0.0 if scan_start is None else scan_start
    raw_stop = float(raw.times[-1]) if scan_stop is None else scan_stop
    if raw_stop <= raw_start:
        raise ValueError("scan_stop must be greater than scan_start.")
    base_stream_id = stream_id or _safe_stream_id(scan_raw)
    if slice_duration is None:
        return [ScanSegment(base_stream_id, raw_start, raw_stop, 0.0)]
    if slice_duration <= 0:
        raise ValueError("slice_duration must be positive.")

    starts: list[float]
    if slice_starts:
        starts = [float(start) for start in slice_starts]
    elif slice_count:
        rng = np.random.default_rng(slice_seed)
        starts = []
        target_set = set(map(str, target_classes or [])) or None
        tries = 0
        while len(starts) < slice_count and tries < max(1000, slice_count * 500):
            tries += 1
            start = float(rng.uniform(raw_start, raw_stop - slice_duration))
            if scan_events is not None and exclude_events_from_threshold_window and threshold_window is not None:
                if _event_mask_in_window(
                    scan_events,
                    onset_column=onset_column,
                    start=start + threshold_window[0],
                    stop=start + threshold_window[1],
                    label_column=label_column,
                ).any():
                    continue
            if scan_events is not None and require_target_event and detection_window is not None:
                if not _event_mask_in_window(
                    scan_events,
                    onset_column=onset_column,
                    start=start + detection_window[0],
                    stop=start + detection_window[1],
                    labels=target_set,
                    label_column=label_column,
                ).any():
                    continue
            starts.append(start)
        if len(starts) < slice_count:
            raise ValueError(f"Only selected {len(starts)} random slice(s); requested {slice_count}.")
    else:
        starts = list(np.arange(raw_start, raw_stop - slice_duration + 1e-12, slice_duration))

    segments = []
    for index, start in enumerate(starts):
        stop = start + slice_duration
        if start < raw_start or stop > raw_stop:
            raise ValueError(f"Slice [{start}, {stop}] is outside scan interval [{raw_start}, {raw_stop}].")
        segments.append(ScanSegment(f"{base_stream_id}_slice{index:03d}", start, stop, start))
    return segments


def _scan_raw_probabilities(
    *,
    scan_raw: Path,
    model: object,
    encoder: LabelEncoder,
    channel_names: Sequence[str],
    n_window_samples: int,
    segments: Sequence[ScanSegment],
    scan_step: float,
    decoder: str,
    emission_mode: str,
    feature_preprocessor: str = "none",
    pca_components: int | float | None = None,
    tuning_metadata: dict[str, object] | None = None,
    subject: str | None,
    train_window: tuple[float, float],
    baseline: tuple[float | None, float | None] | None,
    demean_window: bool,
) -> pd.DataFrame:
    if scan_step <= 0:
        raise ValueError("scan_step must be positive.")
    raw = mne.io.read_raw_fif(scan_raw, preload=False, verbose="error")
    raw.pick(list(channel_names))
    sfreq = float(raw.info["sfreq"])
    half_window = n_window_samples // 2
    rows = []
    classes = list(encoder.classes_)
    for segment in segments:
        centers = np.arange(segment.start + (n_window_samples / sfreq) / 2.0, segment.stop - (n_window_samples / sfreq) / 2.0 + 1e-12, scan_step)
        features = []
        metadata = []
        for center in centers:
            center_sample = int(raw.time_as_index([float(center)], use_rounding=True)[0])
            start_sample = center_sample - half_window
            stop_sample = start_sample + n_window_samples
            if start_sample < 0 or stop_sample > raw.n_times:
                continue
            data = raw.get_data(start=start_sample, stop=stop_sample)
            if data.shape != (len(channel_names), n_window_samples):
                continue
            data = _apply_epoch_baseline_to_window(data, train_window=train_window, baseline=baseline)
            if demean_window:
                data = data - data.mean(axis=1, keepdims=True)
            window_start = start_sample / sfreq - segment.output_origin
            window_stop = stop_sample / sfreq - segment.output_origin
            features.append(data.reshape(-1))
            metadata.append((float(center) - segment.output_origin, window_start, window_stop))
        if not features:
            continue
        probabilities = predict_emission_probabilities(model, np.vstack(features), emission_mode=emission_mode)
        predictions = probabilities.argmax(axis=1)
        for sample_index, ((time, window_start, window_stop), probabilities_row, predicted_label) in enumerate(
            zip(metadata, probabilities, predictions, strict=True)
        ):
            row = {
                "stream_id": segment.stream_id,
                "decoder": decoder,
                "emission_mode": emission_mode,
                "feature_preprocessor": feature_preprocessor,
                "pca_components": "" if pca_components is None else pca_components,
                "time": time,
                "window_start": window_start,
                "window_stop": window_stop,
                "sample_index": sample_index,
                "predicted_label": int(predicted_label),
                "predicted_class": str(classes[int(predicted_label)]),
                "confidence": float(probabilities_row[int(predicted_label)]),
            }
            if subject is not None:
                row = {"subject": subject, **row}
            row.update(tuning_metadata or {})
            for class_index, class_name in enumerate(classes):
                row[f"class_{class_index}"] = str(class_name)
                row[f"prob_class_{class_index}"] = float(probabilities_row[class_index])
            rows.append(row)
    return pd.DataFrame(rows)


def _annotation_table(
    *,
    scan_events: pd.DataFrame | None,
    segments: Sequence[ScanSegment],
    onset_column: str,
    label_column: str,
    target_classes: Sequence[str],
    annotation_latency: float,
    detection_window: tuple[float, float] | None,
) -> pd.DataFrame:
    if scan_events is None:
        return pd.DataFrame(columns=["stream_id", "annotation_id", "stimulus_class", "stimulus_label", "stimulus_onset_time", "onset_time"])
    rows = []
    targets = set(map(str, target_classes))
    for segment in segments:
        event_frame = scan_events.loc[scan_events[label_column].astype(str).isin(targets)].copy()
        event_frame = event_frame.loc[(event_frame[onset_column] >= segment.start) & (event_frame[onset_column] <= segment.stop)]
        for _, event in event_frame.iterrows():
            stimulus_onset = float(event[onset_column] - segment.output_origin)
            onset_time = stimulus_onset + annotation_latency
            if detection_window is not None and not (detection_window[0] <= onset_time <= detection_window[1]):
                continue
            rows.append(
                {
                    "stream_id": segment.stream_id,
                    "annotation_id": f"{segment.stream_id}_{len(rows):05d}",
                    "stimulus_class": str(event[label_column]),
                    "stimulus_label": str(event[label_column]),
                    "stimulus_onset_time": stimulus_onset,
                    "onset_time": onset_time,
                }
            )
    return pd.DataFrame(rows)


def _held_out_event_metrics(
    *,
    model: object,
    encoder: LabelEncoder,
    scan_raw: Path,
    scan_events: pd.DataFrame | None,
    onset_column: str,
    label_column: str,
    train_window: tuple[float, float],
    picks: Sequence[str],
    baseline: tuple[float | None, float | None] | None,
    decoder: str,
    emission_mode: str,
    feature_preprocessor: str,
    pca_components: int | float | None,
    tuning_metadata: dict[str, object],
    demean_window: bool,
) -> pd.DataFrame:
    if scan_events is None or scan_events.empty:
        return pd.DataFrame()
    event_frame = scan_events.loc[scan_events[label_column].astype(str).isin(set(map(str, encoder.classes_)))].copy()
    if event_frame.empty:
        return pd.DataFrame()
    event_id = {str(class_name): index + 1 for index, class_name in enumerate(encoder.classes_)}
    features, raw_labels, _channel_names, _n_window = _event_features(
        raw_path=scan_raw,
        events=event_frame,
        onset_column=onset_column,
        label_column=label_column,
        event_id=event_id,
        window=train_window,
        picks=list(picks),
        baseline=baseline,
        demean_window=demean_window,
    )
    labels = encoder.transform(raw_labels)
    probabilities = predict_emission_probabilities(model, features, emission_mode=emission_mode)
    predictions = probabilities.argmax(axis=1)
    row = {
        "decoder": decoder,
        "emission_mode": emission_mode,
        "feature_preprocessor": feature_preprocessor,
        "pca_components": "" if pca_components is None else pca_components,
        "n_events": len(labels),
        "accuracy": accuracy_score(labels, predictions),
        "log_loss": log_loss(labels, probabilities, labels=np.arange(len(encoder.classes_))),
    }
    row.update(tuning_metadata or {})
    return pd.DataFrame([row])


# pylint: disable-next=too-many-arguments,too-many-locals
def run_continuous_stimulus_scan(
    *,
    train_raw: Path,
    train_events: pd.DataFrame,
    scan_raw: Path,
    scan_events: pd.DataFrame | None = None,
    out_dir: Path,
    onset_column: str = "onset",
    label_column: str = "stimulus_class",
    train_window: tuple[float, float] = (0.1, 0.2),
    picks: str = "data",
    baseline: tuple[float | None, float | None] | None = None,
    decoder: str = "logistic",
    emission_mode: str = "calibrated",
    max_iter: int = 1000,
    feature_preprocessor: str = "none",
    pca_components: int | float | str | None = None,
    tune_hyperparameters: bool = False,
    tuning_cv_splits: int = 3,
    tuning_scoring: str = "accuracy",
    tuning_c_grid: Sequence[float] | str | None = None,
    demean_window: bool = False,
    scan_step: float = 0.025,
    scan_start: float | None = None,
    scan_stop: float | None = None,
    slice_duration: float | None = None,
    slice_starts: Sequence[float] | None = None,
    slice_count: int | None = None,
    slice_seed: int = 13,
    stream_id: str | None = None,
    subject: str | None = None,
    target_classes: Sequence[str] | None = None,
    threshold_window: tuple[float, float] = (0.0, 0.8),
    threshold_quantile: float = 0.95,
    threshold_method: str = "max_run",
    score_mode: str = "class_probability",
    detection_window: tuple[float, float] | None = None,
    min_consecutive: int = 1,
    min_duration: float | None = None,
    merge_gap: float | None = None,
    refractory: float | None = None,
    conflict_resolution: str = "none",
    match_tolerance: float = 0.1,
    annotation_latency: float | None = None,
    require_target_event: bool = False,
    exclude_events_from_threshold_window: bool = False,
) -> ContinuousStimulusScanResult:
    """Train an event-locked decoder, scan raw data, and detect stimulus events."""

    out_dir.mkdir(parents=True, exist_ok=True)
    decoder_name = normalize_decoder_name(decoder)
    emission_mode_name = normalize_emission_mode(emission_mode)
    feature_preprocessor_name = normalize_feature_preprocessor(feature_preprocessor)
    if feature_preprocessor_name == "none" and pca_components is not None:
        raise ValueError("pca_components can only be set when feature_preprocessor is 'pca' or 'pca_whiten'.")
    pca_components_value = normalize_pca_components(pca_components) if feature_preprocessor_name != "none" else None
    tuning_scoring_name = normalize_tuning_scoring(tuning_scoring)
    tuning_c_grid_values = parse_c_grid(tuning_c_grid)
    model, encoder, channel_names, n_window_samples, train_counts, tuning_metadata = _fit_decoder(
        train_raw=train_raw,
        train_events=train_events,
        onset_column=onset_column,
        label_column=label_column,
        train_window=train_window,
        picks=picks,
        baseline=baseline,
        decoder=decoder_name,
        emission_mode=emission_mode_name,
        max_iter=max_iter,
        feature_preprocessor=feature_preprocessor_name,
        pca_components=pca_components_value,
        tune_hyperparameters=tune_hyperparameters,
        tuning_cv_splits=tuning_cv_splits,
        tuning_scoring=tuning_scoring_name,
        tuning_c_grid=tuning_c_grid_values,
        demean_window=demean_window,
    )
    targets = list(target_classes or [str(encoder.classes_[0])])
    latency = float(np.mean(train_window)) if annotation_latency is None else annotation_latency
    split_id = _continuous_split_id(train_raw=train_raw, scan_raw=scan_raw, slice_seed=slice_seed)
    preprocessing_hash = _continuous_preprocessing_hash(
        train_window=train_window,
        picks=picks,
        baseline=baseline,
        demean_window=demean_window,
        scan_step=scan_step,
        n_window_samples=n_window_samples,
        channel_names=channel_names,
    )
    model_hash = _continuous_model_hash(
        decoder=decoder_name,
        emission_mode=emission_mode_name,
        max_iter=max_iter,
        train_window=train_window,
        feature_preprocessor=feature_preprocessor_name,
        pca_components=pca_components_value,
        tune_hyperparameters=tune_hyperparameters,
        tuning_cv_splits=tuning_cv_splits,
        tuning_scoring=tuning_scoring_name,
        tuning_c_grid=tuning_c_grid_values,
        tuning_metadata=tuning_metadata,
    )
    segments = build_scan_segments(
        scan_raw=scan_raw,
        scan_start=scan_start,
        scan_stop=scan_stop,
        slice_duration=slice_duration,
        slice_starts=slice_starts,
        slice_count=slice_count,
        slice_seed=slice_seed,
        scan_events=scan_events,
        onset_column=onset_column,
        label_column=label_column,
        target_classes=targets,
        threshold_window=threshold_window,
        detection_window=detection_window,
        require_target_event=require_target_event,
        exclude_events_from_threshold_window=exclude_events_from_threshold_window,
        stream_id=stream_id,
    )
    observations = _scan_raw_probabilities(
        scan_raw=scan_raw,
        model=model,
        encoder=encoder,
        channel_names=channel_names,
        n_window_samples=n_window_samples,
        segments=segments,
        scan_step=scan_step,
        decoder=decoder_name,
        emission_mode=emission_mode_name,
        feature_preprocessor=feature_preprocessor_name,
        pca_components=pca_components_value,
        tuning_metadata=tuning_metadata,
        subject=subject,
        train_window=train_window,
        baseline=baseline,
        demean_window=demean_window,
    )
    observations = _standardize_stream_observations(
        observations,
        subject=subject,
        split_id=split_id,
        slice_seed=slice_seed,
        decoder=decoder_name,
        emission_mode=emission_mode_name,
        train_time=float(np.mean(train_window)),
        preprocessing_hash=preprocessing_hash,
        model_hash=model_hash,
    )
    annotations = _annotation_table(
        scan_events=scan_events,
        segments=segments,
        onset_column=onset_column,
        label_column=label_column,
        target_classes=targets,
        annotation_latency=latency,
        detection_window=detection_window,
    )
    event_metrics = _held_out_event_metrics(
        model=model,
        encoder=encoder,
        scan_raw=scan_raw,
        scan_events=scan_events,
        onset_column=onset_column,
        label_column=label_column,
        train_window=train_window,
        picks=channel_names,
        baseline=baseline,
        decoder=decoder_name,
        emission_mode=emission_mode_name,
        feature_preprocessor=feature_preprocessor_name,
        pca_components=pca_components_value,
        tuning_metadata=tuning_metadata,
        demean_window=demean_window,
    )
    if not event_metrics.empty and subject is not None:
        event_metrics.insert(0, "subject", subject)
    if not train_counts.empty:
        train_counts.to_csv(out_dir / "training_class_counts.csv", index=False)

    thresholds = fit_stimulus_detection_thresholds(
        observations,
        threshold_window=threshold_window,
        threshold_quantile=threshold_quantile,
        threshold_method=threshold_method,
        score_mode=score_mode,
        target_classes=targets,
        stream_columns=("stream_id",),
        min_consecutive=min_consecutive,
        min_duration=min_duration,
    )
    events = detect_stimulus_events(
        observations,
        thresholds=thresholds,
        threshold_window=threshold_window,
        threshold_quantile=threshold_quantile,
        threshold_method=threshold_method,
        score_mode=score_mode,
        target_classes=targets,
        stream_columns=("stream_id",),
        detection_window=detection_window,
        min_consecutive=min_consecutive,
        min_duration=min_duration,
        merge_gap=merge_gap,
        refractory=refractory,
        conflict_resolution=conflict_resolution,
    )
    if not annotations.empty:
        events = match_stimulus_annotations(events, annotations, stream_columns=("stream_id",), match_tolerance=match_tolerance)
    summary = summarize_stimulus_events(
        events,
        annotations=annotations if not annotations.empty else None,
        observations=observations,
        stream_columns=("stream_id",),
    )

    observations.to_csv(out_dir / "stream_observations.csv", index=False)
    annotations.to_csv(out_dir / "stimulus_annotations.csv", index=False)
    thresholds.to_csv(out_dir / "stimulus_thresholds.csv", index=False)
    events.to_csv(out_dir / "stimulus_events.csv", index=False)
    summary.to_csv(out_dir / "stimulus_summary.csv", index=False)
    event_metrics.to_csv(out_dir / "heldout_event_metrics.csv", index=False)
    return ContinuousStimulusScanResult(observations, annotations, thresholds, events, summary, event_metrics)


def _optional_window(values: Sequence[float] | None) -> tuple[float, float] | None:
    return None if values is None else (float(values[0]), float(values[1]))


def _optional_baseline(values: Sequence[str] | None) -> tuple[float | None, float | None] | None:
    if values is None:
        return None
    parsed: list[float | None] = []
    for value in values:
        parsed.append(None if value.lower() == "none" else float(value))
    return (parsed[0], parsed[1])


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an event-locked decoder and scan a held-out continuous raw stream for stimulus-like events.")
    parser.add_argument("--train-raw", type=Path, required=True)
    parser.add_argument("--train-events", type=Path, required=True)
    parser.add_argument("--scan-raw", type=Path, required=True)
    parser.add_argument("--scan-events", type=Path)
    parser.add_argument("--events-sep", default="auto")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--onset-column", default="onset")
    parser.add_argument("--label-column", default="stimulus_class")
    parser.add_argument("--source-column")
    parser.add_argument("--positive-pattern")
    parser.add_argument("--negative-pattern")
    parser.add_argument("--positive-label", default="positive")
    parser.add_argument("--negative-label", default="negative")
    parser.add_argument("--case-sensitive", action="store_true")
    parser.add_argument("--target-class", action="append", dest="target_classes")
    parser.add_argument("--max-train-events-per-class", type=int)
    parser.add_argument("--train-window", nargs=2, type=float, required=True, metavar=("START", "STOP"))
    parser.add_argument("--picks", default="data")
    parser.add_argument("--epoch-baseline", nargs=2, metavar=("START", "STOP"))
    parser.add_argument("--decoder", default="logistic")
    parser.add_argument("--emission-mode", default="calibrated")
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--feature-preprocessor", choices=FEATURE_PREPROCESSOR_RUN_CHOICES, default="none")
    parser.add_argument(
        "--pca-components",
        help="PCA component count or explained-variance fraction. Only valid with --feature-preprocessor pca or pca-whiten.",
    )
    parser.add_argument(
        "--tune-hyperparameters",
        action="store_true",
        help="Use nested inner-CV hyperparameter selection on the event-locked training epochs before scanning.",
    )
    parser.add_argument("--tuning-cv-splits", type=int, default=3, help="Maximum number of inner CV folds for --tune-hyperparameters.")
    parser.add_argument(
        "--tuning-scoring",
        choices=TUNING_SCORING_CHOICES,
        default="accuracy",
        help="Inner-CV objective for --tune-hyperparameters.",
    )
    parser.add_argument(
        "--tuning-c-grid",
        default=",".join(str(value) for value in parse_c_grid(None)),
        help="Comma-separated positive C values for tuned logistic regression and linear SVM.",
    )
    parser.add_argument("--demean-window", action="store_true")
    parser.add_argument("--scan-step", type=float, default=0.025)
    parser.add_argument("--scan-start", type=float)
    parser.add_argument("--scan-stop", type=float)
    parser.add_argument("--slice-duration", type=float)
    parser.add_argument("--slice-start", action="append", type=float, dest="slice_starts")
    parser.add_argument("--slice-count", type=int)
    parser.add_argument("--slice-seed", type=int, default=13)
    parser.add_argument("--stream-id")
    parser.add_argument("--subject")
    parser.add_argument("--threshold-window", nargs=2, type=float, default=(0.0, 0.8), metavar=("START", "STOP"))
    parser.add_argument("--threshold-quantile", type=float, default=0.95)
    parser.add_argument("--threshold-method", choices=THRESHOLD_METHODS, default="max_run")
    parser.add_argument("--score-mode", choices=SCORE_MODES, default="class_probability")
    parser.add_argument("--detection-window", nargs=2, type=float, metavar=("START", "STOP"))
    parser.add_argument("--min-consecutive", type=int, default=1)
    parser.add_argument("--min-duration", type=float)
    parser.add_argument("--merge-gap", type=float)
    parser.add_argument("--refractory", type=float)
    parser.add_argument("--conflict-resolution", choices=CONFLICT_RESOLUTION_MODES, default="none")
    parser.add_argument("--match-tolerance", type=float, default=0.1)
    parser.add_argument("--annotation-latency", type=float)
    parser.add_argument("--require-target-event", action="store_true")
    parser.add_argument("--exclude-events-from-threshold-window", action="store_true")
    args = parser.parse_args()

    train_events = label_event_table(
        _read_table(args.train_events, args.events_sep),
        onset_column=args.onset_column,
        label_column=args.label_column,
        source_column=args.source_column,
        positive_pattern=args.positive_pattern,
        negative_pattern=args.negative_pattern,
        positive_label=args.positive_label,
        negative_label=args.negative_label,
        case_sensitive=args.case_sensitive,
    )
    train_events = _limit_per_class(
        train_events,
        onset_column=args.onset_column,
        label_column=args.label_column,
        max_events_per_class=args.max_train_events_per_class,
    )
    scan_events = None
    if args.scan_events is not None:
        scan_events = label_event_table(
            _read_table(args.scan_events, args.events_sep),
            onset_column=args.onset_column,
            label_column=args.label_column,
            source_column=args.source_column,
            positive_pattern=args.positive_pattern,
            negative_pattern=args.negative_pattern,
            positive_label=args.positive_label,
            negative_label=args.negative_label,
            case_sensitive=args.case_sensitive,
        )

    result = run_continuous_stimulus_scan(
        train_raw=args.train_raw,
        train_events=train_events,
        scan_raw=args.scan_raw,
        scan_events=scan_events,
        out_dir=args.out_dir,
        onset_column=args.onset_column,
        label_column=args.label_column,
        train_window=tuple(args.train_window),
        picks=args.picks,
        baseline=_optional_baseline(args.epoch_baseline),
        decoder=args.decoder,
        emission_mode=args.emission_mode,
        max_iter=args.max_iter,
        feature_preprocessor=args.feature_preprocessor,
        pca_components=args.pca_components,
        tune_hyperparameters=args.tune_hyperparameters,
        tuning_cv_splits=args.tuning_cv_splits,
        tuning_scoring=args.tuning_scoring,
        tuning_c_grid=args.tuning_c_grid,
        demean_window=args.demean_window,
        scan_step=args.scan_step,
        scan_start=args.scan_start,
        scan_stop=args.scan_stop,
        slice_duration=args.slice_duration,
        slice_starts=args.slice_starts,
        slice_count=args.slice_count,
        slice_seed=args.slice_seed,
        stream_id=args.stream_id,
        subject=args.subject,
        target_classes=args.target_classes,
        threshold_window=tuple(args.threshold_window),
        threshold_quantile=args.threshold_quantile,
        threshold_method=args.threshold_method,
        score_mode=args.score_mode,
        detection_window=_optional_window(args.detection_window),
        min_consecutive=args.min_consecutive,
        min_duration=args.min_duration,
        merge_gap=args.merge_gap,
        refractory=args.refractory,
        conflict_resolution=args.conflict_resolution,
        match_tolerance=args.match_tolerance,
        annotation_latency=args.annotation_latency,
        require_target_event=args.require_target_event,
        exclude_events_from_threshold_window=args.exclude_events_from_threshold_window,
    )
    print(f"Wrote stream observations: {args.out_dir / 'stream_observations.csv'}")
    print(f"Wrote stimulus events: {args.out_dir / 'stimulus_events.csv'}")
    print(f"Wrote stimulus summary: {args.out_dir / 'stimulus_summary.csv'}")
    if not result.event_metrics.empty:
        print(result.event_metrics.to_string(index=False))
    if not result.summary.empty:
        print(result.summary.to_string(index=False))


if __name__ == "__main__":
    main()
