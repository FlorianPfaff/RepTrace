from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

_MODEL_PREPROCESSING: dict[int, tuple[tuple[float, float], tuple[float | None, float | None] | None]] = {}


def _apply_epoch_baseline(
    data: np.ndarray,
    *,
    train_window: tuple[float, float],
    baseline: tuple[float | None, float | None] | None,
    sfreq: float,
) -> np.ndarray:
    """Apply MNE-style epoch baseline correction to one scan window."""

    if baseline is None:
        return data
    if data.ndim != 2:
        raise ValueError("Baseline correction expects a channel-by-time window.")
    bmin, bmax = baseline
    win_start, win_stop = train_window
    bmin = win_start if bmin is None else float(bmin)
    bmax = win_stop if bmax is None else float(bmax)
    if bmin > bmax:
        raise ValueError("Baseline start must be less than or equal to baseline stop.")
    relative_times = win_start + np.arange(data.shape[1], dtype=float) / float(sfreq)
    baseline_mask = (relative_times >= bmin - 1e-12) & (relative_times <= bmax + 1e-12)
    if not np.any(baseline_mask):
        raise ValueError(f"Baseline interval ({bmin}, {bmax}) does not overlap scan window {train_window}.")
    return data - data[:, baseline_mask].mean(axis=1, keepdims=True)


def _remember_model_preprocessing(
    model: object,
    *,
    train_window: tuple[float, float],
    baseline: tuple[float | None, float | None] | None,
) -> None:
    train_window = (float(train_window[0]), float(train_window[1]))
    if baseline is not None:
        baseline = (
            None if baseline[0] is None else float(baseline[0]),
            None if baseline[1] is None else float(baseline[1]),
        )
    _MODEL_PREPROCESSING[id(model)] = (train_window, baseline)
    try:
        setattr(model, "_reptrace_train_window", train_window)
        setattr(model, "_reptrace_epoch_baseline", baseline)
    except Exception:  # pragma: no cover - some estimator wrappers may reject dynamic attrs
        pass


def _get_model_preprocessing(model: object) -> tuple[tuple[float, float] | None, tuple[float | None, float | None] | None]:
    train_window = getattr(model, "_reptrace_train_window", None)
    baseline = getattr(model, "_reptrace_epoch_baseline", None)
    if train_window is not None:
        return train_window, baseline
    return _MODEL_PREPROCESSING.get(id(model), (None, None))


def install() -> None:
    """Patch the continuous scanner to baseline-correct scan windows like epochs."""

    from reptrace import continuous_stimulus_scan as scan

    if getattr(scan, "_REPTRACE_CONTINUOUS_BASELINE_PATCHED", False):
        return

    original_fit_decoder = scan._fit_decoder
    original_scan_raw_probabilities = scan._scan_raw_probabilities

    def _fit_decoder_with_preprocessing_metadata(*args: Any, **kwargs: Any):
        result = original_fit_decoder(*args, **kwargs)
        _remember_model_preprocessing(
            result[0],
            train_window=kwargs["train_window"],
            baseline=kwargs.get("baseline"),
        )
        return result

    def _scan_raw_probabilities_with_epoch_baseline(
        *,
        scan_raw: Path,
        model: object,
        encoder: LabelEncoder,
        channel_names: Sequence[str],
        n_window_samples: int,
        segments: Sequence[scan.ScanSegment],
        scan_step: float,
        decoder: str,
        emission_mode: str,
        subject: str | None,
        demean_window: bool,
    ) -> pd.DataFrame:
        train_window, baseline = _get_model_preprocessing(model)
        if baseline is None or train_window is None:
            return original_scan_raw_probabilities(
                scan_raw=scan_raw,
                model=model,
                encoder=encoder,
                channel_names=channel_names,
                n_window_samples=n_window_samples,
                segments=segments,
                scan_step=scan_step,
                decoder=decoder,
                emission_mode=emission_mode,
                subject=subject,
                demean_window=demean_window,
            )

        if scan_step <= 0:
            raise ValueError("scan_step must be positive.")
        raw = scan.mne.io.read_raw_fif(scan_raw, preload=False, verbose="error")
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
                data = _apply_epoch_baseline(data, train_window=train_window, baseline=baseline, sfreq=sfreq)
                if demean_window:
                    data = data - data.mean(axis=1, keepdims=True)
                window_start = start_sample / sfreq - segment.output_origin
                window_stop = stop_sample / sfreq - segment.output_origin
                features.append(data.reshape(-1))
                metadata.append((float(center) - segment.output_origin, window_start, window_stop))
            if not features:
                continue
            probabilities = scan.predict_emission_probabilities(model, np.vstack(features), emission_mode=emission_mode)
            predictions = probabilities.argmax(axis=1)
            for sample_index, ((time, window_start, window_stop), probabilities_row, predicted_label) in enumerate(
                zip(metadata, probabilities, predictions, strict=True)
            ):
                row = {
                    "stream_id": segment.stream_id,
                    "decoder": decoder,
                    "emission_mode": emission_mode,
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
                for class_index, class_name in enumerate(classes):
                    row[f"class_{class_index}"] = str(class_name)
                    row[f"prob_class_{class_index}"] = float(probabilities_row[class_index])
                rows.append(row)
        return pd.DataFrame(rows)

    scan._fit_decoder = _fit_decoder_with_preprocessing_metadata
    scan._scan_raw_probabilities = _scan_raw_probabilities_with_epoch_baseline
    scan._apply_epoch_baseline = _apply_epoch_baseline
    scan._REPTRACE_CONTINUOUS_BASELINE_PATCHED = True
