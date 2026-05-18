from __future__ import annotations

from typing import Any

_MODEL_PREPROCESSING: dict[int, tuple[tuple[float, float], tuple[float | None, float | None] | None]] = {}


def _apply_epoch_baseline(data, *, train_window, baseline, sfreq):
    """Apply MNE-style epoch baseline correction to one scan window."""

    import numpy as np

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


def _remember_model_preprocessing(model: object, *, train_window, baseline) -> None:
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


def _get_model_preprocessing(model: object):
    train_window = getattr(model, "_reptrace_train_window", None)
    baseline = getattr(model, "_reptrace_epoch_baseline", None)
    if train_window is not None:
        return train_window, baseline
    return _MODEL_PREPROCESSING.get(id(model), (None, None))


class _BaselineCorrectingRaw:
    def __init__(self, raw, *, train_window, baseline):
        self._raw = raw
        self._train_window = train_window
        self._baseline = baseline

    def __getattr__(self, name: str):
        return getattr(self._raw, name)

    def pick(self, *args: Any, **kwargs: Any):
        self._raw.pick(*args, **kwargs)
        return self

    def get_data(self, *args: Any, **kwargs: Any):
        data = self._raw.get_data(*args, **kwargs)
        return _apply_epoch_baseline(
            data,
            train_window=self._train_window,
            baseline=self._baseline,
            sfreq=float(self._raw.info["sfreq"]),
        )


def install() -> None:
    """Patch the continuous scanner to baseline-correct scan windows like epochs."""

    from reptrace import continuous_stimulus_scan as scan

    if getattr(scan, "_REPTRACE_CONTINUOUS_BASELINE_PATCHED", False):
        return

    original_fit_decoder = scan._fit_decoder
    original_scan_raw_probabilities = scan._scan_raw_probabilities

    def _fit_decoder_with_preprocessing_metadata(*args: Any, **kwargs: Any):
        result = original_fit_decoder(*args, **kwargs)
        _remember_model_preprocessing(result[0], train_window=kwargs["train_window"], baseline=kwargs.get("baseline"))
        return result

    def _scan_raw_probabilities_with_epoch_baseline(*args: Any, **kwargs: Any):
        train_window, baseline = _get_model_preprocessing(kwargs["model"])
        if baseline is None or train_window is None:
            return original_scan_raw_probabilities(*args, **kwargs)

        original_read_raw_fif = scan.mne.io.read_raw_fif

        def read_raw_fif_with_baseline(*read_args: Any, **read_kwargs: Any):
            raw = original_read_raw_fif(*read_args, **read_kwargs)
            return _BaselineCorrectingRaw(raw, train_window=train_window, baseline=baseline)

        scan.mne.io.read_raw_fif = read_raw_fif_with_baseline
        try:
            return original_scan_raw_probabilities(*args, **kwargs)
        finally:
            scan.mne.io.read_raw_fif = original_read_raw_fif

    scan._fit_decoder = _fit_decoder_with_preprocessing_metadata
    scan._scan_raw_probabilities = _scan_raw_probabilities_with_epoch_baseline
    scan._apply_epoch_baseline = _apply_epoch_baseline
    scan._REPTRACE_CONTINUOUS_BASELINE_PATCHED = True
