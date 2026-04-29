from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def make_logistic_decoder(max_iter: int = 1000):
    """Create the default calibrated-probability baseline decoder."""
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            class_weight="balanced",
            max_iter=max_iter,
            solver="lbfgs",
        ),
    )


def make_cross_validator(labels: np.ndarray, groups: np.ndarray | None, n_splits: int):
    """Create stratified CV splits, optionally preserving group boundaries."""
    _, class_counts = np.unique(labels, return_counts=True)
    if len(class_counts) < 2:
        raise ValueError("Need at least two classes for decoding.")
    if np.min(class_counts) < n_splits:
        raise ValueError(
            f"Need at least {n_splits} examples per class; smallest class has {np.min(class_counts)}."
        )
    if groups is not None:
        unique_groups = np.unique(groups)
        if len(unique_groups) < n_splits:
            raise ValueError(
                f"Need at least {n_splits} groups for grouped CV, found {len(unique_groups)}."
            )
        return StratifiedGroupKFold(n_splits=n_splits).split(
            np.zeros_like(labels),
            labels,
            groups,
        )
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=13).split(
        np.zeros_like(labels),
        labels,
    )


def time_windows(times: np.ndarray, window_ms: float, step_ms: float) -> list[tuple[int, int, float]]:
    """Return sample index windows and their center times for time-resolved decoding."""
    if times.ndim != 1:
        raise ValueError("times must be one-dimensional")
    if len(times) < 2:
        raise ValueError("times must contain at least two samples")
    if window_ms <= 0 or step_ms <= 0:
        raise ValueError("window_ms and step_ms must be positive")

    sfreq = 1000.0 / np.median(np.diff(times * 1000.0))
    window_samples = max(1, int(round((window_ms / 1000.0) * sfreq)))
    step_samples = max(1, int(round((step_ms / 1000.0) * sfreq)))
    windows = []
    for start in range(0, len(times) - window_samples + 1, step_samples):
        stop = start + window_samples
        center = float(np.mean(times[start:stop]))
        windows.append((start, stop, center))
    return windows
