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
