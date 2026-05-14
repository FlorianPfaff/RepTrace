"""Helpers for canonical continuous-stimulus probability observations."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from reptrace.observations import ProbabilityObservationTable, stable_hash


def continuous_split_id(*, train_raw: Path, scan_raw: Path, slice_seed: int) -> str:
    """Return a stable split identifier for a train-run/scan-run continuous experiment."""
    return f"continuous-train-{train_raw.stem}-scan-{scan_raw.stem}-seed-{slice_seed}"


def continuous_preprocessing_hash(
    *,
    train_window: tuple[float, float],
    picks: str | Sequence[str],
    baseline: tuple[float | None, float | None] | None,
    demean_window: bool,
    scan_step: float,
    n_window_samples: int,
    channel_names: Sequence[str],
) -> str:
    """Return a deterministic hash for continuous-scan preprocessing choices."""
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


def continuous_model_hash(
    *,
    decoder: str,
    emission_mode: str,
    max_iter: int,
    train_window: tuple[float, float],
) -> str:
    """Return a deterministic hash for continuous-scan model choices."""
    return stable_hash(
        {
            "backend": "sklearn",
            "decoder": decoder,
            "emission_mode": emission_mode,
            "max_iter": max_iter,
            "train_window": train_window,
        }
    )


def standardize_continuous_observations(
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
    """Return continuous stream observations with canonical probability-observation columns."""
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
