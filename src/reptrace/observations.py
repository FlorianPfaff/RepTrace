"""Canonical probability-observation tables emitted by RepTrace decoders."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

STANDARD_OBSERVATION_COLUMNS: tuple[str, ...] = (
    "subject",
    "session",
    "fold",
    "split_id",
    "seed",
    "decoder",
    "backend",
    "emission_mode",
    "train_time",
    "test_time",
    "time",
    "window_start",
    "window_stop",
    "sample_index",
    "sequence_id",
    "true_label",
    "true_class",
    "predicted_label",
    "predicted_class",
    "probability_true_class",
    "confidence",
    "is_correct",
    "calibration_fold",
    "preprocessing_hash",
    "model_hash",
)


def stable_hash(payload: Mapping[str, Any] | Sequence[Any] | str | int | float | None, *, length: int = 16) -> str:
    """Return a deterministic short hash for model/preprocessing provenance."""
    encoded = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:length]


def _optional_value(values: Sequence[object] | np.ndarray | None, index: int, default: object = "") -> object:
    if values is None:
        return default
    return values[index]


def _as_python_scalar(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    return value


@dataclass(frozen=True)
class ProbabilityObservationTable:
    """A standardized held-out probability table produced by decoder backends.

    The table is intentionally CSV-friendly. Every backend should emit these
    shared provenance columns plus one ``prob_class_*`` column per class.
    """

    frame: pd.DataFrame

    @classmethod
    def empty(cls) -> "ProbabilityObservationTable":
        """Return an empty canonical observation table."""
        return cls(pd.DataFrame(columns=STANDARD_OBSERVATION_COLUMNS))

    @classmethod
    def concat(cls, tables: Sequence["ProbabilityObservationTable"]) -> "ProbabilityObservationTable":
        """Concatenate observation tables while preserving row order."""
        if not tables:
            return cls.empty()
        return cls(pd.concat([table.frame for table in tables], ignore_index=True))

    @classmethod
    def from_decoded_fold(
        cls,
        *,
        probabilities: np.ndarray,
        test_labels: Sequence[int] | np.ndarray,
        predictions: Sequence[int] | np.ndarray,
        class_names: Sequence[object],
        test_indices: Sequence[int] | np.ndarray,
        original_indices: Sequence[int] | np.ndarray | None = None,
        subject: str | None = None,
        session_values: Sequence[object] | np.ndarray | None = None,
        group_values: Sequence[object] | np.ndarray | None = None,
        fold: int,
        split_id: str,
        seed: int | None,
        decoder: str,
        backend: str,
        emission_mode: str,
        train_time: float | None,
        test_time: float,
        window_start: float | None = None,
        window_stop: float | None = None,
        calibration_fold: str | int | None = None,
        preprocessing_hash: str | None = None,
        model_hash: str | None = None,
    ) -> "ProbabilityObservationTable":
        """Build a canonical table from held-out fold predictions."""
        probabilities = np.asarray(probabilities, dtype=float)
        test_labels = np.asarray(test_labels, dtype=int)
        predictions = np.asarray(predictions, dtype=int)
        test_indices = np.asarray(test_indices, dtype=int)
        if original_indices is None:
            original_indices = np.arange(max(test_indices.max(initial=0) + 1, len(test_indices)))
        original_indices = np.asarray(original_indices)

        if probabilities.ndim != 2:
            raise ValueError("probabilities must have shape (n_samples, n_classes).")
        if probabilities.shape[0] != len(test_labels) or probabilities.shape[0] != len(predictions) or probabilities.shape[0] != len(test_indices):
            raise ValueError("probabilities, labels, predictions, and test_indices must contain the same number of samples.")
        if probabilities.shape[1] != len(class_names):
            raise ValueError("class_names must match the number of probability columns.")

        rows: list[dict[str, object]] = []
        for local_position, filtered_index in enumerate(test_indices):
            true_label = int(test_labels[local_position])
            predicted_label = int(predictions[local_position])
            sample_index = _as_python_scalar(original_indices[filtered_index])
            row: dict[str, object] = {
                "subject": "" if subject is None else subject,
                "session": _as_python_scalar(_optional_value(session_values, int(filtered_index))),
                "fold": int(fold),
                "split_id": split_id,
                "seed": "" if seed is None else int(seed),
                "decoder": decoder,
                "backend": backend,
                "emission_mode": emission_mode,
                "train_time": "" if train_time is None else float(train_time),
                "test_time": float(test_time),
                "time": float(test_time),
                "window_start": "" if window_start is None else float(window_start),
                "window_stop": "" if window_stop is None else float(window_stop),
                "sample_index": sample_index,
                "sequence_id": sample_index,
                "true_label": true_label,
                "true_class": str(class_names[true_label]),
                "predicted_label": predicted_label,
                "predicted_class": str(class_names[predicted_label]),
                "probability_true_class": float(probabilities[local_position, true_label]),
                "confidence": float(probabilities[local_position].max()),
                "is_correct": bool(predicted_label == true_label),
                "calibration_fold": "" if calibration_fold is None else calibration_fold,
                "preprocessing_hash": "" if preprocessing_hash is None else preprocessing_hash,
                "model_hash": "" if model_hash is None else model_hash,
            }
            if group_values is not None:
                row["group"] = _as_python_scalar(_optional_value(group_values, int(filtered_index)))
            for class_index, class_name in enumerate(class_names):
                row[f"class_{class_index}"] = str(class_name)
                row[f"prob_class_{class_index}"] = float(probabilities[local_position, class_index])
            rows.append(row)

        return cls(pd.DataFrame(rows))

    @property
    def probability_columns(self) -> tuple[str, ...]:
        """Return class-probability columns in numeric class order."""
        columns = [column for column in self.frame.columns if column.startswith("prob_class_")]

        def sort_key(column: str) -> tuple[int, str]:
            suffix = column.removeprefix("prob_class_")
            return (int(suffix), suffix) if suffix.isdigit() else (10_000, suffix)

        return tuple(sorted(columns, key=sort_key))

    def validate(self):
        """Validate this table with the existing observation-schema validator."""
        from reptrace.observation_schema import validate_probability_observations

        return validate_probability_observations(self.frame)

    def to_csv(self, path: str | Path) -> None:
        """Write observations to CSV."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.frame.to_csv(path, index=False)
