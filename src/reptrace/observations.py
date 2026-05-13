"""Canonical probability-observation tables emitted by RepTrace workflows."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

STANDARD_OBSERVATION_COLUMNS: tuple[str, ...] = (
    "subject",
    "session",
    "stream_id",
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


def stable_hash(payload: object, *, length: int = 16) -> str:
    """Return a deterministic short hash for model/preprocessing provenance."""
    encoded = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:length]


def _probability_sort_key(column: str) -> tuple[int, str]:
    suffix = column.removeprefix("prob_class_")
    return (int(suffix), suffix) if suffix.isdigit() else (10_000, suffix)


def probability_columns(frame: pd.DataFrame) -> tuple[str, ...]:
    """Return ``prob_class_*`` columns in class-index order."""
    return tuple(sorted((column for column in frame.columns if column.startswith("prob_class_")), key=_probability_sort_key))


def _value_at(values: Sequence[object] | np.ndarray | pd.Series | None, index: int, default: object = "") -> object:
    if values is None:
        return default
    value = values[index]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _empty_or_missing(series: pd.Series) -> pd.Series:
    missing = series.isna()
    if series.dtype == object:
        missing |= series.astype(str).eq("")
    return missing


@dataclass(frozen=True)
class ProbabilityObservationTable:
    """CSV-friendly table of held-out or stream-level class probabilities.

    The wrapper deliberately stays lightweight: workflows can continue to own
    their domain-specific rows, while this class supplies a shared provenance
    contract, canonical column ordering, validation, and file writing.
    """

    frame: pd.DataFrame

    @classmethod
    def empty(cls) -> "ProbabilityObservationTable":
        """Return an empty table with the canonical base columns."""
        return cls(pd.DataFrame(columns=STANDARD_OBSERVATION_COLUMNS))

    @classmethod
    def from_rows(cls, rows: Sequence[Mapping[str, object]]) -> "ProbabilityObservationTable":
        """Create a table from row dictionaries."""
        return cls(pd.DataFrame(list(rows)))

    @classmethod
    def concat(cls, tables: Sequence["ProbabilityObservationTable"]) -> "ProbabilityObservationTable":
        """Concatenate tables while preserving row order."""
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
        fold: int | str,
        decoder: str,
        backend: str,
        emission_mode: str,
        time: float,
        original_indices: Sequence[int] | np.ndarray | None = None,
        subject: str | None = None,
        session_values: Sequence[object] | np.ndarray | pd.Series | None = None,
        group_values: Sequence[object] | np.ndarray | pd.Series | None = None,
        split_id: str = "",
        seed: int | str | None = None,
        train_time: float | None = None,
        window_start: float | None = None,
        window_stop: float | None = None,
        calibration_fold: str | int | None = None,
        preprocessing_hash: str = "",
        model_hash: str = "",
    ) -> "ProbabilityObservationTable":
        """Build canonical rows from held-out fold predictions."""
        probabilities = np.asarray(probabilities, dtype=float)
        test_labels = np.asarray(test_labels, dtype=int)
        predictions = np.asarray(predictions, dtype=int)
        test_indices = np.asarray(test_indices, dtype=int)
        if probabilities.ndim != 2:
            raise ValueError("probabilities must have shape (n_samples, n_classes).")
        if probabilities.shape[0] != len(test_labels) or probabilities.shape[0] != len(predictions) or probabilities.shape[0] != len(test_indices):
            raise ValueError("probabilities, labels, predictions, and test_indices must contain the same number of samples.")
        if probabilities.shape[1] != len(class_names):
            raise ValueError("class_names must match the number of probability columns.")
        if original_indices is None:
            original_indices = np.arange(int(test_indices.max(initial=-1)) + 1)
        original_indices = np.asarray(original_indices)

        rows: list[dict[str, object]] = []
        for local_position, filtered_index in enumerate(test_indices):
            true_label = int(test_labels[local_position])
            predicted_label = int(predictions[local_position])
            sample_index = _value_at(original_indices, int(filtered_index))
            row: dict[str, object] = {
                "subject": "" if subject is None else subject,
                "session": _value_at(session_values, int(filtered_index)),
                "fold": fold,
                "split_id": split_id,
                "seed": "" if seed is None else seed,
                "decoder": decoder,
                "backend": backend,
                "emission_mode": emission_mode,
                "train_time": float(time if train_time is None else train_time),
                "test_time": float(time),
                "time": float(time),
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
                "preprocessing_hash": preprocessing_hash,
                "model_hash": model_hash,
            }
            if group_values is not None:
                row["group"] = _value_at(group_values, int(filtered_index))
            for class_index, class_name in enumerate(class_names):
                row[f"class_{class_index}"] = str(class_name)
                row[f"prob_class_{class_index}"] = float(probabilities[local_position, class_index])
            rows.append(row)
        return cls.from_rows(rows).standardized()

    @property
    def probability_columns(self) -> tuple[str, ...]:
        """Return probability columns in class-index order."""
        return probability_columns(self.frame)

    def standardized(self, *, defaults: Mapping[str, object] | None = None) -> "ProbabilityObservationTable":
        """Return a table with canonical provenance columns and ordering.

        Missing ``test_time`` and ``train_time`` are mirrored from ``time`` when
        available. Remaining missing canonical columns are filled from
        ``defaults`` or as empty strings.
        """
        result = self.frame.copy()
        defaults = dict(defaults or {})
        if "time" not in result.columns and "test_time" in result.columns:
            result["time"] = result["test_time"]
        if "test_time" not in result.columns and "time" in result.columns:
            result["test_time"] = result["time"]
        if "train_time" not in result.columns and "time" in result.columns:
            result["train_time"] = result["time"]
        for column in STANDARD_OBSERVATION_COLUMNS:
            if column not in result.columns:
                result[column] = defaults.get(column, "")
        for column, value in defaults.items():
            if column not in result.columns:
                result[column] = value
                continue
            missing = _empty_or_missing(result[column])
            if bool(missing.any()):
                result.loc[missing, column] = value
        ordered = [column for column in STANDARD_OBSERVATION_COLUMNS if column in result.columns]
        ordered.extend(column for column in result.columns if column not in ordered)
        return ProbabilityObservationTable(result.loc[:, ordered])

    def validate(self, **kwargs: object):
        """Validate the table with ``reptrace.observation_schema``."""
        from reptrace.observation_schema import validate_probability_observations

        return validate_probability_observations(self.frame, **kwargs)

    def to_csv(self, path: str | Path) -> None:
        """Write observations to CSV."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.frame.to_csv(path, index=False)
