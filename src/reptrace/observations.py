from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd


def stable_hash(payload: Mapping[str, object] | Sequence[object] | str) -> str:
    """Return a short stable hash for model and preprocessing metadata."""

    encoded = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


@dataclass(frozen=True)
class ProbabilityObservationValidation:
    """Validation result for a probability-observation table."""

    is_valid: bool
    errors: tuple[str, ...] = ()


@dataclass(frozen=True)
class ProbabilityObservationTable:
    """Small wrapper for canonical held-out probability observation rows."""

    frame: pd.DataFrame

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
        session_values: Sequence[object] | np.ndarray | None = None,
        group_values: Sequence[object] | np.ndarray | None = None,
        split_id: str = "",
        seed: int | str | None = None,
        train_time: float | None = None,
        window_start: float | None = None,
        window_stop: float | None = None,
        calibration_fold: str | int | None = None,
        preprocessing_hash: str = "",
        model_hash: str = "",
    ) -> ProbabilityObservationTable:
        """Build canonical rows from one held-out decoder fold."""

        probabilities = np.asarray(probabilities, dtype=float)
        test_labels = np.asarray(test_labels)
        predictions = np.asarray(predictions)
        test_indices = np.asarray(test_indices)
        if probabilities.ndim != 2:
            raise ValueError("probabilities must be a two-dimensional array.")
        if len(test_labels) != len(predictions) or len(test_labels) != len(probabilities) or len(test_labels) != len(test_indices):
            raise ValueError("test_labels, predictions, probabilities, and test_indices must have matching row counts.")
        if probabilities.shape[1] != len(class_names):
            raise ValueError("probability column count must match class_names.")

        original_indices_array = None if original_indices is None else np.asarray(original_indices)
        session_array = None if session_values is None else np.asarray(session_values)
        group_array = None if group_values is None else np.asarray(group_values)

        rows = []
        for local_position, filtered_index in enumerate(test_indices):
            true_label = int(test_labels[local_position])
            predicted_label = int(predictions[local_position])
            sample_index = (
                int(original_indices_array[filtered_index])
                if original_indices_array is not None
                else int(filtered_index)
            )
            row = {
                "fold": fold,
                "split_id": split_id,
                "seed": "" if seed is None else seed,
                "decoder": decoder,
                "backend": backend,
                "emission_mode": emission_mode,
                "train_time": time if train_time is None else train_time,
                "test_time": time,
                "time": time,
                "window_start": "" if window_start is None else window_start,
                "window_stop": "" if window_stop is None else window_stop,
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
            if subject is not None:
                row = {"subject": subject, **row}
            if session_array is not None:
                row["session"] = session_array[filtered_index]
            if group_array is not None:
                row["group"] = group_array[filtered_index]
            for class_index, class_name in enumerate(class_names):
                row[f"class_{class_index}"] = str(class_name)
                row[f"prob_class_{class_index}"] = float(probabilities[local_position, class_index])
            rows.append(row)
        return cls(pd.DataFrame(rows))

    @classmethod
    def concat(cls, tables: Sequence[ProbabilityObservationTable]) -> ProbabilityObservationTable:
        """Concatenate observation tables."""

        frames = [table.frame for table in tables if not table.frame.empty]
        if not frames:
            return cls(pd.DataFrame())
        return cls(pd.concat(frames, ignore_index=True))

    def standardized(self, *, defaults: Mapping[str, object] | None = None) -> ProbabilityObservationTable:
        """Return a copy with default metadata columns filled when absent."""

        frame = self.frame.copy()
        for column, value in (defaults or {}).items():
            if column not in frame.columns:
                frame[column] = value
            else:
                frame[column] = frame[column].fillna(value)
        return ProbabilityObservationTable(frame)

    def validate(self) -> ProbabilityObservationValidation:
        """Validate the minimal canonical observation-table schema."""

        required = (
            "fold",
            "decoder",
            "backend",
            "emission_mode",
            "time",
            "sample_index",
            "sequence_id",
            "true_label",
            "predicted_label",
            "probability_true_class",
            "confidence",
        )
        errors = [f"missing column: {column}" for column in required if column not in self.frame.columns]
        probability_columns = [column for column in self.frame.columns if column.startswith("prob_class_")]
        if not probability_columns:
            errors.append("missing probability columns: prob_class_*")
        return ProbabilityObservationValidation(is_valid=not errors, errors=tuple(errors))
