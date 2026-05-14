from __future__ import annotations

import argparse
import glob
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

ObservationProfile = Literal["generic", "canonical", "temporal-model", "stimulus-detection"]

PROFILES: tuple[ObservationProfile, ...] = ("generic", "canonical", "temporal-model", "stimulus-detection")
DEFAULT_PROBABILITY_TOLERANCE = 1e-3
DEFAULT_METADATA_COLUMNS = ("subject", "decoder", "emission_mode")
REPORT_COLUMNS = ("severity", "code", "message", "column", "row", "value")
STREAM_FALLBACK_COLUMNS = ("stream_id", "sequence_id", "sample_index")
CANONICAL_REQUIRED_COLUMNS = (
    "time",
    "decoder",
    "backend",
    "emission_mode",
    "split_id",
    "seed",
    "train_time",
    "test_time",
    "preprocessing_hash",
    "model_hash",
)
CANONICAL_RECOMMENDED_COLUMNS = ("subject", "session", "fold", "calibration_fold")
CANONICAL_NUMERIC_COLUMNS = ("seed", "train_time", "test_time")


@dataclass(frozen=True)
class ObservationValidationIssue:
    """One structural or quality issue found in a probability-observation table."""

    severity: str
    code: str
    message: str
    column: str | None = None
    row: int | None = None
    value: object | None = None

    def as_dict(self) -> dict[str, object]:
        """Return a CSV-friendly representation of the issue."""
        return {
            "severity": self.severity,
            "code": self.code,
            "message": self.message,
            "column": "" if self.column is None else self.column,
            "row": "" if self.row is None else self.row,
            "value": "" if self.value is None else self.value,
        }


@dataclass(frozen=True)
class ObservationValidationReport:
    """Validation result for a RepTrace probability-observation table."""

    profile: ObservationProfile
    n_rows: int
    probability_columns: tuple[str, ...]
    issues: tuple[ObservationValidationIssue, ...]

    @property
    def errors(self) -> tuple[ObservationValidationIssue, ...]:
        """Return blocking validation issues."""
        return tuple(issue for issue in self.issues if issue.severity == "error")

    @property
    def warnings(self) -> tuple[ObservationValidationIssue, ...]:
        """Return non-blocking validation issues."""
        return tuple(issue for issue in self.issues if issue.severity == "warning")

    @property
    def is_valid(self) -> bool:
        """Return true when the report has no blocking errors."""
        return not self.errors

    def to_frame(self) -> pd.DataFrame:
        """Return validation issues as a tabular report."""
        if not self.issues:
            return pd.DataFrame(columns=REPORT_COLUMNS)
        return pd.DataFrame([issue.as_dict() for issue in self.issues], columns=REPORT_COLUMNS)

    def raise_for_errors(self) -> None:
        """Raise a compact ValueError if blocking validation errors are present."""
        if self.errors:
            messages = "; ".join(issue.message for issue in self.errors[:5])
            if len(self.errors) > 5:
                messages += f"; and {len(self.errors) - 5} more error(s)"
            raise ValueError(messages)


def _expand_paths(patterns: Sequence[str | Path]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = sorted(glob.glob(str(pattern)))
        if matches:
            paths.extend(Path(match) for match in matches)
        else:
            paths.append(Path(pattern))
    return paths


def probability_columns(frame: pd.DataFrame) -> list[str]:
    """Return ``prob_class_*`` columns in class-index order."""
    columns = [column for column in frame.columns if column.startswith("prob_class_")]
    if not columns:
        raise ValueError("Observation tables must contain probability columns named 'prob_class_*'.")

    def sort_key(column: str) -> tuple[int, str]:
        suffix = column.removeprefix("prob_class_")
        return (int(suffix), suffix) if suffix.isdigit() else (10_000, suffix)

    return sorted(columns, key=sort_key)


def _try_probability_columns(frame: pd.DataFrame, issues: list[ObservationValidationIssue]) -> list[str]:
    try:
        return probability_columns(frame)
    except ValueError as exc:
        issues.append(ObservationValidationIssue("error", "missing_probability_columns", str(exc)))
        return []


def _issue(
    issues: list[ObservationValidationIssue],
    severity: str,
    code: str,
    message: str,
    *,
    column: str | None = None,
    row: int | None = None,
    value: object | None = None,
) -> None:
    issues.append(ObservationValidationIssue(severity, code, message, column=column, row=row, value=value))


def _present_mask(frame: pd.DataFrame, column: str) -> pd.Series:
    values = frame[column]
    present = values.notna()
    if values.dtype == object or pd.api.types.is_string_dtype(values):
        present &= values.astype("string").str.strip().ne("").fillna(False)
    return present


def _numeric_series(frame: pd.DataFrame, column: str, issues: list[ObservationValidationIssue], *, allow_nan: bool = False) -> pd.Series:
    values = pd.to_numeric(frame[column], errors="coerce")
    bad_mask = values.isna() if not allow_nan else values.isna() & frame[column].notna()
    for row_index, value in frame.loc[bad_mask, column].head(20).items():
        _issue(
            issues,
            "error",
            "non_numeric_value",
            f"Column '{column}' contains a non-numeric value.",
            column=column,
            row=int(row_index),
            value=value,
        )
    if int(bad_mask.sum()) > 20:
        _issue(
            issues,
            "error",
            "non_numeric_value_truncated",
            f"Column '{column}' contains {int(bad_mask.sum())} non-numeric values; first 20 are listed.",
            column=column,
        )
    return values


def _validate_time(frame: pd.DataFrame, issues: list[ObservationValidationIssue]) -> None:
    if "time" not in frame.columns:
        _issue(issues, "error", "missing_time", "Observation tables must contain a numeric 'time' column.", column="time")
        return
    values = _numeric_series(frame, "time", issues)
    finite_mask = values.notna() & ~np.isfinite(values.to_numpy(dtype=float))
    for row_index, value in frame.loc[finite_mask, "time"].head(20).items():
        _issue(
            issues,
            "error",
            "non_finite_time",
            "Column 'time' must contain finite numeric values.",
            column="time",
            row=int(row_index),
            value=value,
        )


def _probability_frame(frame: pd.DataFrame, prob_columns: Sequence[str], issues: list[ObservationValidationIssue]) -> pd.DataFrame:
    numeric_columns = []
    for column in prob_columns:
        numeric_columns.append(_numeric_series(frame, column, issues, allow_nan=True))
    if not numeric_columns:
        return pd.DataFrame(index=frame.index)
    probabilities = pd.concat(numeric_columns, axis=1)
    probabilities.columns = list(prob_columns)
    return probabilities


def _validate_probabilities(
    probabilities: pd.DataFrame,
    issues: list[ObservationValidationIssue],
    *,
    tolerance: float,
    require_normalized: bool,
) -> None:
    if probabilities.empty:
        return

    for column in probabilities.columns:
        negative_mask = probabilities[column].notna() & (probabilities[column] < 0.0)
        for row_index, value in probabilities.loc[negative_mask, column].head(20).items():
            _issue(
                issues,
                "error",
                "negative_probability",
                f"Probability column '{column}' contains a negative value.",
                column=column,
                row=int(row_index),
                value=float(value),
            )

    all_missing = probabilities.isna().all(axis=1)
    for row_index in probabilities.loc[all_missing].head(20).index:
        _issue(
            issues,
            "error",
            "missing_probability_row",
            "Each observation row must contain at least one probability value.",
            row=int(row_index),
        )

    valid = probabilities.dropna(how="all")
    if valid.empty:
        return
    row_sums = valid.fillna(0.0).sum(axis=1)
    deviations = (row_sums - 1.0).abs()
    bad_sums = deviations > tolerance
    severity = "error" if require_normalized else "warning"
    code = "probability_sum_error" if require_normalized else "probability_sum_warning"
    for row_index, _deviation in deviations.loc[bad_sums].head(20).items():
        _issue(
            issues,
            severity,
            code,
            f"Probability row sums should be 1.0 within tolerance {tolerance:g}.",
            row=int(row_index),
            value=float(row_sums.loc[row_index]),
        )
    if int(bad_sums.sum()) > 20:
        _issue(
            issues,
            severity,
            f"{code}_truncated",
            f"{int(bad_sums.sum())} probability rows have sums outside tolerance {tolerance:g}; first 20 are listed.",
        )


def _validate_metadata(frame: pd.DataFrame, issues: list[ObservationValidationIssue]) -> None:
    for column in DEFAULT_METADATA_COLUMNS:
        if column not in frame.columns:
            _issue(
                issues,
                "warning",
                f"missing_{column}",
                f"Column '{column}' is recommended for reproducible downstream grouping; readers may fill a default.",
                column=column,
            )
    if "confidence" in frame.columns:
        confidence = _numeric_series(frame, "confidence", issues, allow_nan=True)
        outside = confidence.notna() & ((confidence < 0.0) | (confidence > 1.0))
        for row_index, value in confidence.loc[outside].head(20).items():
            _issue(
                issues,
                "warning",
                "confidence_outside_unit_interval",
                "Column 'confidence' is expected to lie in [0, 1] when present.",
                column="confidence",
                row=int(row_index),
                value=float(value),
            )


def _validate_class_columns(frame: pd.DataFrame, prob_columns: Sequence[str], issues: list[ObservationValidationIssue]) -> None:
    prob_suffixes = {column.removeprefix("prob_class_") for column in prob_columns}
    for column in frame.columns:
        if column.startswith("class_"):
            suffix = column.removeprefix("class_")
            if suffix not in prob_suffixes:
                _issue(
                    issues,
                    "warning",
                    "unmatched_class_column",
                    f"Class-name column '{column}' has no matching 'prob_class_{suffix}' column.",
                    column=column,
                )


def _sequence_key_columns(frame: pd.DataFrame) -> list[str]:
    if "sequence_id" in frame.columns:
        return [column for column in ("subject", "fold", "sequence_id") if column in frame.columns]
    if "sample_index" in frame.columns:
        return [column for column in ("subject", "fold", "sample_index") if column in frame.columns]
    return []


def _validate_group_columns(frame: pd.DataFrame, group_columns: Sequence[str] | None, issues: list[ObservationValidationIssue]) -> None:
    if group_columns is None:
        return
    for column in group_columns:
        if column not in frame.columns:
            _issue(issues, "error", "missing_group_column", f"Requested group column '{column}' is missing.", column=column)


def _validate_temporal_profile(frame: pd.DataFrame, issues: list[ObservationValidationIssue]) -> None:
    if "sequence_id" not in frame.columns and "sample_index" not in frame.columns:
        _issue(
            issues,
            "error",
            "missing_sequence_identifier",
            "Temporal-model observations must contain 'sequence_id' or 'sample_index'.",
        )
        return

    keys = _sequence_key_columns(frame)
    if not keys or "time" not in frame.columns:
        return

    sequence_sizes = frame.groupby(keys, dropna=False).size()
    if not (sequence_sizes >= 2).any():
        _issue(
            issues,
            "error",
            "no_multi_point_sequence",
            "Temporal-model observations need at least one sequence with two or more time points.",
        )
    short_count = int((sequence_sizes < 2).sum())
    if short_count:
        _issue(
            issues,
            "warning",
            "single_point_sequences",
            f"{short_count} temporal sequence(s) contain only one time point and will be ignored by temporal modeling.",
        )

    duplicate_keys = [*keys, "time"]
    duplicate_count = int(frame.duplicated(duplicate_keys, keep=False).sum())
    if duplicate_count:
        _issue(
            issues,
            "warning",
            "duplicate_sequence_time",
            f"{duplicate_count} row(s) share the same sequence key and time value.",
        )


def _validate_stimulus_profile(frame: pd.DataFrame, stream_columns: Sequence[str] | None, issues: list[ObservationValidationIssue]) -> None:
    if stream_columns is not None:
        for column in stream_columns:
            if column not in frame.columns:
                _issue(issues, "error", "missing_stream_column", f"Requested stream column '{column}' is missing.", column=column)
        return

    if not any(column in frame.columns for column in STREAM_FALLBACK_COLUMNS):
        _issue(
            issues,
            "error",
            "missing_stream_identifier",
            "Stimulus-detection observations must contain 'stream_id', 'sequence_id', or 'sample_index'.",
        )


def _validate_required_columns(frame: pd.DataFrame, issues: list[ObservationValidationIssue], columns: Sequence[str]) -> None:
    for column in columns:
        if column not in frame.columns:
            _issue(
                issues,
                "error",
                "missing_canonical_column",
                f"Canonical probability observations must contain column '{column}'.",
                column=column,
            )
            continue
        missing = ~_present_mask(frame, column)
        for row_index, value in frame.loc[missing, column].head(20).items():
            _issue(
                issues,
                "error",
                "empty_canonical_column",
                f"Canonical probability observations must not contain empty values in column '{column}'.",
                column=column,
                row=int(row_index),
                value=value,
            )


def _validate_recommended_columns(frame: pd.DataFrame, issues: list[ObservationValidationIssue], columns: Sequence[str]) -> None:
    for column in columns:
        if column not in frame.columns:
            _issue(
                issues,
                "warning",
                f"missing_recommended_{column}",
                f"Column '{column}' is recommended for canonical probability observations.",
                column=column,
            )


def _numeric_label_to_probability_columns(prob_columns: Sequence[str]) -> dict[int, str]:
    label_columns: dict[int, str] = {}
    for column in prob_columns:
        suffix = column.removeprefix("prob_class_")
        if suffix.isdigit():
            label_columns[int(suffix)] = column
    return label_columns


def _validate_probability_consistency(
    frame: pd.DataFrame,
    probabilities: pd.DataFrame,
    prob_columns: Sequence[str],
    issues: list[ObservationValidationIssue],
    *,
    tolerance: float,
) -> None:
    if probabilities.empty:
        return
    probability_values = probabilities.to_numpy(dtype=float)
    finite_row = np.isfinite(probability_values).any(axis=1)
    filled = np.where(np.isfinite(probability_values), probability_values, -np.inf)
    max_probabilities = np.nanmax(probability_values, axis=1)
    argmax_positions = filled.argmax(axis=1)
    label_columns = _numeric_label_to_probability_columns(prob_columns)
    ordered_labels = [int(column.removeprefix("prob_class_")) if column.removeprefix("prob_class_").isdigit() else None for column in prob_columns]

    if "confidence" in frame.columns:
        confidence = _numeric_series(frame, "confidence", issues, allow_nan=True)
        confidence_values = confidence.to_numpy(dtype=float)
        bad_confidence = confidence.notna() & finite_row & (np.abs(confidence_values - max_probabilities) > tolerance)
        for row_index, value in confidence.loc[bad_confidence].head(20).items():
            _issue(
                issues,
                "error",
                "confidence_probability_mismatch",
                "Column 'confidence' must equal the maximum prob_class_* value within tolerance.",
                column="confidence",
                row=int(row_index),
                value=float(value),
            )

    if "predicted_label" in frame.columns and all(label is not None for label in ordered_labels):
        predicted_label = _numeric_series(frame, "predicted_label", issues, allow_nan=True)
        expected = pd.Series([ordered_labels[position] for position in argmax_positions], index=frame.index, dtype=float)
        bad_prediction = predicted_label.notna() & finite_row & (predicted_label.astype(float) != expected)
        for row_index, value in predicted_label.loc[bad_prediction].head(20).items():
            _issue(
                issues,
                "error",
                "predicted_label_probability_mismatch",
                "Column 'predicted_label' must equal the argmax prob_class_* label.",
                column="predicted_label",
                row=int(row_index),
                value=int(value),
            )

    if "probability_true_class" in frame.columns and "true_label" in frame.columns and label_columns:
        true_label = _numeric_series(frame, "true_label", issues, allow_nan=True)
        probability_true_class = _numeric_series(frame, "probability_true_class", issues, allow_nan=True)
        for row_index, label_value in true_label.dropna().items():
            label = int(label_value)
            column = label_columns.get(label)
            if column is None or pd.isna(probability_true_class.loc[row_index]):
                continue
            expected_probability = float(probabilities.loc[row_index, column])
            observed_probability = float(probability_true_class.loc[row_index])
            if abs(observed_probability - expected_probability) > tolerance:
                _issue(
                    issues,
                    "error",
                    "true_probability_mismatch",
                    "Column 'probability_true_class' must match prob_class_<true_label> within tolerance.",
                    column="probability_true_class",
                    row=int(row_index),
                    value=observed_probability,
                )

    if "predicted_class" in frame.columns and "predicted_label" in frame.columns:
        predicted_label = pd.to_numeric(frame["predicted_label"], errors="coerce")
        for row_index, label_value in predicted_label.dropna().items():
            class_column = f"class_{int(label_value)}"
            if class_column not in frame.columns or pd.isna(frame.loc[row_index, "predicted_class"]):
                continue
            expected_class = str(frame.loc[row_index, class_column])
            observed_class = str(frame.loc[row_index, "predicted_class"])
            if observed_class != expected_class:
                _issue(
                    issues,
                    "error",
                    "predicted_class_mismatch",
                    "Column 'predicted_class' must match class_<predicted_label>.",
                    column="predicted_class",
                    row=int(row_index),
                    value=observed_class,
                )


def _validate_canonical_profile(
    frame: pd.DataFrame,
    probabilities: pd.DataFrame,
    prob_columns: Sequence[str],
    issues: list[ObservationValidationIssue],
    *,
    probability_tolerance: float,
) -> None:
    _validate_required_columns(frame, issues, CANONICAL_REQUIRED_COLUMNS)
    _validate_recommended_columns(frame, issues, CANONICAL_RECOMMENDED_COLUMNS)
    for column in CANONICAL_NUMERIC_COLUMNS:
        if column in frame.columns:
            numeric = _numeric_series(frame, column, issues)
            non_finite = numeric.notna() & ~np.isfinite(numeric.to_numpy(dtype=float))
            for row_index, value in frame.loc[non_finite, column].head(20).items():
                _issue(
                    issues,
                    "error",
                    f"non_finite_{column}",
                    f"Column '{column}' must contain finite numeric values.",
                    column=column,
                    row=int(row_index),
                    value=value,
                )
    if "time" in frame.columns and "test_time" in frame.columns:
        time_values = pd.to_numeric(frame["time"], errors="coerce")
        test_time_values = pd.to_numeric(frame["test_time"], errors="coerce")
        mismatch = time_values.notna() & test_time_values.notna() & ((time_values - test_time_values).abs() > probability_tolerance)
        for row_index, value in test_time_values.loc[mismatch].head(20).items():
            _issue(
                issues,
                "error",
                "time_test_time_mismatch",
                "Column 'time' must match 'test_time' within tolerance for canonical observations.",
                column="test_time",
                row=int(row_index),
                value=float(value),
            )
    _validate_probability_consistency(frame, probabilities, prob_columns, issues, tolerance=probability_tolerance)
    if "stream_id" in frame.columns and _present_mask(frame, "stream_id").any():
        stream_rows = _present_mask(frame, "stream_id")
        for column in ("sample_index", "sequence_id"):
            if column not in frame.columns:
                _issue(
                    issues,
                    "error",
                    "missing_stream_identity_column",
                    f"Canonical continuous observations with 'stream_id' must contain '{column}'.",
                    column=column,
                )
                continue
            missing = stream_rows & ~_present_mask(frame, column)
            for row_index, value in frame.loc[missing, column].head(20).items():
                _issue(
                    issues,
                    "error",
                    "empty_stream_identity_column",
                    f"Canonical continuous observations with 'stream_id' must not contain empty values in '{column}'.",
                    column=column,
                    row=int(row_index),
                    value=value,
                )
        if {"stream_id", "sample_index"}.issubset(frame.columns):
            duplicate_count = int(frame.loc[stream_rows].duplicated(["stream_id", "sample_index"], keep=False).sum())
            if duplicate_count:
                _issue(issues, "error", "duplicate_stream_sample", f"{duplicate_count} row(s) share the same stream_id and sample_index.")


def validate_probability_observations(
    frame: pd.DataFrame,
    *,
    profile: ObservationProfile = "generic",
    probability_tolerance: float = DEFAULT_PROBABILITY_TOLERANCE,
    require_normalized: bool = False,
    group_columns: Sequence[str] | None = None,
    stream_columns: Sequence[str] | None = None,
) -> ObservationValidationReport:
    """Validate a RepTrace probability-observation table.

    Structural problems that prevent downstream interpretation are reported as
    errors. Reproducibility or quality concerns that current readers can often
    fill or tolerate are reported as warnings.
    """
    if profile not in PROFILES:
        raise ValueError(f"profile must be one of {PROFILES}.")
    if probability_tolerance < 0:
        raise ValueError("probability_tolerance must be non-negative.")

    issues: list[ObservationValidationIssue] = []
    _validate_time(frame, issues)
    _validate_group_columns(frame, group_columns, issues)
    prob_columns = _try_probability_columns(frame, issues)
    probabilities = _probability_frame(frame, prob_columns, issues)
    _validate_probabilities(
        probabilities,
        issues,
        tolerance=probability_tolerance,
        require_normalized=require_normalized or profile == "canonical",
    )
    _validate_metadata(frame, issues)
    _validate_class_columns(frame, prob_columns, issues)

    if profile == "canonical":
        _validate_canonical_profile(frame, probabilities, prob_columns, issues, probability_tolerance=probability_tolerance)
    elif profile == "temporal-model":
        _validate_temporal_profile(frame, issues)
    elif profile == "stimulus-detection":
        _validate_stimulus_profile(frame, stream_columns, issues)

    return ObservationValidationReport(
        profile=profile,
        n_rows=int(len(frame)),
        probability_columns=tuple(prob_columns),
        issues=tuple(issues),
    )


def _normalize_probability_rows(frame: pd.DataFrame, prob_columns: Sequence[str]) -> pd.DataFrame:
    normalized = frame.copy()
    probabilities = normalized.loc[:, list(prob_columns)].apply(pd.to_numeric, errors="coerce")
    row_sums = probabilities.fillna(0.0).sum(axis=1)
    valid = row_sums > 0
    normalized.loc[valid, list(prob_columns)] = probabilities.loc[valid].div(row_sums.loc[valid], axis=0)
    return normalized


def _add_reader_defaults(frame: pd.DataFrame, csv_path: Path, *, profile: ObservationProfile) -> pd.DataFrame:
    result = frame.copy()
    if profile != "canonical":
        if profile == "temporal-model" and "sequence_id" not in result.columns and "sample_index" in result.columns:
            result["sequence_id"] = result["sample_index"]
        if profile == "stimulus-detection" and "stream_id" not in result.columns and "sequence_id" not in result.columns:
            if "sample_index" in result.columns:
                result["sequence_id"] = result["sample_index"]
            else:
                result["stream_id"] = csv_path.stem
        if "subject" not in result.columns:
            result["subject"] = csv_path.stem
        if "decoder" not in result.columns:
            result["decoder"] = "decoder"
        if "emission_mode" not in result.columns:
            result["emission_mode"] = "calibrated"
    if "source_file" not in result.columns:
        result["source_file"] = csv_path.name
    return result


def read_validated_probability_observations(
    csv_paths: Sequence[str | Path],
    *,
    profile: ObservationProfile = "generic",
    probability_tolerance: float = DEFAULT_PROBABILITY_TOLERANCE,
    require_normalized: bool = False,
    normalize: bool = False,
    add_defaults: bool = True,
) -> pd.DataFrame:
    """Read one or more probability-observation CSV files and validate them."""
    paths = _expand_paths(csv_paths)
    if not paths:
        raise ValueError("At least one observation CSV path is required.")

    frames = []
    for csv_path in paths:
        frame = pd.read_csv(csv_path)
        if add_defaults:
            frame = _add_reader_defaults(frame, csv_path, profile=profile)
        prob_columns = probability_columns(frame)
        if normalize:
            frame = _normalize_probability_rows(frame, prob_columns)
        report = validate_probability_observations(
            frame,
            profile=profile,
            probability_tolerance=probability_tolerance,
            require_normalized=require_normalized,
        )
        report.raise_for_errors()
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def summarize_probability_observations(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a compact one-row summary for a probability-observation table."""
    prob_columns = probability_columns(frame)
    probabilities = frame.loc[:, prob_columns].apply(pd.to_numeric, errors="coerce")
    row_sums = probabilities.fillna(0.0).sum(axis=1)
    summary: dict[str, object] = {
        "n_rows": int(len(frame)),
        "n_probability_columns": int(len(prob_columns)),
        "probability_columns": ",".join(prob_columns),
        "max_probability_sum_abs_deviation": float((row_sums - 1.0).abs().max()) if len(row_sums) else np.nan,
    }
    if "time" in frame.columns:
        times = pd.to_numeric(frame["time"], errors="coerce")
        summary["time_min"] = float(times.min()) if times.notna().any() else np.nan
        summary["time_max"] = float(times.max()) if times.notna().any() else np.nan
        summary["n_time_points"] = int(times.nunique(dropna=True))
    for column in ("subject", "sequence_id", "stream_id", "decoder", "emission_mode"):
        if column in frame.columns:
            summary[f"n_{column}s"] = int(frame[column].nunique(dropna=True))
    return pd.DataFrame([summary])


def _read_cli_frame(paths: Sequence[str | Path], *, profile: ObservationProfile, add_defaults: bool) -> pd.DataFrame:
    frames = []
    for csv_path in _expand_paths(paths):
        frame = pd.read_csv(csv_path)
        if add_defaults:
            frame = _add_reader_defaults(frame, csv_path, profile=profile)
        frames.append(frame)
    if not frames:
        raise ValueError("At least one observation CSV path is required.")
    return pd.concat(frames, ignore_index=True)


def _write_frame(path: Path | None, frame: pd.DataFrame) -> None:
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point for probability-observation schema validation."""
    parser = argparse.ArgumentParser(description="Validate RepTrace probability-observation CSV files.")
    parser.add_argument("observation_csv", nargs="+", help="Observation CSVs or glob patterns with time and prob_class_* columns.")
    parser.add_argument("--profile", choices=PROFILES, default="generic", help="Validation profile for a downstream RepTrace workflow.")
    parser.add_argument("--probability-tolerance", type=float, default=DEFAULT_PROBABILITY_TOLERANCE)
    parser.add_argument("--require-normalized", action="store_true", help="Treat probability row-sum deviations as errors instead of warnings.")
    parser.add_argument("--group-column", action="append", dest="group_columns", help="Column required for downstream grouping. May be repeated.")
    parser.add_argument("--stream-column", action="append", dest="stream_columns", help="Stream identifier column required by stimulus detection. May be repeated.")
    parser.add_argument("--no-defaults", action="store_true", help="Do not fill reader-compatible default subject/decoder/emission columns before validation.")
    parser.add_argument("--normalize-out", type=Path, help="Optional CSV path for row-normalized probability observations.")
    parser.add_argument("--report-out", type=Path, help="Optional CSV path for validation issues.")
    parser.add_argument("--summary-out", type=Path, help="Optional CSV path for a compact table summary.")
    parser.add_argument("--warn-only", action="store_true", help="Return success even when validation errors are present.")
    args = parser.parse_args(argv)

    try:
        frame = _read_cli_frame(args.observation_csv, profile=args.profile, add_defaults=not args.no_defaults)
        prob_columns = _try_probability_columns(frame, [])
        if args.normalize_out is not None and prob_columns:
            normalized = _normalize_probability_rows(frame, prob_columns)
            _write_frame(args.normalize_out, normalized)

        report = validate_probability_observations(
            frame,
            profile=args.profile,
            probability_tolerance=args.probability_tolerance,
            require_normalized=args.require_normalized,
            group_columns=args.group_columns,
            stream_columns=args.stream_columns,
        )
        _write_frame(args.report_out, report.to_frame())
        if args.summary_out is not None:
            _write_frame(args.summary_out, summarize_probability_observations(frame))
    except Exception as exc:
        print(f"Observation validation failed: {exc}", file=sys.stderr)
        return 2

    print(
        f"Validated {report.n_rows} row(s), {len(report.probability_columns)} probability column(s): "
        f"{len(report.errors)} error(s), {len(report.warnings)} warning(s)."
    )
    if args.report_out is None and report.issues:
        print(report.to_frame().to_string(index=False))

    if report.errors and not args.warn_only:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
