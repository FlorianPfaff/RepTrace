from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mne
import pandas as pd

from reptrace.metadata import add_binary_label


@dataclass(frozen=True)
class ManifestValidation:
    """Validation result for one manifest row."""

    subject: str
    ok: bool
    messages: list[str]


def _missing(value: Any) -> bool:
    return value is None or pd.isna(value) or str(value).strip() == ""


def _value(row: pd.Series, column: str, default: str | None = None) -> str | None:
    if column not in row or _missing(row[column]):
        return default
    return str(row[column])


def _int_value(row: pd.Series, column: str, default: int) -> int:
    value = _value(row, column)
    return default if value is None else int(float(value))


def _resolve(value: str | None, base_dir: Path) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = base_dir / path
    return path


def _load_metadata_for_row(row: pd.Series, base_dir: Path) -> tuple[pd.DataFrame | None, list[str]]:
    messages: list[str] = []
    metadata_csv = _resolve(_value(row, "metadata_csv"), base_dir)
    events_csv = _resolve(_value(row, "events_csv"), base_dir)

    if events_csv is not None:
        if not events_csv.exists():
            return None, [f"events_csv does not exist: {events_csv}"]
        source_column = _value(row, "source_column")
        positive_pattern = _value(row, "positive_pattern")
        label_column = _value(row, "label_column", "condition") or "condition"
        if source_column is None:
            messages.append("events_csv is set but source_column is missing")
        if positive_pattern is None:
            messages.append("events_csv is set but positive_pattern is missing")
        if messages:
            return None, messages
        events = pd.read_csv(events_csv)
        try:
            metadata = add_binary_label(
                events,
                source_column=source_column,
                positive_pattern=positive_pattern,
                negative_pattern=_value(row, "negative_pattern"),
                label_column=label_column,
                positive_label=_value(row, "positive_label", "positive") or "positive",
                negative_label=_value(row, "negative_label", "negative") or "negative",
                case_sensitive=(_value(row, "case_sensitive") or "").lower() in {"1", "true", "yes", "y"},
            )
        except ValueError as exc:
            return None, [str(exc)]
        return metadata, []

    if metadata_csv is not None:
        if not metadata_csv.exists():
            return None, [f"metadata_csv does not exist: {metadata_csv}"]
        return pd.read_csv(metadata_csv), []

    return None, []


def _validate_class_balance(metadata: pd.DataFrame, label_column: str, n_splits: int) -> list[str]:
    messages: list[str] = []
    if label_column not in metadata.columns:
        return [f"label column '{label_column}' is missing"]

    labels = metadata[label_column].dropna()
    class_counts = labels.value_counts()
    if len(class_counts) < 2:
        messages.append(f"label column '{label_column}' must contain at least two classes")
    if not class_counts.empty and int(class_counts.min()) < n_splits:
        messages.append(
            f"smallest class has {int(class_counts.min())} trial(s), fewer than n_splits={n_splits}"
        )
    return messages


def validate_manifest(
    manifest_csv: Path,
    *,
    default_label_column: str | None = None,
    default_group_column: str | None = None,
    default_n_splits: int = 5,
) -> list[ManifestValidation]:
    """Validate staged files and metadata referenced by a benchmark manifest."""
    manifest = pd.read_csv(manifest_csv)
    required = {"subject", "epochs"}
    missing_columns = sorted(required.difference(manifest.columns))
    if missing_columns:
        raise ValueError(f"Manifest is missing required columns: {missing_columns}")

    base_dir = manifest_csv.parent
    validations: list[ManifestValidation] = []
    for _, row in manifest.iterrows():
        subject = _value(row, "subject", "<missing>") or "<missing>"
        messages: list[str] = []
        epochs_path = _resolve(_value(row, "epochs"), base_dir)
        label_column = _value(row, "label_column", default_label_column)
        group_column = _value(row, "group_column", default_group_column)
        n_splits = _int_value(row, "n_splits", default_n_splits)

        if label_column is None:
            messages.append("label_column is missing")
        if epochs_path is None:
            messages.append("epochs path is missing")
        elif not epochs_path.exists():
            messages.append(f"epochs file does not exist: {epochs_path}")

        metadata, metadata_messages = _load_metadata_for_row(row, base_dir)
        messages.extend(metadata_messages)

        epochs_metadata: pd.DataFrame | None = None
        if epochs_path is not None and epochs_path.exists():
            try:
                epochs = mne.read_epochs(epochs_path, preload=False, verbose="error")
                epochs_metadata = epochs.metadata.copy() if epochs.metadata is not None else None
                n_epochs = len(epochs)
            except Exception as exc:  # pragma: no cover - MNE raises several concrete IO errors.
                messages.append(f"could not read epochs file: {exc}")
                n_epochs = None
            else:
                if metadata is not None and len(metadata) != n_epochs:
                    messages.append(
                        f"metadata rows ({len(metadata)}) do not match epochs ({n_epochs})"
                    )
        else:
            n_epochs = None

        effective_metadata = metadata if metadata is not None else epochs_metadata
        if effective_metadata is None:
            if not any("metadata_csv" in message or "events_csv" in message for message in messages):
                messages.append("no metadata source available; provide metadata_csv, events_csv, or epochs metadata")
        elif label_column is not None:
            messages.extend(_validate_class_balance(effective_metadata, label_column, n_splits))
            if group_column is not None:
                if group_column not in effective_metadata.columns:
                    messages.append(f"group column '{group_column}' is missing")
                elif effective_metadata[group_column].dropna().nunique() < n_splits:
                    messages.append(
                        f"group column '{group_column}' has fewer than n_splits={n_splits} unique groups"
                    )

        validations.append(
            ManifestValidation(subject=subject, ok=not messages, messages=messages)
        )
    return validations


def validation_report_frame(validations: list[ManifestValidation]) -> pd.DataFrame:
    """Return a tabular validation report."""
    rows = []
    for validation in validations:
        rows.append(
            {
                "subject": validation.subject,
                "ok": validation.ok,
                "messages": " | ".join(validation.messages),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate files and metadata referenced by a RepTrace benchmark manifest."
    )
    parser.add_argument("manifest_csv", type=Path)
    parser.add_argument("--label-column")
    parser.add_argument("--group-column")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--report-out", type=Path)
    args = parser.parse_args()

    validations = validate_manifest(
        args.manifest_csv,
        default_label_column=args.label_column,
        default_group_column=args.group_column,
        default_n_splits=args.n_splits,
    )
    report = validation_report_frame(validations)
    if args.report_out is not None:
        args.report_out.parent.mkdir(parents=True, exist_ok=True)
        report.to_csv(args.report_out, index=False)
        print(f"Wrote {args.report_out}")
    for row in report.itertuples(index=False):
        status = "ok" if row.ok else "error"
        detail = "" if row.ok else f": {row.messages}"
        print(f"{status}\t{row.subject}{detail}")
    if not all(validation.ok for validation in validations):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
