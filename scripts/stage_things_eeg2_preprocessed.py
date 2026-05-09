"""Stage THINGS-EEG2 author-preprocessed arrays for RepTrace benchmarks."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import mne
import numpy as np
import pandas as pd

DEFAULT_DECODERS = ("logistic", "lda", "linear_svm")
PARTITION_ALIASES = {"train": "training", "training": "training", "test": "test"}


@dataclass(frozen=True)
class SubjectStageResult:
    subject: str
    epochs_path: Path
    metadata_path: Path
    n_trials: int
    n_conditions: int
    labels: tuple[str, ...]


def _normalise_partition(value: str) -> str:
    partition = value.strip().lower()
    if partition not in PARTITION_ALIASES:
        raise ValueError(f"Unknown THINGS-EEG2 partition '{value}'. Expected one of: {sorted(PARTITION_ALIASES)}")
    return PARTITION_ALIASES[partition]


def _normalise_subject(subject: str | int) -> str:
    text = str(subject).strip()
    if text.startswith("sub-"):
        return text
    return f"sub-{int(text):02d}"


def _normalise_key(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    if not text:
        return ""
    try:
        number = float(text)
    except ValueError:
        return text
    return str(int(number)) if number.is_integer() else text


def _partition_filename(partition: str) -> str:
    return f"preprocessed_eeg_{_normalise_partition(partition)}.npy"


def _candidate_subject_dirs(things_root: Path, subject: str) -> list[Path]:
    return [
        things_root / "eeg_dataset" / "preprocessed_data" / subject,
        things_root / "eeg_dataset" / "Preprocessed_data" / subject,
        things_root / "preprocessed_data" / subject,
        things_root / "Preprocessed_data" / subject,
        things_root / "derivatives" / "preprocessed_eeg" / subject,
        things_root / subject,
    ]


def find_preprocessed_file(things_root: Path, subject: str, partition: str) -> Path:
    file_name = _partition_filename(partition)
    for subject_dir in _candidate_subject_dirs(things_root, subject):
        candidate = subject_dir / file_name
        if candidate.is_file():
            return candidate
    searched = "\n  ".join(str(path / file_name) for path in _candidate_subject_dirs(things_root, subject))
    raise FileNotFoundError(f"Could not find {file_name} for {subject}. Searched:\n  {searched}")


def load_label_map(
    label_map_csv: Path | None,
    *,
    key_column: str,
    label_column: str,
    partition_column: str | None = None,
) -> dict[tuple[str | None, str], str]:
    if label_map_csv is None:
        return {}
    if not label_map_csv.is_file():
        raise FileNotFoundError(f"Label map does not exist: {label_map_csv}")
    frame = pd.read_csv(label_map_csv)
    missing = [column for column in (key_column, label_column) if column not in frame.columns]
    if partition_column is not None and partition_column not in frame.columns:
        missing.append(partition_column)
    if missing:
        raise ValueError(f"Label map {label_map_csv} is missing required column(s): {missing}")

    mapping: dict[tuple[str | None, str], str] = {}
    for row in frame.itertuples(index=False):
        row_dict = row._asdict()
        key = _normalise_key(row_dict[key_column])
        label = str(row_dict[label_column]).strip()
        if not key or not label or label.lower() == "nan":
            continue
        partition_key = _normalise_partition(str(row_dict[partition_column])) if partition_column else None
        mapping[(partition_key, key)] = label
    if not mapping:
        raise ValueError(f"Label map {label_map_csv} did not contain any usable rows.")
    return mapping


def _label_for_condition(condition_id: int, *, partition: str, label_map: dict[tuple[str | None, str], str]) -> str:
    key = _normalise_key(condition_id)
    return label_map.get((_normalise_partition(partition), key), label_map.get((None, key), f"condition_{condition_id:05d}"))


def _condition_rows(*, subject: str, partition: str, condition_id: int, label: str, label_column: str, n_repetitions: int) -> list[dict[str, object]]:
    return [
        {
            "subject": subject,
            "partition": partition,
            "image_condition": condition_id,
            "repetition": repetition,
            label_column: label,
        }
        for repetition in range(n_repetitions)
    ]


def _as_epochs_array(data: np.ndarray, *, ch_names: Iterable[str], times: np.ndarray, metadata: pd.DataFrame) -> mne.EpochsArray:
    times = np.asarray(times, dtype=float)
    if times.ndim != 1 or len(times) < 2:
        raise ValueError("THINGS-EEG2 time vector must be one-dimensional with at least two samples.")
    sfreq = float(1.0 / np.median(np.diff(times)))
    info = mne.create_info(list(ch_names), sfreq=sfreq, ch_types="eeg")
    try:
        info.set_montage("standard_1020", on_missing="ignore")
    except Exception:
        pass
    events = np.column_stack([np.arange(len(metadata), dtype=int), np.zeros(len(metadata), dtype=int), np.ones(len(metadata), dtype=int)])
    return mne.EpochsArray(data, info, events=events, tmin=float(times[0]), event_id={"image": 1}, metadata=metadata, verbose="error")


def stage_subject(
    *,
    things_root: Path,
    staged_dir: Path,
    subject: str,
    partition: str,
    label_map: dict[tuple[str | None, str], str],
    label_column: str,
    target_labels: set[str] | None = None,
    max_conditions_per_label: int | None = None,
    max_repetitions: int | None = None,
    overwrite: bool = False,
) -> SubjectStageResult:
    partition = _normalise_partition(partition)
    subject = _normalise_subject(subject)
    source_path = find_preprocessed_file(things_root, subject, partition)
    payload = np.load(source_path, allow_pickle=True).item()
    if "preprocessed_eeg_data" not in payload:
        raise ValueError(f"{source_path} does not contain 'preprocessed_eeg_data'.")
    source_data = np.asarray(payload["preprocessed_eeg_data"])
    if source_data.ndim != 4:
        raise ValueError(f"Expected condition x repetition x channel x time data in {source_path}; got {source_data.shape}.")
    ch_names = payload.get("ch_names")
    times = payload.get("times")
    if ch_names is None or times is None:
        raise ValueError(f"{source_path} must contain 'ch_names' and 'times'.")

    staged_dir.mkdir(parents=True, exist_ok=True)
    epochs_path = staged_dir / f"{subject}_things_eeg2_{partition}_epo.fif"
    metadata_path = staged_dir / f"{subject}_things_eeg2_{partition}_metadata.csv"
    if not overwrite and epochs_path.is_file() and metadata_path.is_file():
        metadata = pd.read_csv(metadata_path)
        labels = tuple(sorted(str(label) for label in metadata[label_column].dropna().unique()))
        return SubjectStageResult(subject, epochs_path, metadata_path, len(metadata), metadata["image_condition"].nunique(), labels)

    selected_blocks: list[np.ndarray] = []
    metadata_rows: list[dict[str, object]] = []
    label_condition_counts: defaultdict[str, int] = defaultdict(int)
    n_repetitions = source_data.shape[1] if max_repetitions is None else min(max_repetitions, source_data.shape[1])
    if n_repetitions <= 0:
        raise ValueError("max_repetitions must leave at least one repetition per condition.")

    for condition_index in range(source_data.shape[0]):
        condition_id = condition_index + 1
        label = _label_for_condition(condition_id, partition=partition, label_map=label_map)
        if target_labels is not None and label not in target_labels:
            continue
        if max_conditions_per_label is not None and label_condition_counts[label] >= max_conditions_per_label:
            continue
        label_condition_counts[label] += 1
        selected_blocks.append(source_data[condition_index, :n_repetitions])
        metadata_rows.extend(_condition_rows(subject=subject, partition=partition, condition_id=condition_id, label=label, label_column=label_column, n_repetitions=n_repetitions))

    if not selected_blocks:
        raise ValueError(f"No THINGS-EEG2 conditions remained for {subject} after label/target filtering.")
    labels = tuple(sorted(label_condition_counts))
    if len(labels) < 2:
        raise ValueError(f"Need at least two labels after filtering; got {labels}.")
    trial_data = np.concatenate(selected_blocks, axis=0).reshape((-1, source_data.shape[-2], source_data.shape[-1]))
    metadata = pd.DataFrame(metadata_rows)
    if len(metadata) != len(trial_data):
        raise ValueError(f"Metadata/trial mismatch for {subject}: {len(metadata)} rows vs {len(trial_data)} trials.")
    epochs = _as_epochs_array(trial_data, ch_names=ch_names, times=np.asarray(times), metadata=metadata)
    epochs.save(epochs_path, overwrite=True)
    metadata.to_csv(metadata_path, index=False)
    return SubjectStageResult(subject, epochs_path, metadata_path, len(metadata), len(selected_blocks), labels)


def write_manifest(
    results: list[SubjectStageResult],
    manifest_out: Path,
    *,
    decoders: Iterable[str],
    label_column: str,
    group_column: str | None,
    tmin: float,
    tmax: float,
    window_ms: float,
    step_ms: float,
    n_splits: int,
) -> None:
    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["subject", "decoder", "epochs", "metadata_csv", "label_column", "group_column", "tmin", "tmax", "window_ms", "step_ms", "n_splits"]
    with manifest_out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            for decoder in decoders:
                writer.writerow(
                    {
                        "subject": result.subject,
                        "decoder": decoder,
                        "epochs": str(result.epochs_path),
                        "metadata_csv": str(result.metadata_path),
                        "label_column": label_column,
                        "group_column": group_column or "",
                        "tmin": tmin,
                        "tmax": tmax,
                        "window_ms": window_ms,
                        "step_ms": step_ms,
                        "n_splits": n_splits,
                    }
                )


def _split_csv_or_space(value: str) -> list[str]:
    return [part.strip() for chunk in value.split(",") for part in chunk.split() if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--things-root", type=Path, required=True)
    parser.add_argument("--staged-dir", type=Path, required=True)
    parser.add_argument("--manifest-out", type=Path, required=True)
    parser.add_argument("--subjects", nargs="+", default=[str(i) for i in range(1, 11)])
    parser.add_argument("--partition", default="test", choices=["test", "training"])
    parser.add_argument("--label-map-csv", type=Path)
    parser.add_argument("--label-map-key-column", default="image_condition")
    parser.add_argument("--label-map-label-column", default="label")
    parser.add_argument("--label-map-partition-column")
    parser.add_argument("--label-column", default="condition")
    parser.add_argument("--group-column", default="image_condition")
    parser.add_argument("--target-labels", nargs="*")
    parser.add_argument("--max-conditions-per-label", type=int)
    parser.add_argument("--max-repetitions", type=int)
    parser.add_argument("--decoders", nargs="+", default=list(DEFAULT_DECODERS))
    parser.add_argument("--tmin", type=float, default=-0.1)
    parser.add_argument("--tmax", type=float, default=0.6)
    parser.add_argument("--window-ms", type=float, default=20.0)
    parser.add_argument("--step-ms", type=float, default=10.0)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    label_map = load_label_map(args.label_map_csv, key_column=args.label_map_key_column, label_column=args.label_map_label_column, partition_column=args.label_map_partition_column)
    target_labels = None if not args.target_labels else {label for value in args.target_labels for label in _split_csv_or_space(value)}
    staged_results = []
    for subject in args.subjects:
        result = stage_subject(
            things_root=args.things_root,
            staged_dir=args.staged_dir,
            subject=subject,
            partition=args.partition,
            label_map=label_map,
            label_column=args.label_column,
            target_labels=target_labels,
            max_conditions_per_label=args.max_conditions_per_label,
            max_repetitions=args.max_repetitions,
            overwrite=args.overwrite,
        )
        staged_results.append(result)
        print(f"Staged {result.subject}: {result.n_trials} trials, {result.n_conditions} image conditions, labels={','.join(result.labels)}")

    write_manifest(
        staged_results,
        args.manifest_out,
        decoders=args.decoders,
        label_column=args.label_column,
        group_column=args.group_column,
        tmin=args.tmin,
        tmax=args.tmax,
        window_ms=args.window_ms,
        step_ms=args.step_ms,
        n_splits=args.n_splits,
    )
    print(f"Wrote THINGS-EEG2 RepTrace manifest: {args.manifest_out}")


if __name__ == "__main__":
    main()
