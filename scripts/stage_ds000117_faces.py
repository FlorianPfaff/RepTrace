"""Stage OpenNeuro ds000117 face-recognition MEG data for RepTrace."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import mne
import numpy as np
import pandas as pd

DEFAULT_DECODERS = ("logistic", "lda", "linear_svm")
DEFAULT_FACE_CONDITIONS = ("Famous", "Unfamiliar")
DEFAULT_SCRAMBLED_CONDITIONS = ("Scrambled",)


@dataclass(frozen=True)
class SubjectStageResult:
    subject: str
    epochs_path: Path
    metadata_path: Path
    n_trials: int
    labels: tuple[str, ...]
    runs: tuple[str, ...]


def _normalise_subject(subject: str | int) -> str:
    text = str(subject).strip()
    if text.startswith("sub-"):
        return text
    return f"sub-{int(text):02d}"


def _normalise_run(run: str | int) -> str:
    text = str(run).strip()
    return text if text.startswith("run-") else f"{int(text):02d}"


def _split_csv_or_space(value: str) -> list[str]:
    return [part.strip() for chunk in value.split(",") for part in chunk.split() if part.strip()]


def _condition_key(value: object) -> str:
    return str(value).strip().casefold()


def _condition_label(
    condition: object,
    *,
    face_conditions: Iterable[str],
    scrambled_conditions: Iterable[str],
) -> str | None:
    key = _condition_key(condition)
    face_keys = {_condition_key(value) for value in face_conditions}
    scrambled_keys = {_condition_key(value) for value in scrambled_conditions}
    if key in face_keys:
        return "face"
    if key in scrambled_keys:
        return "scrambled"
    return None


def _run_paths(
    bids_root: Path,
    *,
    subject: str,
    session: str,
    task: str,
    run: str,
) -> tuple[Path, Path]:
    meg_dir = bids_root / subject / f"ses-{session}" / "meg"
    stem = f"{subject}_ses-{session}_task-{task}_run-{run}"
    return meg_dir / f"{stem}_meg.fif", meg_dir / f"{stem}_events.tsv"


def _metadata_for_run(
    events_path: Path,
    *,
    subject: str,
    run: str,
    condition_column: str,
    face_conditions: Iterable[str],
    scrambled_conditions: Iterable[str],
    max_events_per_label: int | None = None,
) -> pd.DataFrame:
    frame = pd.read_csv(events_path, sep="\t")
    required = {"onset", condition_column}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{events_path} is missing required column(s): {missing}")

    rows = []
    label_counts: dict[str, int] = {}
    for row in frame.to_dict(orient="records"):
        label = _condition_label(
            row[condition_column],
            face_conditions=face_conditions,
            scrambled_conditions=scrambled_conditions,
        )
        if label is None:
            continue
        count = label_counts.get(label, 0)
        if max_events_per_label is not None and count >= max_events_per_label:
            continue
        label_counts[label] = count + 1
        rows.append(
            {
                **row,
                "subject": subject,
                "run": run,
                "original_condition": str(row[condition_column]),
                "condition": label,
            }
        )

    metadata = pd.DataFrame(rows)
    if metadata.empty:
        raise ValueError(f"No face/scrambled events found in {events_path}.")
    labels = set(metadata["condition"].dropna())
    if labels != {"face", "scrambled"}:
        raise ValueError(f"Expected both face and scrambled events in {events_path}; got {sorted(labels)}.")
    return metadata.reset_index(drop=True)


def _events_from_metadata(raw: mne.io.BaseRaw, metadata: pd.DataFrame) -> np.ndarray:
    onsets = pd.to_numeric(metadata["onset"], errors="raise").to_numpy(dtype=float)
    samples = raw.time_as_index(onsets, use_rounding=True) + raw.first_samp
    event_codes = metadata["condition"].map({"face": 1, "scrambled": 2}).to_numpy(dtype=int)
    return np.column_stack([samples, np.zeros(len(samples), dtype=int), event_codes])


def _stage_run(
    raw_path: Path,
    events_path: Path,
    *,
    subject: str,
    run: str,
    tmin: float,
    tmax: float,
    baseline: tuple[float | None, float | None] | None,
    picks: str,
    condition_column: str,
    face_conditions: Iterable[str],
    scrambled_conditions: Iterable[str],
    max_events_per_label: int | None,
) -> mne.Epochs:
    if not raw_path.is_file():
        raise FileNotFoundError(f"Missing raw FIF file: {raw_path}")
    if not events_path.is_file():
        raise FileNotFoundError(f"Missing events TSV file: {events_path}")

    raw = mne.io.read_raw_fif(raw_path, preload=False, verbose="error")
    metadata = _metadata_for_run(
        events_path,
        subject=subject,
        run=run,
        condition_column=condition_column,
        face_conditions=face_conditions,
        scrambled_conditions=scrambled_conditions,
        max_events_per_label=max_events_per_label,
    )
    epochs = mne.Epochs(
        raw,
        _events_from_metadata(raw, metadata),
        event_id={"face": 1, "scrambled": 2},
        tmin=tmin,
        tmax=tmax,
        baseline=baseline,
        metadata=metadata,
        picks=picks,
        preload=True,
        reject_by_annotation=True,
        verbose="error",
    )
    return epochs


def stage_subject(
    *,
    bids_root: Path,
    staged_dir: Path,
    subject: str,
    session: str,
    task: str,
    runs: Iterable[str],
    tmin: float,
    tmax: float,
    baseline: tuple[float | None, float | None] | None,
    picks: str,
    condition_column: str,
    face_conditions: Iterable[str],
    scrambled_conditions: Iterable[str],
    max_events_per_label: int | None = None,
    overwrite: bool = False,
) -> SubjectStageResult:
    subject = _normalise_subject(subject)
    normalised_runs = tuple(_normalise_run(run) for run in runs)
    staged_dir.mkdir(parents=True, exist_ok=True)
    epochs_path = staged_dir / f"{subject}_ds000117_faces_epo.fif"
    metadata_path = staged_dir / f"{subject}_ds000117_faces_metadata.csv"

    if not overwrite and epochs_path.is_file() and metadata_path.is_file():
        metadata = pd.read_csv(metadata_path)
        labels = tuple(sorted(str(label) for label in metadata["condition"].dropna().unique()))
        return SubjectStageResult(subject, epochs_path, metadata_path, len(metadata), labels, normalised_runs)

    run_epochs = []
    for run in normalised_runs:
        raw_path, events_path = _run_paths(
            bids_root,
            subject=subject,
            session=session,
            task=task,
            run=run,
        )
        run_epochs.append(
            _stage_run(
                raw_path,
                events_path,
                subject=subject,
                run=run,
                tmin=tmin,
                tmax=tmax,
                baseline=baseline,
                picks=picks,
                condition_column=condition_column,
                face_conditions=face_conditions,
                scrambled_conditions=scrambled_conditions,
                max_events_per_label=max_events_per_label,
            )
        )

    epochs = run_epochs[0] if len(run_epochs) == 1 else mne.concatenate_epochs(run_epochs, add_offset=True, verbose="error")
    if epochs.metadata is None:
        raise ValueError(f"Staged epochs for {subject} do not contain metadata.")
    metadata = epochs.metadata.reset_index(drop=True)
    labels = tuple(sorted(str(label) for label in metadata["condition"].dropna().unique()))
    epochs.save(epochs_path, overwrite=True)
    metadata.to_csv(metadata_path, index=False)
    return SubjectStageResult(subject, epochs_path, metadata_path, len(metadata), labels, normalised_runs)


def write_manifest(
    results: list[SubjectStageResult],
    manifest_out: Path,
    *,
    decoders: Iterable[str],
    label_column: str,
    group_column: str,
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
                        "group_column": group_column,
                        "tmin": tmin,
                        "tmax": tmax,
                        "window_ms": window_ms,
                        "step_ms": step_ms,
                        "n_splits": n_splits,
                    }
                )


def _baseline(value: str) -> tuple[float | None, float | None] | None:
    if value.strip().lower() in {"none", "off", "false"}:
        return None
    start, stop = _split_csv_or_space(value)
    return (None if start.lower() == "none" else float(start), None if stop.lower() == "none" else float(stop))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bids-root", type=Path, required=True)
    parser.add_argument("--staged-dir", type=Path, required=True)
    parser.add_argument("--manifest-out", type=Path, required=True)
    parser.add_argument("--subjects", nargs="+", default=["01"])
    parser.add_argument("--session", default="meg")
    parser.add_argument("--task", default="facerecognition")
    parser.add_argument("--runs", nargs="+", default=["01", "02"])
    parser.add_argument("--condition-column", default="trial_type")
    parser.add_argument("--face-conditions", nargs="+", default=list(DEFAULT_FACE_CONDITIONS))
    parser.add_argument("--scrambled-conditions", nargs="+", default=list(DEFAULT_SCRAMBLED_CONDITIONS))
    parser.add_argument("--picks", default="meg")
    parser.add_argument("--tmin", type=float, default=-0.2)
    parser.add_argument("--tmax", type=float, default=0.8)
    parser.add_argument("--baseline", default="None,0")
    parser.add_argument("--max-events-per-label", type=int)
    parser.add_argument("--decoders", nargs="+", default=list(DEFAULT_DECODERS))
    parser.add_argument("--window-ms", type=float, default=20.0)
    parser.add_argument("--step-ms", type=float, default=10.0)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    face_conditions = [label for value in args.face_conditions for label in _split_csv_or_space(value)]
    scrambled_conditions = [label for value in args.scrambled_conditions for label in _split_csv_or_space(value)]
    baseline = _baseline(args.baseline)
    staged_results = []
    for subject in args.subjects:
        result = stage_subject(
            bids_root=args.bids_root,
            staged_dir=args.staged_dir,
            subject=subject,
            session=args.session,
            task=args.task,
            runs=args.runs,
            tmin=args.tmin,
            tmax=args.tmax,
            baseline=baseline,
            picks=args.picks,
            condition_column=args.condition_column,
            face_conditions=face_conditions,
            scrambled_conditions=scrambled_conditions,
            max_events_per_label=args.max_events_per_label,
            overwrite=args.overwrite,
        )
        staged_results.append(result)
        print(f"Staged {result.subject}: {result.n_trials} trials, labels={','.join(result.labels)}, runs={','.join(result.runs)}")

    write_manifest(
        staged_results,
        args.manifest_out,
        decoders=args.decoders,
        label_column="condition",
        group_column="run",
        tmin=args.tmin,
        tmax=args.tmax,
        window_ms=args.window_ms,
        step_ms=args.step_ms,
        n_splits=args.n_splits,
    )
    print(f"Wrote ds000117 RepTrace manifest: {args.manifest_out}")


if __name__ == "__main__":
    main()
