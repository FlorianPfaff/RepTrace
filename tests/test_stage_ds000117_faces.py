from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import mne
import numpy as np
import pandas as pd


_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "stage_ds000117_faces.py"
_SPEC = importlib.util.spec_from_file_location("stage_ds000117_faces", _SCRIPT_PATH)
assert _SPEC is not None
stage_ds000117_faces = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
sys.modules[_SPEC.name] = stage_ds000117_faces
_SPEC.loader.exec_module(stage_ds000117_faces)


def test_metadata_for_run_builds_binary_face_labels(tmp_path: Path):
    events = tmp_path / "sub-01_ses-meg_task-facerecognition_run-01_events.tsv"
    events.write_text(
        "onset\tduration\ttrial_type\n"
        "0.10\t0.20\tFamous\n"
        "0.30\t0.20\tUnfamiliar\n"
        "0.50\t0.20\tScrambled\n"
        "0.70\t0.20\tOther\n",
        encoding="utf-8",
    )

    metadata = stage_ds000117_faces._metadata_for_run(
        events,
        subject="sub-01",
        run="01",
        condition_column="trial_type",
        face_conditions=("Famous", "Unfamiliar"),
        scrambled_conditions=("Scrambled",),
    )

    assert metadata["condition"].tolist() == ["face", "face", "scrambled"]
    assert metadata["original_condition"].tolist() == ["Famous", "Unfamiliar", "Scrambled"]
    assert metadata["run"].tolist() == ["01", "01", "01"]


def test_metadata_for_run_can_limit_events_per_label(tmp_path: Path):
    events = tmp_path / "events.tsv"
    events.write_text(
        "onset\ttrial_type\n"
        "0.10\tFamous\n"
        "0.20\tUnfamiliar\n"
        "0.30\tScrambled\n"
        "0.40\tScrambled\n",
        encoding="utf-8",
    )

    metadata = stage_ds000117_faces._metadata_for_run(
        events,
        subject="sub-01",
        run="01",
        condition_column="trial_type",
        face_conditions=("Famous", "Unfamiliar"),
        scrambled_conditions=("Scrambled",),
        max_events_per_label=1,
    )

    assert metadata["condition"].tolist() == ["face", "scrambled"]


def test_events_from_metadata_uses_raw_sampling():
    info = mne.create_info(["MEG0111"], sfreq=1000.0, ch_types="mag")
    raw = mne.io.RawArray(np.zeros((1, 1000)), info, first_samp=100, verbose="error")
    metadata = pd.DataFrame({"onset": [0.010, 0.025], "condition": ["face", "scrambled"]})

    events = stage_ds000117_faces._events_from_metadata(raw, metadata)

    assert events.tolist() == [[110, 0, 1], [125, 0, 2]]


def test_write_manifest_repeats_rows_for_decoders(tmp_path: Path):
    result = stage_ds000117_faces.SubjectStageResult(
        subject="sub-01",
        epochs_path=tmp_path / "sub-01_epo.fif",
        metadata_path=tmp_path / "sub-01_metadata.csv",
        n_trials=3,
        labels=("face", "scrambled"),
        runs=("01", "02"),
    )
    manifest = tmp_path / "manifest.csv"

    stage_ds000117_faces.write_manifest(
        [result],
        manifest,
        decoders=("logistic", "lda"),
        label_column="condition",
        group_column="run",
        tmin=-0.2,
        tmax=0.8,
        window_ms=20.0,
        step_ms=10.0,
        n_splits=5,
        max_iter=5000,
    )

    rows = pd.read_csv(manifest)
    assert rows["decoder"].tolist() == ["logistic", "lda"]
    assert rows["epochs"].tolist() == ["sub-01_epo.fif", "sub-01_epo.fif"]
    assert rows["metadata_csv"].tolist() == ["sub-01_metadata.csv", "sub-01_metadata.csv"]
    assert rows["label_column"].tolist() == ["condition", "condition"]
    assert rows["group_column"].tolist() == ["run", "run"]
    assert rows["max_iter"].tolist() == [5000, 5000]
