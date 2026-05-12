from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "print_ds000117_download_commands.py"
_SPEC = importlib.util.spec_from_file_location("print_ds000117_download_commands", _SCRIPT_PATH)
assert _SPEC is not None
print_ds000117_download_commands = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
sys.modules[_SPEC.name] = print_ds000117_download_commands
_SPEC.loader.exec_module(print_ds000117_download_commands)


def test_ds000117_face_run_includes_are_exact_paths():
    includes = print_ds000117_download_commands.ds000117_face_run_includes(
        subjects=["1", "sub-02"],
        runs=["1", "02"],
    )

    assert includes == [
        "sub-01/ses-meg/meg/sub-01_ses-meg_task-facerecognition_run-01_meg.fif",
        "sub-01/ses-meg/meg/sub-01_ses-meg_task-facerecognition_run-01_events.tsv",
        "sub-01/ses-meg/meg/sub-01_ses-meg_task-facerecognition_run-02_meg.fif",
        "sub-01/ses-meg/meg/sub-01_ses-meg_task-facerecognition_run-02_events.tsv",
        "sub-02/ses-meg/meg/sub-02_ses-meg_task-facerecognition_run-01_meg.fif",
        "sub-02/ses-meg/meg/sub-02_ses-meg_task-facerecognition_run-01_events.tsv",
        "sub-02/ses-meg/meg/sub-02_ses-meg_task-facerecognition_run-02_meg.fif",
        "sub-02/ses-meg/meg/sub-02_ses-meg_task-facerecognition_run-02_events.tsv",
    ]


def test_openneuro_download_commands_use_one_include_per_command():
    commands = print_ds000117_download_commands.openneuro_download_commands(
        ["sub-01/ses-meg/meg/sub-01_ses-meg_task-facerecognition_run-01_events.tsv"],
        target_dir=Path("data/ds000117"),
    )

    assert commands == [
        "openneuro-py download --dataset ds000117 --tag 1.1.0 --target-dir data/ds000117 --include sub-01/ses-meg/meg/sub-01_ses-meg_task-facerecognition_run-01_events.tsv"
    ]
