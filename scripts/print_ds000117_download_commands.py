"""Print exact OpenNeuro download commands for ds000117 MEG face runs."""

from __future__ import annotations

import argparse
import shlex
from pathlib import Path


def _normalise_subject(subject: str | int) -> str:
    text = str(subject).strip()
    if text.startswith("sub-"):
        return text
    return f"sub-{int(text):02d}"


def _normalise_run(run: str | int) -> str:
    text = str(run).strip()
    return text if text.startswith("run-") else f"{int(text):02d}"


def ds000117_face_run_includes(
    *,
    subjects: list[str],
    runs: list[str],
    session: str = "meg",
    task: str = "facerecognition",
) -> list[str]:
    """Return exact OpenNeuro include paths for ds000117 face-recognition runs."""

    includes = []
    for subject_value in subjects:
        subject = _normalise_subject(subject_value)
        for run_value in runs:
            run = _normalise_run(run_value)
            stem = f"{subject}_ses-{session}_task-{task}_run-{run}"
            base = f"{subject}/ses-{session}/meg/{stem}"
            includes.extend([f"{base}_meg.fif", f"{base}_events.tsv"])
    return includes


def openneuro_download_commands(
    includes: list[str],
    *,
    dataset: str = "ds000117",
    tag: str = "1.1.0",
    target_dir: Path = Path("data/ds000117"),
) -> list[str]:
    """Return one exact-include openneuro-py command per file."""

    commands = []
    for include in includes:
        parts = [
            "openneuro-py",
            "download",
            "--dataset",
            dataset,
            "--tag",
            tag,
            "--target-dir",
            target_dir.as_posix(),
            "--include",
            include,
        ]
        commands.append(" ".join(shlex.quote(part) for part in parts))
    return commands


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subjects", nargs="+", default=["01"])
    parser.add_argument("--runs", nargs="+", default=["01", "02"])
    parser.add_argument("--session", default="meg")
    parser.add_argument("--task", default="facerecognition")
    parser.add_argument("--dataset", default="ds000117")
    parser.add_argument("--tag", default="1.1.0")
    parser.add_argument("--target-dir", type=Path, default=Path("data/ds000117"))
    args = parser.parse_args()

    includes = ds000117_face_run_includes(
        subjects=args.subjects,
        runs=args.runs,
        session=args.session,
        task=args.task,
    )
    commands = openneuro_download_commands(
        includes,
        dataset=args.dataset,
        tag=args.tag,
        target_dir=args.target_dir,
    )
    print("\n".join(commands))


if __name__ == "__main__":
    main()
