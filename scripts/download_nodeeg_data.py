"""Stage NOD-EEG benchmark files from OwnCloud/WebDAV via rclone."""

from __future__ import annotations

import argparse
import os
import subprocess  # nosec B404
from pathlib import Path


EXPECTED_NODEEG_SUBJECTS = (
    "01",
    "02",
    "03",
    "04",
    "05",
    "06",
    "07",
    "08",
    "09",
    "10",
    "11",
    "12",
    "13",
    "14",
    "24",
    "26",
    "27",
    "29",
    "30",
)
DEFAULT_INCLUDE_PATTERNS = ("sub-*_epo.fif", "sub-*_events.csv")


def _split_csv(value: str | None) -> list[str]:
    if value is None:
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


def _required_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise SystemExit(f"Missing required environment variable: {name}")
    return value


def _run(args: list[str], *, capture_stdout: bool = False) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(  # nosec B603
        args,
        check=False,
        text=True,
        stdout=subprocess.PIPE if capture_stdout else None,
        stderr=subprocess.PIPE if capture_stdout else None,
    )
    if completed.returncode != 0:
        if completed.stderr:
            print(completed.stderr)
        raise SystemExit(f"Command failed: {args[0]} {args[1] if len(args) > 1 else ''}".strip())
    return completed


def _remote_path(remote_path: str) -> str:
    cleaned = remote_path.strip().replace("\\", "/").strip("/")
    return ":webdav:" if not cleaned else f":webdav:{cleaned}"


def _rclone_webdav_options(*, url: str, user: str, obscured_password: str) -> list[str]:
    return [
        "--webdav-url",
        url,
        "--webdav-vendor",
        "owncloud",
        "--webdav-user",
        user,
        "--webdav-pass",
        obscured_password,
    ]


def _staged_subjects(data_root: Path) -> set[str]:
    subjects = set()
    for subject in EXPECTED_NODEEG_SUBJECTS:
        if (data_root / f"sub-{subject}_epo.fif").exists() and (data_root / f"sub-{subject}_events.csv").exists():
            subjects.add(subject)
    return subjects


def _validate_staged_subjects(data_root: Path, required_count: int) -> None:
    staged = _staged_subjects(data_root)
    if len(staged) < required_count:
        expected = ", ".join(f"sub-{subject}" for subject in EXPECTED_NODEEG_SUBJECTS)
        found = ", ".join(f"sub-{subject}" for subject in sorted(staged)) or "none"
        raise SystemExit(f"Expected at least {required_count} staged NOD-EEG subjects in {data_root}, found {len(staged)} ({found}). Expected subjects: {expected}")


def download_nodeeg_data(
    *,
    data_root: Path,
    remote_path: str,
    include_patterns: list[str],
    required_subject_count: int,
    rclone_binary: str,
) -> None:
    webdav_url = _required_env("NODEEG_WEBDAV_URL")
    webdav_user = _required_env("NODEEG_DATA_KEY")
    webdav_password = _required_env("NODEEG_DATA_PASSWORD")

    obscured = _run([rclone_binary, "obscure", webdav_password], capture_stdout=True).stdout.strip()
    if not obscured:
        raise SystemExit("rclone did not return an obscured WebDAV password.")

    data_root.mkdir(parents=True, exist_ok=True)
    copy_args = [
        rclone_binary,
        "copy",
        _remote_path(remote_path),
        str(data_root),
        *_rclone_webdav_options(url=webdav_url, user=webdav_user, obscured_password=obscured),
        "--checkers",
        "4",
        "--transfers",
        "2",
        "--stats",
        "30s",
        "--progress",
    ]
    for pattern in include_patterns:
        copy_args.extend(["--include", pattern])
    copy_args.extend(["--exclude", "*"])

    _run(copy_args)
    _validate_staged_subjects(data_root, required_subject_count)
    for index, path in enumerate(sorted(data_root.glob("sub-*_*.*")), start=1):
        print(f"Staged file #{index}: {path.name}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/nod"), help="Local flat NOD-EEG staging directory.")
    parser.add_argument("--remote-path", default="", help="Optional path below the WebDAV share root.")
    parser.add_argument("--include", default=",".join(DEFAULT_INCLUDE_PATTERNS), help="Comma-separated rclone include patterns.")
    parser.add_argument("--require-subject-count", type=int, default=len(EXPECTED_NODEEG_SUBJECTS), help="Minimum staged subject count required after download.")
    parser.add_argument("--rclone-binary", default="rclone")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    include_patterns = _split_csv(args.include)
    if not include_patterns:
        raise SystemExit("At least one --include pattern is required.")
    download_nodeeg_data(
        data_root=args.data_root,
        remote_path=args.remote_path,
        include_patterns=include_patterns,
        required_subject_count=args.require_subject_count,
        rclone_binary=args.rclone_binary,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
