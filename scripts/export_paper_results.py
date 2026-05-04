"""Export compact RepTrace result artifacts into a paper repository."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from reptrace.plot_calibration import plot_reliability_diagram


DEFAULT_PATTERNS = (
    "summary.csv",
    "summary.png",
    "reliability.png",
    "report.md",
    "calibration_report.md",
    "paired_stats.csv",
    "paired_stats.md",
    "reliability_bins.csv",
    "inference_*_time.csv",
    "inference_*_clusters.csv",
    "validation.csv",
)


def collect_artifacts(source_dir: Path, patterns: tuple[str, ...]) -> list[Path]:
    """Return compact artifact files matching the provided glob patterns."""
    artifacts: list[Path] = []
    for pattern in patterns:
        artifacts.extend(path for path in source_dir.glob(pattern) if path.is_file())
    return sorted(set(artifacts))


def total_size(paths: list[Path]) -> int:
    """Return total byte size for paths."""
    return sum(path.stat().st_size for path in paths)


def export_artifacts(
    source_dir: Path,
    destination_dir: Path,
    *,
    patterns: tuple[str, ...] = DEFAULT_PATTERNS,
    max_mb: float = 50.0,
    dry_run: bool = False,
    plot_reliability: bool = False,
    reliability_window: tuple[float, float] | None = None,
    reliability_title: str | None = None,
) -> list[Path]:
    """Copy compact artifacts from ``source_dir`` to ``destination_dir``."""
    source_dir = source_dir.resolve()
    destination_dir = destination_dir.resolve()
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory does not exist: {source_dir}")

    artifacts = collect_artifacts(source_dir, patterns)
    if not artifacts:
        raise FileNotFoundError(f"No artifacts matched in {source_dir}")

    size_mb = total_size(artifacts) / (1024 * 1024)
    if size_mb > max_mb:
        raise ValueError(f"Matched artifacts are {size_mb:.2f} MB, above limit {max_mb:.2f} MB")

    if dry_run:
        return artifacts

    destination_dir.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    for artifact in artifacts:
        target = destination_dir / artifact.name
        shutil.copy2(artifact, target)
        copied.append(target)
    reliability_bins = destination_dir / "reliability_bins.csv"
    reliability_plot = destination_dir / "reliability.png"
    if plot_reliability and reliability_bins.exists() and not reliability_plot.exists():
        plot_reliability_diagram(
            reliability_bins,
            reliability_plot,
            time_window=reliability_window,
            title=reliability_title,
        )
        copied.append(reliability_plot)
    return copied


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_dir", type=Path, help="RepTrace result directory to export from.")
    parser.add_argument("destination_dir", type=Path, help="Paper-repo result directory to copy into.")
    parser.add_argument("--max-mb", type=float, default=50.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--pattern", action="append", dest="patterns", help="Additional or replacement glob pattern.")
    parser.add_argument("--plot-reliability", action="store_true", help="Generate reliability.png from copied reliability_bins.csv.")
    parser.add_argument("--reliability-window", type=float, nargs=2, metavar=("START", "STOP"))
    parser.add_argument("--reliability-title")
    args = parser.parse_args()

    patterns = tuple(args.patterns) if args.patterns else DEFAULT_PATTERNS
    exported = export_artifacts(
        args.source_dir,
        args.destination_dir,
        patterns=patterns,
        max_mb=args.max_mb,
        dry_run=args.dry_run,
        plot_reliability=args.plot_reliability,
        reliability_window=tuple(args.reliability_window) if args.reliability_window else None,
        reliability_title=args.reliability_title,
    )
    size_mb = total_size(exported) / (1024 * 1024)
    action = "Would export" if args.dry_run else "Exported"
    print(f"{action} {len(exported)} file(s), {size_mb:.3f} MB total:")
    for path in exported:
        print(path)


if __name__ == "__main__":
    main()
