from __future__ import annotations

import argparse
import glob
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from reptrace.onset_detection import (
    DEFAULT_THRESHOLD_QUANTILE,
    DEFAULT_THRESHOLD_WINDOW,
    detect_onsets_from_csvs,
)

DEFAULT_OBSERVATIONS_GLOB = "observations/*_observations.csv"


@dataclass(frozen=True)
class TaskOnsetOutput:
    """Output paths and counts for one task-directory onset run."""

    task: str
    task_dir: Path
    observation_csvs: list[Path]
    events_csv: Path
    summary_csv: Path
    n_events: int
    n_summary_rows: int


@dataclass(frozen=True)
class OnsetWorkflowRun:
    """Top-level outputs from a multi-task onset workflow."""

    out_dir: Path
    task_outputs: list[TaskOnsetOutput]
    summary_all_csv: Path
    events_all_csv: Path | None
    plot_path: Path | None


def _expand_task_dirs(patterns: list[str]) -> list[Path]:
    task_dirs: list[Path] = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if matches:
            task_dirs.extend(Path(match) for match in matches)
        else:
            task_dirs.append(Path(pattern))
    return task_dirs


def _task_name(task_dir: Path) -> str:
    return task_dir.name or task_dir.resolve().name


def _observation_paths(task_dir: Path, observations_glob: str) -> list[Path]:
    return sorted(task_dir.glob(observations_glob))


def _insert_task_columns(frame: pd.DataFrame, *, task: str, task_dir: Path) -> pd.DataFrame:
    tagged = frame.copy()
    tagged.insert(0, "task_dir", str(task_dir))
    tagged.insert(0, "task", task)
    return tagged


# pylint: disable-next=too-many-arguments
def run_task_onset_detection(
    task_dir: Path,
    *,
    out_dir: Path,
    observations_glob: str = DEFAULT_OBSERVATIONS_GLOB,
    threshold_window: tuple[float, float] = DEFAULT_THRESHOLD_WINDOW,
    threshold_quantile: float = DEFAULT_THRESHOLD_QUANTILE,
    score_column: str = "confidence",
    detection_start: float | None = None,
    min_consecutive: int = 1,
    min_duration: float | None = None,
    require_stable_prediction: bool = False,
) -> tuple[TaskOnsetOutput, pd.DataFrame, pd.DataFrame]:
    """Run onset detection for one benchmark task directory.

    The task directory is expected to contain observation files under
    ``observations/*_observations.csv`` unless ``observations_glob`` is changed.
    Outputs are written under ``out_dir / task_dir.name``.
    """

    task_dir = task_dir.resolve()
    task = _task_name(task_dir)
    observation_csvs = _observation_paths(task_dir, observations_glob)
    if not observation_csvs:
        raise FileNotFoundError(f"No observation CSVs found in {task_dir / observations_glob}.")

    events, summary = detect_onsets_from_csvs(
        observation_csvs,
        threshold_window=threshold_window,
        threshold_quantile=threshold_quantile,
        score_column=score_column,
        detection_start=detection_start,
        min_consecutive=min_consecutive,
        min_duration=min_duration,
        require_stable_prediction=require_stable_prediction,
    )
    tagged_events = _insert_task_columns(events, task=task, task_dir=task_dir)
    tagged_summary = _insert_task_columns(summary, task=task, task_dir=task_dir)

    task_out_dir = out_dir / task
    task_out_dir.mkdir(parents=True, exist_ok=True)
    events_csv = task_out_dir / "onset_events.csv"
    summary_csv = task_out_dir / "onset_summary.csv"
    tagged_events.to_csv(events_csv, index=False)
    tagged_summary.to_csv(summary_csv, index=False)

    output = TaskOnsetOutput(
        task=task,
        task_dir=task_dir,
        observation_csvs=observation_csvs,
        events_csv=events_csv,
        summary_csv=summary_csv,
        n_events=len(tagged_events),
        n_summary_rows=len(tagged_summary),
    )
    return output, tagged_events, tagged_summary


def plot_onset_summary(summary: pd.DataFrame, out_path: Path) -> Path:
    """Plot compact onset latency and false-alarm summaries."""

    if summary.empty:
        raise ValueError("Cannot plot an empty onset summary.")
    required = {"task", "post_detection_latency_median", "false_alarm_rate", "post_zero_detected_rate"}
    missing = sorted(required.difference(summary.columns))
    if missing:
        raise ValueError(f"Onset summary is missing required columns for plotting: {missing}")

    sort_columns = ["task", "decoder", "emission_mode"] if "decoder" in summary else ["task"]
    frame = summary.copy().sort_values(sort_columns)
    labels = frame["task"].astype(str)
    if "decoder" in frame.columns:
        labels = labels + "\n" + frame["decoder"].astype(str)
    if "emission_mode" in frame.columns:
        labels = labels + " / " + frame["emission_mode"].astype(str)

    fig, axes = plt.subplots(1, 2, figsize=(max(7.0, 0.8 * len(frame)), 4.2))
    positions = range(len(frame))

    axes[0].bar(positions, frame["post_detection_latency_median"])
    axes[0].set_xticks(list(positions))
    axes[0].set_xticklabels(labels, rotation=35, ha="right")
    axes[0].set_ylabel("Median post-zero onset latency (s)")
    axes[0].set_title("Onset latency")
    axes[0].axhline(0.0, color="0.4", linewidth=1.0)
    axes[0].grid(axis="y", color="0.9", linewidth=0.8)

    width = 0.38
    axes[1].bar(
        [position - width / 2 for position in positions],
        frame["false_alarm_rate"],
        width=width,
        label="false alarm",
    )
    axes[1].bar(
        [position + width / 2 for position in positions],
        frame["post_zero_detected_rate"],
        width=width,
        label="post-zero detected",
    )
    axes[1].set_xticks(list(positions))
    axes[1].set_xticklabels(labels, rotation=35, ha="right")
    axes[1].set_ylabel("Rate")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_title("Detection quality")
    axes[1].legend(loc="best")
    axes[1].grid(axis="y", color="0.9", linewidth=0.8)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


# pylint: disable-next=too-many-arguments,too-many-locals
def run_onset_workflow(
    task_dirs: list[Path],
    *,
    out_dir: Path,
    observations_glob: str = DEFAULT_OBSERVATIONS_GLOB,
    threshold_window: tuple[float, float] = DEFAULT_THRESHOLD_WINDOW,
    threshold_quantile: float = DEFAULT_THRESHOLD_QUANTILE,
    score_column: str = "confidence",
    detection_start: float | None = None,
    min_consecutive: int = 1,
    min_duration: float | None = None,
    require_stable_prediction: bool = False,
    allow_missing: bool = False,
    write_combined_events: bool = False,
    plot_out: Path | None = None,
) -> OnsetWorkflowRun:
    """Run onset detection across multiple task result directories."""

    if not task_dirs:
        raise ValueError("At least one task directory is required.")

    out_dir.mkdir(parents=True, exist_ok=True)
    task_outputs: list[TaskOnsetOutput] = []
    event_frames: list[pd.DataFrame] = []
    summary_frames: list[pd.DataFrame] = []
    missing_tasks: list[str] = []

    for task_dir in task_dirs:
        try:
            output, events, summary = run_task_onset_detection(
                task_dir,
                out_dir=out_dir,
                observations_glob=observations_glob,
                threshold_window=threshold_window,
                threshold_quantile=threshold_quantile,
                score_column=score_column,
                detection_start=detection_start,
                min_consecutive=min_consecutive,
                min_duration=min_duration,
                require_stable_prediction=require_stable_prediction,
            )
        except FileNotFoundError:
            if not allow_missing:
                raise
            missing_tasks.append(str(task_dir))
            continue
        task_outputs.append(output)
        event_frames.append(events)
        summary_frames.append(summary)

    if not summary_frames:
        raise FileNotFoundError(
            "No task observation CSVs were found. Missing task directories: "
            + ", ".join(missing_tasks)
        )

    summary_all = pd.concat(summary_frames, ignore_index=True)
    summary_all_csv = out_dir / "onset_summary_all.csv"
    summary_all.to_csv(summary_all_csv, index=False)

    events_all_csv = None
    if write_combined_events:
        events_all = pd.concat(event_frames, ignore_index=True)
        events_all_csv = out_dir / "onset_events_all.csv"
        events_all.to_csv(events_all_csv, index=False)

    plot_path = None
    if plot_out is not None:
        plot_path = plot_onset_summary(summary_all, plot_out)

    return OnsetWorkflowRun(
        out_dir=out_dir,
        task_outputs=task_outputs,
        summary_all_csv=summary_all_csv,
        events_all_csv=events_all_csv,
        plot_path=plot_path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RepTrace onset detection across task result directories.")
    parser.add_argument(
        "--task-dir",
        action="append",
        required=True,
        help="Task result directory. Repeat for multiple tasks; glob patterns are accepted.",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--observations-glob", default=DEFAULT_OBSERVATIONS_GLOB)
    parser.add_argument(
        "--threshold-window",
        nargs=2,
        type=float,
        default=DEFAULT_THRESHOLD_WINDOW,
        metavar=("START", "STOP"),
    )
    parser.add_argument("--threshold-quantile", type=float, default=DEFAULT_THRESHOLD_QUANTILE)
    parser.add_argument("--score-column", default="confidence")
    parser.add_argument("--detection-start", type=float)
    parser.add_argument("--min-consecutive", type=int, default=1)
    parser.add_argument("--min-duration", type=float)
    parser.add_argument("--require-stable-prediction", action="store_true")
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument("--write-combined-events", action="store_true")
    parser.add_argument("--plot-out", type=Path)
    args = parser.parse_args()

    run = run_onset_workflow(
        _expand_task_dirs(args.task_dir),
        out_dir=args.out_dir,
        observations_glob=args.observations_glob,
        threshold_window=tuple(args.threshold_window),
        threshold_quantile=args.threshold_quantile,
        score_column=args.score_column,
        detection_start=args.detection_start,
        min_consecutive=args.min_consecutive,
        min_duration=args.min_duration,
        require_stable_prediction=args.require_stable_prediction,
        allow_missing=args.allow_missing,
        write_combined_events=args.write_combined_events,
        plot_out=args.plot_out,
    )
    print(f"Wrote combined onset summary: {run.summary_all_csv}")
    if run.events_all_csv is not None:
        print(f"Wrote combined onset events: {run.events_all_csv}")
    if run.plot_path is not None:
        print(f"Wrote onset plot: {run.plot_path}")
    for output in run.task_outputs:
        print(
            f"{output.task}: {len(output.observation_csvs)} observation file(s), "
            f"{output.n_events} event row(s), {output.n_summary_rows} summary row(s)."
        )


if __name__ == "__main__":
    main()
