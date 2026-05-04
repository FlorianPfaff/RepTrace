from __future__ import annotations

import argparse
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from pandas.errors import EmptyDataError

from reptrace.benchmark import run_benchmark_manifest
from reptrace.decoding import DECODER_CHOICES, normalize_decoder_name
from reptrace.emission_compare import compare_temporal_summary
from reptrace.plot_time_decode import plot_time_decode_results
from reptrace.semantic_stages import analyze_semantic_stages
from reptrace.temporal_model import fit_temporal_models
from reptrace.validate_manifest import validate_manifest, validation_report_frame


@dataclass(frozen=True)
class Paper2Task:
    """One NOD task used in the calibration-aware temporal-state workflow."""

    task_id: str
    label: str
    manifest_name: str


@dataclass(frozen=True)
class Paper2TaskOutput:
    """Compact paths produced for one Paper 2 task."""

    task: Paper2Task
    task_dir: Path
    manifest_csv: Path
    validation_csv: Path
    temporal_summary_csv: Path
    state_trace_csv: Path
    emission_compare_csv: Path
    emission_compare_report: Path
    semantic_time_csv: Path
    semantic_stages_csv: Path
    semantic_stages_report: Path


@dataclass(frozen=True)
class Paper2WorkflowRun:
    """Top-level Paper 2 workflow outputs."""

    out_dir: Path
    task_outputs: list[Paper2TaskOutput]
    paper2_summary_csv: Path
    paper2_figure: Path
    paper2_report: Path
    command_log: Path
    exported_artifacts: list[Path]


DEFAULT_TASKS = (
    Paper2Task("nod_animate", "NOD animate/inanimate", "nod_animate_all.csv"),
    Paper2Task("nod_canine_device", "NOD canine/device", "nod_superclass_canine_device_all.csv"),
    Paper2Task("nod_container_covering", "NOD container/covering", "nod_superclass_container_covering_all.csv"),
)
DEFAULT_TASK_IDS = tuple(task.task_id for task in DEFAULT_TASKS)
DEFAULT_DECODERS = ("logistic", "linear_svm")
PAPER2_COMPACT_PATTERNS = (
    "paper2_summary.csv",
    "paper2_stage_reliability.png",
    "paper2_evidence.md",
    "paper2_commands.md",
    "temporal_model_all.csv",
    "emission_compare_all.csv",
    "semantic_stage_time_all.csv",
    "semantic_stages_all.csv",
    "*/validation.csv",
    "*/summary.csv",
    "*/summary.png",
    "*/temporal_model.csv",
    "*/emission_compare.csv",
    "*/emission_compare.md",
    "*/semantic_stage_time.csv",
    "*/semantic_stages.csv",
    "*/semantic_stages.md",
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _normal_decoders(decoders: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    if not decoders:
        raise ValueError("At least one decoder is required.")
    return tuple(normalize_decoder_name(decoder) for decoder in decoders)


def _selected_tasks(task_ids: tuple[str, ...] | list[str] | None) -> tuple[Paper2Task, ...]:
    tasks_by_id = {task.task_id: task for task in DEFAULT_TASKS}
    if task_ids is None:
        return DEFAULT_TASKS
    unknown = sorted(set(task_ids).difference(tasks_by_id))
    if unknown:
        raise ValueError(f"Unknown Paper 2 task(s): {', '.join(unknown)}")
    return tuple(tasks_by_id[task_id] for task_id in task_ids)


def _display_path(path: Path, base: Path) -> str:
    try:
        return str(path.resolve().relative_to(base.resolve()))
    except ValueError:
        return str(path)


def prepare_paper2_manifest(
    source_manifest: Path,
    out_manifest: Path,
    *,
    data_root: Path | None,
    decoders: tuple[str, ...],
    max_subjects: int | None = None,
    expected_subjects: int | None = 19,
) -> pd.DataFrame:
    """Prepare a task manifest with runner-local data paths and decoder rows."""
    manifest = pd.read_csv(source_manifest)
    if "subject" not in manifest.columns:
        raise ValueError(f"{source_manifest} is missing a subject column.")

    if max_subjects is not None:
        if max_subjects < 1:
            raise ValueError("max_subjects must be positive.")
        keep_subjects = list(dict.fromkeys(manifest["subject"].astype(str)))[:max_subjects]
        manifest = manifest.loc[manifest["subject"].astype(str).isin(keep_subjects)].copy()

    n_subjects = manifest["subject"].astype(str).nunique()
    if max_subjects is None and expected_subjects is not None and n_subjects != expected_subjects:
        raise ValueError(f"{source_manifest} has {n_subjects} unique subject(s), expected {expected_subjects}.")

    if data_root is not None:
        data_root = data_root.expanduser().resolve()
        for column in ("epochs", "events_csv"):
            if column in manifest.columns:
                manifest[column] = manifest[column].map(lambda value: str(data_root / Path(str(value)).name))

    decoder_rows = []
    for decoder in decoders:
        decoder_frame = manifest.copy()
        decoder_frame["decoder"] = decoder
        decoder_rows.append(decoder_frame)
    prepared = pd.concat(decoder_rows, ignore_index=True)
    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    prepared.to_csv(out_manifest, index=False)
    return prepared


def _validate_prepared_manifest(manifest_csv: Path, validation_csv: Path) -> pd.DataFrame:
    validations = validate_manifest(manifest_csv)
    report = validation_report_frame(validations)
    validation_csv.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(validation_csv, index=False)
    if not all(validation.ok for validation in validations):
        failures = report.loc[~report["ok"], ["subject", "messages"]].head(5).to_dict("records")
        raise ValueError(f"Manifest validation failed for {manifest_csv}: {failures}")
    return report


def _task_observation_paths(task_dir: Path) -> list[Path]:
    return sorted((task_dir / "observations").glob("*_observations.csv"))


def _read_optional_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except EmptyDataError:
        return pd.DataFrame()


def _tagged_csv(task_output: Paper2TaskOutput, path: Path) -> pd.DataFrame:
    frame = _read_optional_csv(path)
    if frame.empty:
        return frame
    frame.insert(0, "task_label", task_output.task.label)
    frame.insert(0, "task", task_output.task.task_id)
    return frame


def _write_combined_csv(task_outputs: list[Paper2TaskOutput], attr_name: str, out_csv: Path) -> pd.DataFrame:
    frames = []
    for task_output in task_outputs:
        frames.append(_tagged_csv(task_output, getattr(task_output, attr_name)))
    combined = pd.concat([frame for frame in frames if not frame.empty], ignore_index=True) if any(not frame.empty for frame in frames) else pd.DataFrame()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out_csv, index=False)
    return combined


def _mode_subset(frame: pd.DataFrame, task: str, decoder: str, emission_mode: str) -> pd.DataFrame:
    required = {"task", "decoder", "emission_mode"}
    if frame.empty or not required.issubset(frame.columns):
        return pd.DataFrame()
    return frame[
        (frame["task"].astype(str) == task)
        & (frame["decoder"].astype(str) == decoder)
        & (frame["emission_mode"].astype(str) == emission_mode)
    ]


def _stage_stats(stages: pd.DataFrame, stage_time: pd.DataFrame, task: str, decoder: str, emission_mode: str) -> dict[str, float | int]:
    stage_subset = _mode_subset(stages, task, decoder, emission_mode)
    time_subset = _mode_subset(stage_time, task, decoder, emission_mode)
    peak_subjects = 0
    if "posterior_true_class_mean" in time_subset and "n_subjects" in time_subset and not time_subset.empty:
        peak_subjects = int(time_subset.loc[time_subset["posterior_true_class_mean"].idxmax(), "n_subjects"])
    return {
        f"{emission_mode}_n_stages": int(len(stage_subset)),
        f"{emission_mode}_mean_stage_duration": float(stage_subset["duration"].mean()) if "duration" in stage_subset and not stage_subset.empty else float("nan"),
        f"{emission_mode}_peak_posterior_true_class": float(time_subset["posterior_true_class_mean"].max())
        if "posterior_true_class_mean" in time_subset and not time_subset.empty
        else float("nan"),
        f"{emission_mode}_peak_n_subjects": peak_subjects,
        f"{emission_mode}_stage_subjects_min": int(stage_subset["n_subjects_min"].min())
        if "n_subjects_min" in stage_subset and not stage_subset.empty
        else 0,
    }


def build_paper2_summary(emission_compare: pd.DataFrame, stages: pd.DataFrame, stage_time: pd.DataFrame) -> pd.DataFrame:
    """Build the central compact table for the Paper 2 evidence note."""
    if emission_compare.empty:
        return pd.DataFrame()

    rows = []
    for row in emission_compare.itertuples(index=False):
        task = str(row.task)
        decoder = str(row.decoder)
        rows.append(
            {
                "task": task,
                "task_label": str(row.task_label),
                "decoder": decoder,
                "preferred_emission_mode": str(row.preferred_emission_mode),
                "delta_control_margin": float(row.delta_control_margin),
                "calibrated_control_margin": float(row.calibrated_control_margin),
                "uncalibrated_control_margin": float(row.uncalibrated_control_margin),
                "delta_effect_minus_baseline_gain": float(row.delta_effect_minus_baseline_gain),
                "calibrated_best_stay_probability": float(row.calibrated_best_stay_probability),
                "uncalibrated_best_stay_probability": float(row.uncalibrated_best_stay_probability),
                **_stage_stats(stages, stage_time, task, decoder, "calibrated"),
                **_stage_stats(stages, stage_time, task, decoder, "uncalibrated"),
            }
        )
    return pd.DataFrame(rows).sort_values(["task", "decoder"]).reset_index(drop=True)


def plot_paper2_stage_reliability(
    paper2_summary: pd.DataFrame,
    stage_time: pd.DataFrame,
    out_path: Path,
    *,
    plot_decoder: str | None = None,
) -> Path:
    """Plot the Paper 2 control-margin and semantic-stage reliability summary."""
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))

    ax = axes[0]
    if paper2_summary.empty:
        ax.text(0.5, 0.5, "No emission comparison rows", ha="center", va="center")
        ax.axis("off")
    else:
        summary = paper2_summary.copy()
        task_labels = list(dict.fromkeys(summary["task_label"].astype(str)))
        decoders = list(dict.fromkeys(summary["decoder"].astype(str)))
        width = 0.8 / max(len(decoders), 1)
        x = range(len(task_labels))
        for decoder_index, decoder in enumerate(decoders):
            values = []
            for task_label in task_labels:
                match = summary[(summary["task_label"] == task_label) & (summary["decoder"] == decoder)]
                values.append(float(match["delta_control_margin"].iloc[0]) if not match.empty else float("nan"))
            offsets = [position - 0.4 + width / 2 + decoder_index * width for position in x]
            ax.bar(offsets, values, width=width, label=decoder)
        ax.axhline(0.0, color="0.35", linewidth=1.0)
        ax.set_xticks(list(x))
        ax.set_xticklabels(task_labels, rotation=25, ha="right")
        ax.set_ylabel("Delta control margin")
        ax.set_title("Calibrated vs uncalibrated emissions")
        ax.legend(loc="best")
        ax.grid(axis="y", color="0.9", linewidth=0.8)

    ax = axes[1]
    if stage_time.empty or "posterior_true_class_mean" not in stage_time.columns:
        ax.text(0.5, 0.5, "No semantic-stage time rows", ha="center", va="center")
        ax.axis("off")
    else:
        plot_frame = stage_time.copy()
        available_decoders = list(dict.fromkeys(plot_frame["decoder"].astype(str))) if "decoder" in plot_frame else []
        if plot_decoder is None:
            plot_decoder = "linear_svm" if "linear_svm" in available_decoders else (available_decoders[0] if available_decoders else None)
        if plot_decoder is not None and "decoder" in plot_frame:
            plot_frame = plot_frame.loc[plot_frame["decoder"].astype(str) == plot_decoder]
        grouped = (
            plot_frame.groupby(["emission_mode", "time"], as_index=False)["posterior_true_class_mean"]
            .mean()
            .sort_values(["emission_mode", "time"])
        )
        for emission_mode, group in grouped.groupby("emission_mode", sort=True):
            ax.plot(group["time"], group["posterior_true_class_mean"], label=str(emission_mode))
        ax.axvline(0.0, color="0.6", linestyle=":", linewidth=1.0)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Posterior on true class")
        ax.set_title(f"Semantic-stage reliability ({plot_decoder or 'decoder'})")
        ax.legend(loc="best")
        ax.grid(True, color="0.9", linewidth=0.8)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def _markdown_table(frame: pd.DataFrame, columns: list[str]) -> list[str]:
    if frame.empty:
        return ["No rows are available yet."]
    display = frame[columns].copy()
    for column in display.select_dtypes(include="number").columns:
        display[column] = display[column].map(lambda value: "" if pd.isna(value) else f"{value:.4f}")
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in display.itertuples(index=False):
        lines.append("| " + " | ".join(map(str, row)) + " |")
    return lines


def build_evidence_report(
    paper2_summary: pd.DataFrame,
    *,
    out_dir: Path,
    paper2_summary_csv: Path,
    figure_path: Path,
    temporal_all_csv: Path,
    emission_all_csv: Path,
    stage_time_all_csv: Path,
    stages_all_csv: Path,
) -> str:
    """Build a compact Paper 2 evidence note from generated artifacts."""
    columns = [
        "task_label",
        "decoder",
        "preferred_emission_mode",
        "delta_control_margin",
        "delta_effect_minus_baseline_gain",
        "calibrated_peak_posterior_true_class",
        "uncalibrated_peak_posterior_true_class",
        "calibrated_peak_n_subjects",
        "uncalibrated_peak_n_subjects",
    ]
    lines = [
        "# Paper 2 Evidence: Calibration-Aware Temporal State Inference",
        "",
        "Central claim under test: calibrated decoder probabilities can change downstream temporal state inference, not only reported uncertainty.",
        "",
        "## Central Table",
        "",
        *_markdown_table(paper2_summary, [column for column in columns if column in paper2_summary.columns]),
        "",
        "## Compact Artifacts",
        "",
        f"- Central table: `{_display_path(paper2_summary_csv, out_dir)}`",
        f"- Summary figure: `{_display_path(figure_path, out_dir)}`",
        f"- Temporal model rows: `{_display_path(temporal_all_csv, out_dir)}`",
        f"- Emission comparison rows: `{_display_path(emission_all_csv, out_dir)}`",
        f"- Semantic-stage time rows: `{_display_path(stage_time_all_csv, out_dir)}`",
        f"- Semantic-stage intervals: `{_display_path(stages_all_csv, out_dir)}`",
        "",
        "## Reading Rule",
        "",
        "The primary Paper 2 metric is `delta_control_margin`: calibrated observed effect-window persistence gain minus the strongest calibrated control, compared with the same uncalibrated margin. Positive values favor calibrated emissions. Semantic-stage rows are supporting evidence and should be interpreted only together with the shuffled-time, shuffled-label, and baseline-window controls.",
        "",
    ]
    return "\n".join(lines)


def _write_command_log(
    out_path: Path,
    *,
    command_line: str,
    task_outputs: list[Paper2TaskOutput],
    decoders: tuple[str, ...],
    n_permutations: int,
    stay_grid_size: int,
) -> None:
    lines = [
        "# Paper 2 RepTrace Commands",
        "",
        "Top-level command:",
        "",
        "```bash",
        command_line,
        "```",
        "",
        f"Decoders: {', '.join(decoders)}",
        "Emission modes: calibrated and uncalibrated, generated in the same benchmark call so folds, subjects, tasks, and time windows are identical.",
        "",
    ]
    for task_output in task_outputs:
        task_dir = task_output.task_dir
        lines.extend(
            [
                f"## {task_output.task.label}",
                "",
                "```bash",
                f"python -m reptrace.benchmark {task_output.manifest_csv} --out-dir {task_dir} --aggregate-out {task_dir / 'summary.csv'} --plot-out {task_dir / 'summary.png'} --calibration-dir {task_dir / 'calibration'} --observation-dir {task_dir / 'observations'} --emission-mode both --chance 0.5 --resume",
                f"python -m reptrace.temporal_model \"{task_dir / 'observations' / '*_observations.csv'}\" --out-summary {task_output.temporal_summary_csv} --out-states {task_output.state_trace_csv} --n-permutations {n_permutations} --stay-grid-size {stay_grid_size}",
                f"python -m reptrace.emission_compare {task_output.temporal_summary_csv} --out-csv {task_output.emission_compare_csv} --out-report {task_output.emission_compare_report}",
                f"python -m reptrace.semantic_stages {task_output.state_trace_csv} --out-time {task_output.semantic_time_csv} --out-stages {task_output.semantic_stages_csv} --out-report {task_output.semantic_stages_report}",
                "```",
                "",
            ]
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def _collect_compact_artifacts(source_dir: Path, patterns: tuple[str, ...] = PAPER2_COMPACT_PATTERNS) -> list[Path]:
    artifacts: list[Path] = []
    for pattern in patterns:
        artifacts.extend(path for path in source_dir.glob(pattern) if path.is_file())
    return sorted(set(artifacts))


def export_paper2_artifacts(
    source_dir: Path,
    destination_dir: Path,
    *,
    max_mb: float = 50.0,
    dry_run: bool = False,
) -> list[Path]:
    """Copy compact Paper 2 artifacts to the paper repository."""
    source_dir = source_dir.resolve()
    destination_dir = destination_dir.resolve()
    artifacts = _collect_compact_artifacts(source_dir)
    if not artifacts:
        raise FileNotFoundError(f"No compact Paper 2 artifacts found in {source_dir}.")

    size_mb = sum(path.stat().st_size for path in artifacts) / (1024 * 1024)
    if size_mb > max_mb:
        raise ValueError(f"Compact Paper 2 artifacts are {size_mb:.2f} MB, above limit {max_mb:.2f} MB.")
    if dry_run:
        return artifacts

    copied: list[Path] = []
    for artifact in artifacts:
        relative = artifact.relative_to(source_dir)
        target = destination_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(artifact, target)
        copied.append(target)
    return copied


def run_paper2_workflow(
    *,
    out_dir: Path,
    data_root: Path | None = None,
    paper_export_dir: Path | None = None,
    task_ids: tuple[str, ...] | None = None,
    decoders: tuple[str, ...] = DEFAULT_DECODERS,
    n_permutations: int = 100,
    random_seed: int = 13,
    stay_grid_size: int = 200,
    posterior_threshold: float = 0.6,
    match_threshold: float = 0.6,
    min_duration: float = 0.04,
    max_subjects: int | None = None,
    expected_subjects: int | None = 19,
    resume: bool = True,
    max_export_mb: float = 50.0,
    command_line: str = "python -m reptrace.paper2_workflow",
) -> Paper2WorkflowRun:
    """Run the reproducible Paper 2 NOD temporal-state workflow."""
    repo_root = _repo_root()
    out_dir = out_dir.resolve()
    tasks = _selected_tasks(task_ids)
    decoders = _normal_decoders(decoders)
    task_outputs: list[Paper2TaskOutput] = []

    for task in tasks:
        task_dir = out_dir / task.task_id
        manifest_csv = task_dir / "manifest.csv"
        validation_csv = task_dir / "validation.csv"
        source_manifest = repo_root / "benchmarks" / task.manifest_name
        prepare_paper2_manifest(
            source_manifest,
            manifest_csv,
            data_root=data_root,
            decoders=decoders,
            max_subjects=max_subjects,
            expected_subjects=expected_subjects,
        )
        _validate_prepared_manifest(manifest_csv, validation_csv)
        run_benchmark_manifest(
            manifest_csv,
            out_dir=task_dir,
            aggregate_out=task_dir / "summary.csv",
            plot_out=task_dir / "summary.png",
            chance=0.5,
            default_emission_mode="both",
            calibration_dir=task_dir / "calibration",
            observation_dir=task_dir / "observations",
            resume=resume,
        )

        observation_paths = _task_observation_paths(task_dir)
        if not observation_paths:
            raise FileNotFoundError(f"No observation CSVs were produced for {task.task_id}.")

        temporal_summary_csv = task_dir / "temporal_model.csv"
        state_trace_csv = task_dir / "state_trace.csv"
        fit_temporal_models(
            observation_paths,
            n_permutations=n_permutations,
            random_seed=random_seed,
            stay_grid_size=stay_grid_size,
            out_summary=temporal_summary_csv,
            out_states=state_trace_csv,
        )
        emission_compare_csv = task_dir / "emission_compare.csv"
        emission_compare_report = task_dir / "emission_compare.md"
        compare_temporal_summary(
            temporal_summary_csv,
            out_csv=emission_compare_csv,
            out_report=emission_compare_report,
        )
        semantic_time_csv = task_dir / "semantic_stage_time.csv"
        semantic_stages_csv = task_dir / "semantic_stages.csv"
        semantic_stages_report = task_dir / "semantic_stages.md"
        analyze_semantic_stages(
            [state_trace_csv],
            posterior_threshold=posterior_threshold,
            match_threshold=match_threshold,
            min_duration=min_duration,
            out_time=semantic_time_csv,
            out_stages=semantic_stages_csv,
            out_report=semantic_stages_report,
        )

        try:
            plot_time_decode_results(task_dir / "summary.csv", task_dir / "summary.png", chance=0.5, title=task.label)
        except ValueError:
            pass

        task_outputs.append(
            Paper2TaskOutput(
                task=task,
                task_dir=task_dir,
                manifest_csv=manifest_csv,
                validation_csv=validation_csv,
                temporal_summary_csv=temporal_summary_csv,
                state_trace_csv=state_trace_csv,
                emission_compare_csv=emission_compare_csv,
                emission_compare_report=emission_compare_report,
                semantic_time_csv=semantic_time_csv,
                semantic_stages_csv=semantic_stages_csv,
                semantic_stages_report=semantic_stages_report,
            )
        )

    temporal_all_csv = out_dir / "temporal_model_all.csv"
    emission_all_csv = out_dir / "emission_compare_all.csv"
    stage_time_all_csv = out_dir / "semantic_stage_time_all.csv"
    stages_all_csv = out_dir / "semantic_stages_all.csv"
    _write_combined_csv(task_outputs, "temporal_summary_csv", temporal_all_csv)
    emission_compare = _write_combined_csv(task_outputs, "emission_compare_csv", emission_all_csv)
    stage_time = _write_combined_csv(task_outputs, "semantic_time_csv", stage_time_all_csv)
    stages = _write_combined_csv(task_outputs, "semantic_stages_csv", stages_all_csv)

    paper2_summary = build_paper2_summary(emission_compare, stages, stage_time)
    paper2_summary_csv = out_dir / "paper2_summary.csv"
    paper2_summary.to_csv(paper2_summary_csv, index=False)
    paper2_figure = out_dir / "paper2_stage_reliability.png"
    plot_paper2_stage_reliability(paper2_summary, stage_time, paper2_figure)
    paper2_report = out_dir / "paper2_evidence.md"
    paper2_report.write_text(
        build_evidence_report(
            paper2_summary,
            out_dir=out_dir,
            paper2_summary_csv=paper2_summary_csv,
            figure_path=paper2_figure,
            temporal_all_csv=temporal_all_csv,
            emission_all_csv=emission_all_csv,
            stage_time_all_csv=stage_time_all_csv,
            stages_all_csv=stages_all_csv,
        ),
        encoding="utf-8",
    )
    command_log = out_dir / "paper2_commands.md"
    _write_command_log(
        command_log,
        command_line=command_line,
        task_outputs=task_outputs,
        decoders=decoders,
        n_permutations=n_permutations,
        stay_grid_size=stay_grid_size,
    )

    exported_artifacts: list[Path] = []
    if paper_export_dir is not None:
        exported_artifacts = export_paper2_artifacts(out_dir, paper_export_dir, max_mb=max_export_mb)

    return Paper2WorkflowRun(
        out_dir=out_dir,
        task_outputs=task_outputs,
        paper2_summary_csv=paper2_summary_csv,
        paper2_figure=paper2_figure,
        paper2_report=paper2_report,
        command_log=command_log,
        exported_artifacts=exported_artifacts,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the RepTrace Paper 2 calibration-aware temporal-state workflow.")
    parser.add_argument("--out-dir", type=Path, default=Path("results/paper2_temporal_state_inference"))
    parser.add_argument("--data-root", type=Path, help="Directory containing staged NOD sub-*_epo.fif and sub-*_events.csv files.")
    parser.add_argument("--paper-export-dir", type=Path, help="Optional paper-repo result directory for compact artifacts.")
    parser.add_argument("--task", action="append", choices=DEFAULT_TASK_IDS, dest="task_ids", help="Task to run. Repeat to select multiple tasks.")
    parser.add_argument("--decoders", nargs="+", choices=DECODER_CHOICES, default=list(DEFAULT_DECODERS))
    parser.add_argument("--n-permutations", type=int, default=100)
    parser.add_argument("--random-seed", type=int, default=13)
    parser.add_argument("--stay-grid-size", type=int, default=200)
    parser.add_argument("--posterior-threshold", type=float, default=0.6)
    parser.add_argument("--match-threshold", type=float, default=0.6)
    parser.add_argument("--min-duration", type=float, default=0.04)
    parser.add_argument("--max-subjects", type=int, help="Smoke-test limit on the number of subjects per task.")
    parser.add_argument("--expected-subjects", type=int, default=19)
    parser.add_argument("--no-resume", action="store_true", help="Rerun subject-decoder outputs even when existing complete files are present.")
    parser.add_argument("--max-export-mb", type=float, default=50.0)
    args = parser.parse_args()

    command_line = "python -m reptrace.paper2_workflow " + " ".join(sys.argv[1:])
    run = run_paper2_workflow(
        out_dir=args.out_dir,
        data_root=args.data_root,
        paper_export_dir=args.paper_export_dir,
        task_ids=tuple(args.task_ids) if args.task_ids else None,
        decoders=tuple(args.decoders),
        n_permutations=args.n_permutations,
        random_seed=args.random_seed,
        stay_grid_size=args.stay_grid_size,
        posterior_threshold=args.posterior_threshold,
        match_threshold=args.match_threshold,
        min_duration=args.min_duration,
        max_subjects=args.max_subjects,
        expected_subjects=args.expected_subjects,
        resume=not args.no_resume,
        max_export_mb=args.max_export_mb,
        command_line=command_line,
    )
    print(f"Wrote Paper 2 summary: {run.paper2_summary_csv}")
    print(f"Wrote Paper 2 figure: {run.paper2_figure}")
    print(f"Wrote Paper 2 evidence note: {run.paper2_report}")
    if run.exported_artifacts:
        size_mb = sum(path.stat().st_size for path in run.exported_artifacts) / (1024 * 1024)
        print(f"Exported {len(run.exported_artifacts)} compact artifact(s), {size_mb:.3f} MB.")


if __name__ == "__main__":
    main()
