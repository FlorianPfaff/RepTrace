from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd

STAGE_GROUP_COLUMNS = ("decoder", "emission_mode")


def _expand_paths(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if matches:
            paths.extend(Path(match) for match in matches)
        else:
            paths.append(Path(pattern))
    return paths


def posterior_columns(frame: pd.DataFrame) -> list[str]:
    """Return posterior state columns in state-index order."""
    columns = [column for column in frame.columns if column.startswith("posterior_state_")]
    if not columns:
        raise ValueError("State trace CSVs must contain columns named 'posterior_state_*'.")

    def sort_key(column: str) -> tuple[int, str]:
        suffix = column.removeprefix("posterior_state_")
        return (int(suffix), suffix) if suffix.isdigit() else (10_000, suffix)

    return sorted(columns, key=sort_key)


def _state_names(frame: pd.DataFrame, columns: list[str]) -> list[str]:
    names = []
    for index, column in enumerate(columns):
        state_column = f"state_{column.removeprefix('posterior_state_')}"
        if state_column in frame.columns and frame[state_column].notna().any():
            names.append(str(frame[state_column].dropna().iloc[0]))
        else:
            names.append(str(index))
    return names


def read_state_traces(csv_paths: list[Path]) -> pd.DataFrame:
    """Read state traces emitted by ``reptrace.temporal_model``."""
    if not csv_paths:
        raise ValueError("At least one state trace CSV path is required.")

    frames = []
    for csv_path in csv_paths:
        frame = pd.read_csv(csv_path)
        missing = [column for column in ("time", "viterbi_class") if column not in frame.columns]
        if missing:
            raise ValueError(f"{csv_path} is missing required columns: {missing}")
        posterior_columns(frame)
        if "subject" not in frame.columns:
            frame["subject"] = csv_path.stem
        if "decoder" not in frame.columns:
            frame["decoder"] = "decoder"
        if "emission_mode" not in frame.columns:
            frame["emission_mode"] = "calibrated"
        if "sequence_id" not in frame.columns:
            if "sample_index" in frame.columns:
                frame["sequence_id"] = frame["sample_index"]
            else:
                frame["sequence_id"] = np.arange(len(frame))
        frame["subject"] = frame["subject"].astype(str)
        frame["decoder"] = frame["decoder"].astype(str)
        frame["emission_mode"] = frame["emission_mode"].astype(str)
        frame["source_file"] = csv_path.name
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _sequence_keys(frame: pd.DataFrame) -> pd.Series:
    key_columns = [column for column in ("subject", "fold", "sequence_id") if column in frame.columns]
    return frame[key_columns].astype(str).agg("|".join, axis=1)


def _stage_group_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in STAGE_GROUP_COLUMNS if column in frame.columns]


def _add_true_class_alignment(frame: pd.DataFrame, columns: list[str], state_names: list[str]) -> pd.DataFrame:
    aligned = frame.copy()
    aligned["sequence_key"] = _sequence_keys(aligned)
    aligned["viterbi_matches_true_class"] = aligned["viterbi_class"].astype(str) == aligned["true_class"].astype(str)
    aligned["posterior_true_class"] = np.nan

    for state_name, posterior_column in zip(state_names, columns, strict=True):
        mask = aligned["true_class"].astype(str) == state_name
        aligned.loc[mask, "posterior_true_class"] = aligned.loc[mask, posterior_column]

    return aligned


def _time_sem(series: pd.Series) -> float:
    if series.notna().sum() < 2:
        return 0.0
    return float(series.sem())


def summarize_category_timecourse(state_traces: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Summarize category-conditioned state stability over time."""
    columns = posterior_columns(state_traces)
    state_names = _state_names(state_traces, columns)
    if "true_class" not in state_traces.columns:
        return summarize_dominant_timecourse(state_traces), state_names

    aligned = _add_true_class_alignment(state_traces, columns, state_names)
    summary = (
        aligned.groupby([*_stage_group_columns(aligned), "true_class", "time"], as_index=False)
        .agg(
            n_observations=("time", "size"),
            n_subjects=("subject", "nunique"),
            n_sequences=("sequence_key", "nunique"),
            posterior_true_class_mean=("posterior_true_class", "mean"),
            posterior_true_class_sem=("posterior_true_class", _time_sem),
            viterbi_match_rate=("viterbi_matches_true_class", "mean"),
            viterbi_posterior_mean=("viterbi_posterior", "mean") if "viterbi_posterior" in aligned.columns else ("posterior_true_class", "mean"),
        )
        .sort_values([*_stage_group_columns(aligned), "true_class", "time"])
        .reset_index(drop=True)
    )
    return summary, state_names


def summarize_dominant_timecourse(state_traces: pd.DataFrame) -> pd.DataFrame:
    """Summarize dominant latent-state stability when true category labels are absent."""
    columns = posterior_columns(state_traces)
    state_names = _state_names(state_traces, columns)
    frame = state_traces.copy()
    frame["sequence_key"] = _sequence_keys(frame)
    rows = []
    group_columns = _stage_group_columns(frame)
    for keys, group in frame.groupby([*group_columns, "time"], sort=True):
        key_values = keys if isinstance(keys, tuple) else (keys,)
        group_values = dict(zip([*group_columns, "time"], key_values, strict=True))
        means = group[columns].mean().to_numpy(dtype=float)
        order = np.argsort(means)
        dominant_index = int(order[-1])
        runner_up = float(means[order[-2]]) if len(order) > 1 else float("nan")
        dominant_class = state_names[dominant_index]
        rows.append(
            {
                "decoder": group_values.get("decoder", "decoder"),
                "emission_mode": group_values.get("emission_mode", "calibrated"),
                "true_class": dominant_class,
                "time": float(group_values["time"]),
                "n_observations": len(group),
                "n_subjects": group["subject"].nunique(),
                "n_sequences": group["sequence_key"].nunique(),
                "posterior_true_class_mean": float(means[dominant_index]),
                "posterior_true_class_sem": 0.0,
                "viterbi_match_rate": float((group["viterbi_class"].astype(str) == dominant_class).mean()),
                "viterbi_posterior_mean": float(group["viterbi_posterior"].mean()) if "viterbi_posterior" in group.columns else float(means[dominant_index]),
                "posterior_margin": float(means[dominant_index] - runner_up) if len(order) > 1 else float("nan"),
            }
        )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    return result.sort_values([*_stage_group_columns(result), "true_class", "time"]).reset_index(drop=True)


def _contiguous_segments(mask: np.ndarray) -> list[tuple[int, int]]:
    segments = []
    start = None
    for index, value in enumerate(mask):
        if value and start is None:
            start = index
        elif not value and start is not None:
            segments.append((start, index - 1))
            start = None
    if start is not None:
        segments.append((start, len(mask) - 1))
    return segments


def detect_stable_stages(
    time_summary: pd.DataFrame,
    *,
    posterior_threshold: float = 0.6,
    match_threshold: float = 0.6,
    min_duration: float = 0.04,
) -> pd.DataFrame:
    """Detect contiguous semantic stages from a category-conditioned time summary."""
    rows = []
    group_columns = [*_stage_group_columns(time_summary), "true_class"]
    for keys, group in time_summary.sort_values("time").groupby(group_columns, sort=True):
        key_values = keys if isinstance(keys, tuple) else (keys,)
        group_values = dict(zip(group_columns, key_values, strict=True))
        stable = (
            (group["posterior_true_class_mean"].to_numpy(dtype=float) >= posterior_threshold)
            & (group["viterbi_match_rate"].to_numpy(dtype=float) >= match_threshold)
        )
        times = group["time"].to_numpy(dtype=float)
        for start_index, stop_index in _contiguous_segments(stable):
            start_time = float(times[start_index])
            stop_time = float(times[stop_index])
            duration = stop_time - start_time
            if duration < min_duration:
                continue
            segment = group.iloc[start_index : stop_index + 1]
            peak_row = segment.loc[segment["posterior_true_class_mean"].idxmax()]
            rows.append(
                {
                    "decoder": group_values.get("decoder", "decoder"),
                    "emission_mode": group_values.get("emission_mode", "calibrated"),
                    "semantic_class": group_values["true_class"],
                    "start_time": start_time,
                    "stop_time": stop_time,
                    "duration": duration,
                    "n_timepoints": len(segment),
                    "mean_posterior_true_class": float(segment["posterior_true_class_mean"].mean()),
                    "mean_viterbi_match_rate": float(segment["viterbi_match_rate"].mean()),
                    "peak_time": float(peak_row["time"]),
                    "peak_posterior_true_class": float(peak_row["posterior_true_class_mean"]),
                    "n_subjects_min": int(segment["n_subjects"].min()) if "n_subjects" in segment.columns else 0,
                    "n_sequences_min": int(segment["n_sequences"].min()),
                }
            )
    return pd.DataFrame(rows)


def build_stage_report(
    time_summary: pd.DataFrame,
    stages: pd.DataFrame,
    *,
    posterior_threshold: float,
    match_threshold: float,
    min_duration: float,
) -> str:
    """Build a compact Markdown report for the semantic-stage question."""
    lines = [
        "# RepTrace Semantic Stage Report",
        "",
        "Question: do semantic categories unfold in stable temporal stages?",
        "",
        "A stable stage is a contiguous time range where the posterior assigned to",
        "the trial's semantic class and the Viterbi match rate both exceed the",
        "configured thresholds.",
        "",
        "## Thresholds",
        "",
        f"- Posterior threshold: {posterior_threshold:.3f}",
        f"- Viterbi match threshold: {match_threshold:.3f}",
        f"- Minimum duration: {min_duration:.3f} s",
        "",
        "## Stable Stages",
        "",
    ]

    if stages.empty:
        lines.append("No stable semantic stages were detected at these thresholds.")
    else:
        lines.extend(
            [
                "| Decoder | Emission mode | Semantic class | Start (s) | Stop (s) | Duration (s) | Mean posterior | Mean match | Peak time (s) | Subject min |",
                "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in stages.itertuples(index=False):
            n_subjects_min = row.n_subjects_min if hasattr(row, "n_subjects_min") else 0
            lines.append(
                f"| {row.decoder} | {row.emission_mode} | {row.semantic_class} | {row.start_time:.3f} | {row.stop_time:.3f} | "
                f"{row.duration:.3f} | {row.mean_posterior_true_class:.3f} | {row.mean_viterbi_match_rate:.3f} | {row.peak_time:.3f} | {n_subjects_min} |"
            )

    if not time_summary.empty:
        peaks = (
            time_summary.loc[time_summary.groupby([*_stage_group_columns(time_summary), "true_class"])["posterior_true_class_mean"].idxmax()]
            .sort_values([*_stage_group_columns(time_summary), "true_class"])
            .reset_index(drop=True)
        )
        lines.extend(
            [
                "",
                "## Category Peaks",
                "",
                "| Decoder | Emission mode | Semantic class | Peak time (s) | Peak posterior | Match rate |",
                "| --- | --- | --- | ---: | ---: | ---: |",
            ]
        )
        for row in peaks.itertuples(index=False):
            lines.append(f"| {row.decoder} | {row.emission_mode} | {row.true_class} | {row.time:.3f} | {row.posterior_true_class_mean:.3f} | {row.viterbi_match_rate:.3f} |")

    lines.extend(
        [
            "",
            "Interpretation should be paired with the temporal-model controls. A",
            "stage is strongest when it survives these descriptive thresholds and",
            "the parent temporal model beats shuffled-time, shuffled-label, and",
            "baseline-window controls.",
            "",
        ]
    )
    return "\n".join(lines)


def analyze_semantic_stages(
    state_trace_csvs: list[Path],
    *,
    posterior_threshold: float = 0.6,
    match_threshold: float = 0.6,
    min_duration: float = 0.04,
    out_time: Path | None = None,
    out_stages: Path | None = None,
    out_report: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, str | None]:
    """Analyze whether category-conditioned state traces form stable temporal stages."""
    state_traces = read_state_traces(state_trace_csvs)
    time_summary, _ = summarize_category_timecourse(state_traces)
    stages = detect_stable_stages(
        time_summary,
        posterior_threshold=posterior_threshold,
        match_threshold=match_threshold,
        min_duration=min_duration,
    )

    if out_time is not None:
        out_time.parent.mkdir(parents=True, exist_ok=True)
        time_summary.to_csv(out_time, index=False)
    if out_stages is not None:
        out_stages.parent.mkdir(parents=True, exist_ok=True)
        stages.to_csv(out_stages, index=False)

    report = None
    if out_report is not None:
        report = build_stage_report(
            time_summary,
            stages,
            posterior_threshold=posterior_threshold,
            match_threshold=match_threshold,
            min_duration=min_duration,
        )
        out_report.parent.mkdir(parents=True, exist_ok=True)
        out_report.write_text(report, encoding="utf-8")

    return time_summary, stages, report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ask whether decoded semantic states unfold in stable temporal stages."
    )
    parser.add_argument("state_trace_csv", nargs="+", help="State trace CSVs or glob patterns emitted by reptrace.temporal_model.")
    parser.add_argument("--out-time", type=Path, required=True)
    parser.add_argument("--out-stages", type=Path, required=True)
    parser.add_argument("--out-report", type=Path)
    parser.add_argument("--posterior-threshold", type=float, default=0.6)
    parser.add_argument("--match-threshold", type=float, default=0.6)
    parser.add_argument("--min-duration", type=float, default=0.04)
    args = parser.parse_args()

    paths = _expand_paths(args.state_trace_csv)
    _, stages, _ = analyze_semantic_stages(
        paths,
        posterior_threshold=args.posterior_threshold,
        match_threshold=args.match_threshold,
        min_duration=args.min_duration,
        out_time=args.out_time,
        out_stages=args.out_stages,
        out_report=args.out_report,
    )
    print(f"Wrote semantic-stage time summary: {args.out_time}")
    print(f"Wrote semantic-stage intervals: {args.out_stages}")
    if args.out_report is not None:
        print(f"Wrote semantic-stage report: {args.out_report}")
    print(stages.to_string(index=False) if not stages.empty else "No stable semantic stages detected.")


if __name__ == "__main__":
    main()
