from __future__ import annotations

import argparse
import glob
from pathlib import Path

import pandas as pd

from reptrace.results import read_time_decode_results

COMPARISON_GROUP_COLUMNS = ("decoder", "emission_mode")


def _window_mean(frame: pd.DataFrame, column: str, start: float, stop: float) -> float:
    window = frame[(frame["time"] >= start) & (frame["time"] <= stop)]
    if window.empty:
        raise ValueError(f"No time points found in window [{start}, {stop}].")
    return float(window[column].mean())


def _format_float(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def _comparison_group_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in COMPARISON_GROUP_COLUMNS if column in frame.columns]


def summarize_aggregate_time_decode(
    summary: pd.DataFrame,
    *,
    chance: float = 0.5,
    baseline_window: tuple[float, float] = (-0.1, 0.0),
    effect_window: tuple[float, float] = (0.1, 0.8),
) -> dict[str, float]:
    """Summarize an aggregate time-resolved decoding CSV."""
    required = {"time", "accuracy_mean", "log_loss_mean", "brier_mean", "ece_mean", "n_subjects"}
    missing = sorted(required.difference(summary.columns))
    if missing:
        raise ValueError(f"Summary is missing required columns: {missing}")

    peak = summary.loc[summary["accuracy_mean"].idxmax()]
    effect_mean = _window_mean(summary, "accuracy_mean", *effect_window)
    baseline_mean = _window_mean(summary, "accuracy_mean", *baseline_window)

    return {
        "n_subjects": float(peak["n_subjects"]),
        "chance": chance,
        "peak_time": float(peak["time"]),
        "peak_accuracy": float(peak["accuracy_mean"]),
        "peak_accuracy_sem": float(peak["accuracy_sem"]) if "accuracy_sem" in peak else float("nan"),
        "peak_log_loss": float(peak["log_loss_mean"]),
        "peak_brier": float(peak["brier_mean"]),
        "peak_ece": float(peak["ece_mean"]),
        "baseline_accuracy_mean": baseline_mean,
        "effect_accuracy_mean": effect_mean,
        "effect_accuracy_delta": effect_mean - chance,
    }


def summarize_subject_time_decode(
    csv_paths: list[Path],
    *,
    effect_window: tuple[float, float] = (0.1, 0.8),
) -> pd.DataFrame:
    """Summarize subject-level peak and effect-window decoding accuracy."""
    results = read_time_decode_results(csv_paths)
    by_subject_time = (
        results.groupby(["subject", "time"], as_index=False)["accuracy"]
        .mean()
        .sort_values(["subject", "time"])
    )

    rows = []
    for subject, subject_frame in by_subject_time.groupby("subject", sort=True):
        peak = subject_frame.loc[subject_frame["accuracy"].idxmax()]
        rows.append(
            {
                "subject": subject,
                "peak_time": float(peak["time"]),
                "peak_accuracy": float(peak["accuracy"]),
                "effect_accuracy_mean": _window_mean(subject_frame, "accuracy", *effect_window),
            }
        )

    return pd.DataFrame(rows)


def summarize_decoder_comparison(
    summary: pd.DataFrame,
    *,
    baseline_window: tuple[float, float] = (-0.1, 0.0),
    effect_window: tuple[float, float] = (0.1, 0.8),
) -> pd.DataFrame:
    """Summarize aggregate benchmark metrics separately for each decoder."""
    group_columns = _comparison_group_columns(summary)
    if "decoder" not in group_columns:
        raise ValueError("Summary must contain a 'decoder' column for decoder comparison.")

    rows = []
    for keys, decoder_frame in summary.groupby(group_columns, sort=True):
        key_values = keys if isinstance(keys, tuple) else (keys,)
        group_values = dict(zip(group_columns, key_values, strict=True))
        peak = decoder_frame.loc[decoder_frame["accuracy_mean"].idxmax()]
        baseline_accuracy = _window_mean(decoder_frame, "accuracy_mean", *baseline_window)
        effect_accuracy = _window_mean(decoder_frame, "accuracy_mean", *effect_window)
        rows.append(
            {
                **group_values,
                "n_subjects": int(peak["n_subjects"]),
                "peak_time": float(peak["time"]),
                "peak_accuracy": float(peak["accuracy_mean"]),
                "baseline_accuracy_mean": baseline_accuracy,
                "effect_accuracy_mean": effect_accuracy,
                "effect_minus_baseline": effect_accuracy - baseline_accuracy,
                "peak_log_loss": float(peak["log_loss_mean"]),
                "peak_brier": float(peak["brier_mean"]),
                "peak_ece": float(peak["ece_mean"]),
            }
        )
    return pd.DataFrame(rows).sort_values("effect_minus_baseline", ascending=False).reset_index(drop=True)


def build_time_decode_report(
    summary_csv: Path,
    *,
    subject_csvs: list[Path] | None = None,
    chance: float = 0.5,
    baseline_window: tuple[float, float] = (-0.1, 0.0),
    effect_window: tuple[float, float] = (0.1, 0.8),
) -> str:
    """Build a Markdown report for a time-resolved decoding benchmark."""
    summary = pd.read_csv(summary_csv)
    comparison_columns = _comparison_group_columns(summary)
    has_comparison = "decoder" in summary.columns and summary.groupby(comparison_columns).ngroups > 1
    if has_comparison:
        comparison = summarize_decoder_comparison(
            summary,
            baseline_window=baseline_window,
            effect_window=effect_window,
        )
        has_emission_mode = "emission_mode" in comparison.columns
        lines = [
            "# RepTrace Decoder Comparison Report",
            "",
            f"- Summary CSV: `{summary_csv}`",
            f"- Baseline window: {_format_float(baseline_window[0])} to {_format_float(baseline_window[1])} s",
            f"- Effect window: {_format_float(effect_window[0])} to {_format_float(effect_window[1])} s",
            "",
        ]
        if has_emission_mode:
            lines.extend(
                [
                    "| Decoder | Emission mode | Subjects | Peak time (s) | Peak accuracy | Baseline accuracy | Effect accuracy | Effect minus baseline | Peak log loss | Peak Brier | Peak ECE |",
                    "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
                ]
            )
        else:
            lines.extend(
                [
                    "| Decoder | Subjects | Peak time (s) | Peak accuracy | Baseline accuracy | Effect accuracy | Effect minus baseline | Peak log loss | Peak Brier | Peak ECE |",
                    "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
                ]
            )
        for row in comparison.itertuples(index=False):
            emission_prefix = f"| {row.decoder} | {row.emission_mode} |" if has_emission_mode else f"| {row.decoder} |"
            lines.append(
                f"{emission_prefix} {row.n_subjects} | {_format_float(row.peak_time)} | {_format_float(row.peak_accuracy)} | "
                f"{_format_float(row.baseline_accuracy_mean)} | {_format_float(row.effect_accuracy_mean)} | {_format_float(row.effect_minus_baseline)} | "
                f"{_format_float(row.peak_log_loss)} | {_format_float(row.peak_brier)} | {_format_float(row.peak_ece)} |"
            )
        lines.append("")
        return "\n".join(lines)

    aggregate = summarize_aggregate_time_decode(
        summary,
        chance=chance,
        baseline_window=baseline_window,
        effect_window=effect_window,
    )

    lines = [
        "# RepTrace Time-Decoding Report",
        "",
        f"- Summary CSV: `{summary_csv}`",
        f"- Subjects: {int(aggregate['n_subjects'])}",
        f"- Chance level: {_format_float(aggregate['chance'])}",
        f"- Baseline window: {_format_float(baseline_window[0])} to {_format_float(baseline_window[1])} s",
        f"- Effect window: {_format_float(effect_window[0])} to {_format_float(effect_window[1])} s",
        "",
        "## Aggregate",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Peak aggregate accuracy | {_format_float(aggregate['peak_accuracy'])} |",
        f"| Peak time | {_format_float(aggregate['peak_time'])} s |",
        f"| Accuracy SEM at peak | {_format_float(aggregate['peak_accuracy_sem'])} |",
        f"| Baseline-window mean accuracy | {_format_float(aggregate['baseline_accuracy_mean'])} |",
        f"| Effect-window mean accuracy | {_format_float(aggregate['effect_accuracy_mean'])} |",
        f"| Effect-window delta from chance | {_format_float(aggregate['effect_accuracy_delta'])} |",
        f"| Log loss at peak | {_format_float(aggregate['peak_log_loss'])} |",
        f"| Brier score at peak | {_format_float(aggregate['peak_brier'])} |",
        f"| ECE at peak | {_format_float(aggregate['peak_ece'])} |",
    ]

    if subject_csvs:
        subject_summary = summarize_subject_time_decode(subject_csvs, effect_window=effect_window)
        lines.extend(
            [
                "",
                "## Subjects",
                "",
                "| Subject | Peak time (s) | Peak accuracy | Effect-window mean accuracy |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        for row in subject_summary.itertuples(index=False):
            lines.append(
                f"| {row.subject} | {_format_float(row.peak_time)} | {_format_float(row.peak_accuracy)} | {_format_float(row.effect_accuracy_mean)} |"
            )

    lines.append("")
    return "\n".join(lines)


def _expand_paths(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if matches:
            paths.extend(Path(match) for match in matches)
        else:
            paths.append(Path(pattern))
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a compact Markdown report from time-resolved decoding results."
    )
    parser.add_argument("summary_csv", type=Path)
    parser.add_argument("subject_csv", nargs="*")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--chance", type=float, default=0.5)
    parser.add_argument("--baseline-window", type=float, nargs=2, default=(-0.1, 0.0), metavar=("START", "STOP"))
    parser.add_argument("--effect-window", type=float, nargs=2, default=(0.1, 0.8), metavar=("START", "STOP"))
    args = parser.parse_args()

    report = build_time_decode_report(
        args.summary_csv,
        subject_csvs=_expand_paths(args.subject_csv),
        chance=args.chance,
        baseline_window=tuple(args.baseline_window),
        effect_window=tuple(args.effect_window),
    )
    if args.out is None:
        print(report)
        return

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report, encoding="utf-8")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
