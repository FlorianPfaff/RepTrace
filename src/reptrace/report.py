from __future__ import annotations

import argparse
import glob
from pathlib import Path

import pandas as pd

from reptrace.results import METRIC_COLUMNS, SUMMARY_GROUP_COLUMNS, mean_across_folds, read_time_decode_results

COMPARISON_GROUP_COLUMNS = SUMMARY_GROUP_COLUMNS
SUBJECT_TABLE_LABELS = {
    "decoder": "Decoder",
    "emission_mode": "Emission mode",
    "feature_preprocessor": "Feature preprocessor",
    "pca_components": "PCA components",
    "tuned_hyperparameters": "Tuned",
    "tuning_cv_splits": "Tuning CV splits",
    "tuning_scoring": "Tuning scoring",
    "tuning_c_grid": "Tuning C grid",
    "temporal_mode": "Temporal mode",
    "temporal_train_window_start": "Temporal train start",
    "temporal_train_window_stop": "Temporal train stop",
}
METRIC_HIGHER_IS_BETTER = {
    "accuracy": True,
    "log_loss": False,
    "brier": False,
    "ece": False,
}
METRIC_LABELS = {
    "accuracy": "accuracy",
    "log_loss": "log loss",
    "brier": "Brier score",
    "ece": "ECE",
}


def _window_mean(frame: pd.DataFrame, column: str, start: float, stop: float) -> float:
    window = frame[(frame["time"] >= start) & (frame["time"] <= stop)]
    if window.empty:
        raise ValueError(f"No time points found in window [{start}, {stop}].")
    return float(window[column].mean())


def _format_float(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def _comparison_group_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in COMPARISON_GROUP_COLUMNS if column in frame.columns]


def _raw_csv_has_column(csv_paths: list[Path], column: str) -> bool:
    return any(column in pd.read_csv(csv_path, nrows=0).columns for csv_path in csv_paths)


def _iter_groups(frame: pd.DataFrame, group_columns: list[str]):
    grouper = group_columns[0] if len(group_columns) == 1 else group_columns
    yield from frame.groupby(grouper, sort=True)


def _validate_selection_metric(selection_metric: str) -> str:
    if selection_metric not in METRIC_HIGHER_IS_BETTER:
        allowed = ", ".join(METRIC_HIGHER_IS_BETTER)
        raise ValueError(f"selection_metric must be one of: {allowed}")
    return selection_metric


def _metric_mean_column(selection_metric: str) -> str:
    return f"{_validate_selection_metric(selection_metric)}_mean"


def _metric_label(selection_metric: str) -> str:
    return METRIC_LABELS[_validate_selection_metric(selection_metric)]


def _metric_direction_label(selection_metric: str) -> str:
    selection_metric = _validate_selection_metric(selection_metric)
    return "higher is better" if METRIC_HIGHER_IS_BETTER[selection_metric] else "lower is better"


def _metric_descriptor(selection_metric: str) -> str:
    return f"{_metric_label(selection_metric)} ({_metric_direction_label(selection_metric)})"


def _metric_improvement(effect_mean: float, baseline_mean: float, selection_metric: str) -> float:
    selection_metric = _validate_selection_metric(selection_metric)
    if METRIC_HIGHER_IS_BETTER[selection_metric]:
        return effect_mean - baseline_mean
    return baseline_mean - effect_mean


def _best_metric_row(frame: pd.DataFrame, selection_metric: str, column: str) -> pd.Series:
    selection_metric = _validate_selection_metric(selection_metric)
    if column not in frame.columns:
        raise ValueError(f"Frame is missing selection metric column '{column}'.")
    values = pd.to_numeric(frame[column], errors="coerce")
    if values.notna().sum() == 0:
        raise ValueError(f"Selection metric column '{column}' contains no finite values.")
    index = values.idxmax() if METRIC_HIGHER_IS_BETTER[selection_metric] else values.idxmin()
    return frame.loc[index]


def _selection_window_summary(
    frame: pd.DataFrame,
    column: str,
    selection_metric: str,
    *,
    baseline_window: tuple[float, float],
    effect_window: tuple[float, float],
) -> tuple[float, float, float]:
    baseline_mean = _window_mean(frame, column, *baseline_window)
    effect_mean = _window_mean(frame, column, *effect_window)
    return baseline_mean, effect_mean, _metric_improvement(effect_mean, baseline_mean, selection_metric)


def _active_subject_group_columns(results: pd.DataFrame, csv_paths: list[Path]) -> list[str]:
    columns = _comparison_group_columns(results)
    candidate_columns = []
    for column in columns:
        if column == "emission_mode" and not _raw_csv_has_column(csv_paths, "emission_mode"):
            continue
        candidate_columns.append(column)

    varying_columns = []
    for column in candidate_columns:
        if results[column].nunique(dropna=False) > 1:
            varying_columns.append(column)
    if not varying_columns:
        return []

    active = ["decoder"] if "decoder" in candidate_columns else []
    active.extend(column for column in varying_columns if column != "decoder")
    return active


def _display_group_columns(frame: pd.DataFrame) -> list[str]:
    columns = _comparison_group_columns(frame)
    display = ["decoder"] if "decoder" in columns else []
    display.extend(column for column in columns if column != "decoder" and frame[column].nunique(dropna=False) > 1)
    return display


def _filter_subject_summary_to_summary_groups(subject_summary: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    """Keep subject rows matching explicit singleton decoder/emission groups in summary."""
    filtered = subject_summary
    for column in _comparison_group_columns(summary):
        if column not in filtered.columns:
            continue
        values = summary[column].dropna().astype(str).unique()
        if len(values) != 1:
            continue
        filtered = filtered[filtered[column].astype(str) == values[0]]
    if filtered.empty and not subject_summary.empty:
        raise ValueError("Subject CSVs contain no rows matching the summary CSV decoder/emission condition.")
    return filtered.reset_index(drop=True)


def _append_subject_time_decode_section(
    lines: list[str],
    subject_csvs: list[Path],
    *,
    effect_window: tuple[float, float],
    summary: pd.DataFrame | None = None,
    selection_metric: str = "accuracy",
) -> None:
    selection_metric = _validate_selection_metric(selection_metric)
    subject_summary = summarize_subject_time_decode(subject_csvs, effect_window=effect_window, selection_metric=selection_metric)
    if summary is not None:
        subject_summary = _filter_subject_summary_to_summary_groups(subject_summary, summary)
    subject_group_columns = _comparison_group_columns(subject_summary)
    group_headers = [SUBJECT_TABLE_LABELS.get(column, column.replace("_", " ").title()) for column in subject_group_columns]
    if selection_metric == "accuracy":
        headers = [*group_headers, "Subject", "Peak time (s)", "Peak accuracy", "Effect-window mean accuracy"]
        alignments = [*("---" for _ in group_headers), "---", "---:", "---:", "---:"]
    else:
        metric_label = _metric_label(selection_metric).title()
        headers = [
            *group_headers,
            "Subject",
            "Selected time (s)",
            f"Selected {metric_label}",
            "Accuracy at selected time",
            f"Effect-window mean {metric_label}",
        ]
        alignments = [*("---" for _ in group_headers), "---", "---:", "---:", "---:", "---:"]

    lines.extend(
        [
            "",
            "## Subjects",
            "",
            f"| {' | '.join(headers)} |",
            f"| {' | '.join(alignments)} |",
        ]
    )
    for row in subject_summary.itertuples(index=False):
        group_values = [str(getattr(row, column)) for column in subject_group_columns]
        if selection_metric == "accuracy":
            values = [
                *group_values,
                str(row.subject),
                _format_float(row.peak_time),
                _format_float(row.peak_accuracy),
                _format_float(row.effect_accuracy_mean),
            ]
        else:
            values = [
                *group_values,
                str(row.subject),
                _format_float(row.selected_time),
                _format_float(row.selected_score),
                _format_float(row.peak_accuracy),
                _format_float(row.effect_selection_mean),
            ]
        lines.append(f"| {' | '.join(values)} |")


def summarize_aggregate_time_decode(
    summary: pd.DataFrame,
    *,
    chance: float = 0.5,
    baseline_window: tuple[float, float] = (-0.1, 0.0),
    effect_window: tuple[float, float] = (0.1, 0.8),
    selection_metric: str = "accuracy",
) -> dict[str, float | str | bool]:
    """Summarize an aggregate time-resolved decoding CSV."""
    selection_metric = _validate_selection_metric(selection_metric)
    selection_column = _metric_mean_column(selection_metric)
    required = {"time", "accuracy_mean", "log_loss_mean", "brier_mean", "ece_mean", "n_subjects", selection_column}
    missing = sorted(required.difference(summary.columns))
    if missing:
        raise ValueError(f"Summary is missing required columns: {missing}")

    peak = summary.loc[summary["accuracy_mean"].idxmax()]
    selected = _best_metric_row(summary, selection_metric, selection_column)
    effect_mean = _window_mean(summary, "accuracy_mean", *effect_window)
    baseline_mean = _window_mean(summary, "accuracy_mean", *baseline_window)
    selection_baseline, selection_effect, selection_improvement = _selection_window_summary(
        summary,
        selection_column,
        selection_metric,
        baseline_window=baseline_window,
        effect_window=effect_window,
    )

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
        "selection_metric": selection_metric,
        "selection_higher_is_better": METRIC_HIGHER_IS_BETTER[selection_metric],
        "selected_time": float(selected["time"]),
        "selected_score": float(selected[selection_column]),
        "selected_accuracy": float(selected["accuracy_mean"]),
        "selected_log_loss": float(selected["log_loss_mean"]),
        "selected_brier": float(selected["brier_mean"]),
        "selected_ece": float(selected["ece_mean"]),
        "baseline_selection_mean": selection_baseline,
        "effect_selection_mean": selection_effect,
        "selection_improvement": selection_improvement,
    }


def summarize_subject_time_decode(
    csv_paths: list[Path],
    *,
    effect_window: tuple[float, float] = (0.1, 0.8),
    selection_metric: str = "accuracy",
) -> pd.DataFrame:
    """Summarize subject-level selected time and effect-window decoding metrics."""
    selection_metric = _validate_selection_metric(selection_metric)
    results = read_time_decode_results(csv_paths)
    group_columns = _active_subject_group_columns(results, csv_paths)
    subject_time_keys = [*group_columns, "subject", "time"]
    by_subject_time = mean_across_folds(results, subject_time_keys).sort_values(subject_time_keys)

    rows = []
    subject_keys = [*group_columns, "subject"]
    for keys, subject_frame in _iter_groups(by_subject_time, subject_keys):
        key_values = keys if isinstance(keys, tuple) else (keys,)
        identity = dict(zip(subject_keys, key_values, strict=True))
        selected = _best_metric_row(subject_frame, selection_metric, selection_metric)
        rows.append(
            {
                **identity,
                "peak_time": float(selected["time"]),
                "peak_accuracy": float(selected["accuracy"]),
                "effect_accuracy_mean": _window_mean(subject_frame, "accuracy", *effect_window),
                "selection_metric": selection_metric,
                "selected_time": float(selected["time"]),
                "selected_score": float(selected[selection_metric]),
                "effect_selection_mean": _window_mean(subject_frame, selection_metric, *effect_window),
            }
        )

    return pd.DataFrame(rows).sort_values(subject_keys).reset_index(drop=True)


def summarize_decoder_comparison(
    summary: pd.DataFrame,
    *,
    baseline_window: tuple[float, float] = (-0.1, 0.0),
    effect_window: tuple[float, float] = (0.1, 0.8),
    selection_metric: str = "accuracy",
) -> pd.DataFrame:
    """Summarize aggregate benchmark metrics separately for each decoder."""
    selection_metric = _validate_selection_metric(selection_metric)
    selection_column = _metric_mean_column(selection_metric)
    if selection_column not in summary.columns:
        raise ValueError(f"Summary must contain '{selection_column}' for selection by {selection_metric}.")

    group_columns = _comparison_group_columns(summary)
    if "decoder" not in group_columns:
        raise ValueError("Summary must contain a 'decoder' column for decoder comparison.")

    rows = []
    for keys, decoder_frame in _iter_groups(summary, group_columns):
        key_values = keys if isinstance(keys, tuple) else (keys,)
        group_values = dict(zip(group_columns, key_values, strict=True))
        selected = _best_metric_row(decoder_frame, selection_metric, selection_column)
        baseline_accuracy = _window_mean(decoder_frame, "accuracy_mean", *baseline_window)
        effect_accuracy = _window_mean(decoder_frame, "accuracy_mean", *effect_window)
        selection_baseline, selection_effect, selection_improvement = _selection_window_summary(
            decoder_frame,
            selection_column,
            selection_metric,
            baseline_window=baseline_window,
            effect_window=effect_window,
        )
        rows.append(
            {
                **group_values,
                "n_subjects": int(selected["n_subjects"]),
                "selection_metric": selection_metric,
                "selected_time": float(selected["time"]),
                "selected_score": float(selected[selection_column]),
                "baseline_selection_mean": selection_baseline,
                "effect_selection_mean": selection_effect,
                "selection_improvement": selection_improvement,
                "peak_time": float(selected["time"]),
                "peak_accuracy": float(selected["accuracy_mean"]),
                "baseline_accuracy_mean": baseline_accuracy,
                "effect_accuracy_mean": effect_accuracy,
                "effect_minus_baseline": effect_accuracy - baseline_accuracy,
                "peak_log_loss": float(selected["log_loss_mean"]),
                "peak_brier": float(selected["brier_mean"]),
                "peak_ece": float(selected["ece_mean"]),
            }
        )
    comparison = pd.DataFrame(rows)
    comparison = comparison.sort_values(
        ["selection_improvement", "selected_score"],
        ascending=[False, not METRIC_HIGHER_IS_BETTER[selection_metric]],
        kind="mergesort",
    )
    return comparison.reset_index(drop=True)


def build_time_decode_report(
    summary_csv: Path,
    *,
    subject_csvs: list[Path] | None = None,
    chance: float = 0.5,
    baseline_window: tuple[float, float] = (-0.1, 0.0),
    effect_window: tuple[float, float] = (0.1, 0.8),
    selection_metric: str = "accuracy",
) -> str:
    """Build a Markdown report for a time-resolved decoding benchmark."""
    selection_metric = _validate_selection_metric(selection_metric)
    summary = pd.read_csv(summary_csv)
    comparison_columns = _comparison_group_columns(summary)
    has_comparison = "decoder" in summary.columns and summary.groupby(comparison_columns).ngroups > 1
    if has_comparison:
        comparison = summarize_decoder_comparison(
            summary,
            baseline_window=baseline_window,
            effect_window=effect_window,
            selection_metric=selection_metric,
        )
        display_group_columns = _display_group_columns(comparison)
        group_headers = [
            SUBJECT_TABLE_LABELS.get(column, column.replace("_", " ").title()) for column in display_group_columns
        ]
        if selection_metric == "accuracy":
            headers = [
                *group_headers,
                "Subjects",
                "Peak time (s)",
                "Peak accuracy",
                "Baseline accuracy",
                "Effect accuracy",
                "Effect minus baseline",
                "Peak log loss",
                "Peak Brier",
                "Peak ECE",
            ]
            alignments = [*("---" for _ in group_headers), "---:", "---:", "---:", "---:", "---:", "---:", "---:", "---:", "---:"]
        else:
            metric_label = _metric_label(selection_metric).title()
            headers = [
                *group_headers,
                "Subjects",
                "Selected time (s)",
                f"Selected {metric_label}",
                f"Baseline {metric_label}",
                f"Effect {metric_label}",
                "Selection improvement",
                "Accuracy at selected time",
                "Log loss at selected time",
                "Brier at selected time",
                "ECE at selected time",
            ]
            alignments = [
                *("---" for _ in group_headers),
                "---:",
                "---:",
                "---:",
                "---:",
                "---:",
                "---:",
                "---:",
                "---:",
                "---:",
                "---:",
            ]
        lines = [
            "# RepTrace Decoder Comparison Report",
            "",
            f"- Summary CSV: `{summary_csv}`",
            f"- Baseline window: {_format_float(baseline_window[0])} to {_format_float(baseline_window[1])} s",
            f"- Effect window: {_format_float(effect_window[0])} to {_format_float(effect_window[1])} s",
            f"- Selection metric: {_metric_descriptor(selection_metric)}",
            "",
            f"| {' | '.join(headers)} |",
            f"| {' | '.join(alignments)} |",
        ]
        for row in comparison.itertuples(index=False):
            group_values = [str(getattr(row, column)) for column in display_group_columns]
            if selection_metric == "accuracy":
                values = [
                    *group_values,
                    str(row.n_subjects),
                    _format_float(row.peak_time),
                    _format_float(row.peak_accuracy),
                    _format_float(row.baseline_accuracy_mean),
                    _format_float(row.effect_accuracy_mean),
                    _format_float(row.effect_minus_baseline),
                    _format_float(row.peak_log_loss),
                    _format_float(row.peak_brier),
                    _format_float(row.peak_ece),
                ]
            else:
                values = [
                    *group_values,
                    str(row.n_subjects),
                    _format_float(row.selected_time),
                    _format_float(row.selected_score),
                    _format_float(row.baseline_selection_mean),
                    _format_float(row.effect_selection_mean),
                    _format_float(row.selection_improvement),
                    _format_float(row.peak_accuracy),
                    _format_float(row.peak_log_loss),
                    _format_float(row.peak_brier),
                    _format_float(row.peak_ece),
                ]
            lines.append(f"| {' | '.join(values)} |")
        if subject_csvs:
            _append_subject_time_decode_section(lines, subject_csvs, effect_window=effect_window, selection_metric=selection_metric)
        lines.append("")
        return "\n".join(lines)

    aggregate = summarize_aggregate_time_decode(
        summary,
        chance=chance,
        baseline_window=baseline_window,
        effect_window=effect_window,
        selection_metric=selection_metric,
    )

    lines = [
        "# RepTrace Time-Decoding Report",
        "",
        f"- Summary CSV: `{summary_csv}`",
        f"- Subjects: {int(aggregate['n_subjects'])}",
        f"- Chance level: {_format_float(aggregate['chance'])}",
        f"- Baseline window: {_format_float(baseline_window[0])} to {_format_float(baseline_window[1])} s",
        f"- Effect window: {_format_float(effect_window[0])} to {_format_float(effect_window[1])} s",
        f"- Selection metric: {_metric_descriptor(selection_metric)}",
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
    if selection_metric != "accuracy":
        metric_label = _metric_label(selection_metric).title()
        lines.extend(
            [
                f"| Selected metric | {_metric_descriptor(selection_metric)} |",
                f"| Selected time | {_format_float(aggregate['selected_time'])} s |",
                f"| Selected {metric_label} | {_format_float(aggregate['selected_score'])} |",
                f"| Accuracy at selected time | {_format_float(aggregate['selected_accuracy'])} |",
                f"| Baseline-window mean {metric_label} | {_format_float(aggregate['baseline_selection_mean'])} |",
                f"| Effect-window mean {metric_label} | {_format_float(aggregate['effect_selection_mean'])} |",
                f"| Effect-window improvement in {metric_label} | {_format_float(aggregate['selection_improvement'])} |",
            ]
        )

    if subject_csvs:
        _append_subject_time_decode_section(lines, subject_csvs, effect_window=effect_window, summary=summary, selection_metric=selection_metric)

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
    parser.add_argument(
        "--selection-metric",
        choices=METRIC_COLUMNS,
        default="accuracy",
        help="Metric used to select/rank report rows. Accuracy is maximized; log_loss, brier, and ece are minimized.",
    )
    args = parser.parse_args()

    report = build_time_decode_report(
        args.summary_csv,
        subject_csvs=_expand_paths(args.subject_csv),
        chance=args.chance,
        baseline_window=tuple(args.baseline_window),
        effect_window=tuple(args.effect_window),
        selection_metric=args.selection_metric,
    )
    if args.out is None:
        print(report)
        return

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report, encoding="utf-8")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
