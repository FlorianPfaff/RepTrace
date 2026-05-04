from __future__ import annotations

import argparse
import glob
from pathlib import Path

import pandas as pd

CALIBRATION_METRICS = ("log_loss", "brier", "ece")


def _expand_paths(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if matches:
            paths.extend(Path(match) for match in matches)
        else:
            paths.append(Path(pattern))
    return paths


def _window_mean(frame: pd.DataFrame, column: str, start: float, stop: float) -> float:
    window = frame[(frame["time"] >= start) & (frame["time"] <= stop)]
    if window.empty:
        raise ValueError(f"No time points found in window [{start}, {stop}].")
    return float(window[column].mean())


def _format_float(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def summarize_calibration_metrics(
    summary: pd.DataFrame,
    *,
    baseline_window: tuple[float, float] = (-0.1, 0.0),
    effect_window: tuple[float, float] = (0.1, 0.8),
) -> pd.DataFrame:
    """Summarize accuracy and calibration metrics over benchmark time windows."""
    required = {"time", "accuracy_mean", "log_loss_mean", "brier_mean", "ece_mean", "n_subjects"}
    missing = sorted(required.difference(summary.columns))
    if missing:
        raise ValueError(f"Summary is missing required columns: {missing}")

    group_items = summary.groupby("decoder", sort=True) if "decoder" in summary.columns else [("overall", summary)]
    rows = []
    for decoder, frame in group_items:
        effect = frame[(frame["time"] >= effect_window[0]) & (frame["time"] <= effect_window[1])]
        if effect.empty:
            raise ValueError(f"No time points found in effect window [{effect_window[0]}, {effect_window[1]}].")
        best_ece = effect.loc[effect["ece_mean"].idxmin()]
        rows.append(
            {
                "decoder": str(decoder),
                "n_subjects": int(frame["n_subjects"].max()),
                "baseline_accuracy_mean": _window_mean(frame, "accuracy_mean", *baseline_window),
                "effect_accuracy_mean": _window_mean(frame, "accuracy_mean", *effect_window),
                "effect_log_loss_mean": _window_mean(frame, "log_loss_mean", *effect_window),
                "effect_brier_mean": _window_mean(frame, "brier_mean", *effect_window),
                "effect_ece_mean": _window_mean(frame, "ece_mean", *effect_window),
                "best_ece_time": float(best_ece["time"]),
                "best_ece": float(best_ece["ece_mean"]),
                "accuracy_at_best_ece": float(best_ece["accuracy_mean"]),
                "brier_at_best_ece": float(best_ece["brier_mean"]),
                "log_loss_at_best_ece": float(best_ece["log_loss_mean"]),
            }
        )

    return pd.DataFrame(rows).sort_values(["effect_ece_mean", "effect_brier_mean", "effect_log_loss_mean"]).reset_index(drop=True)


def aggregate_reliability_bins(csv_paths: list[Path]) -> pd.DataFrame:
    """Aggregate reliability-bin CSVs emitted by ``reptrace.mne_time_decode``."""
    if not csv_paths:
        raise ValueError("At least one calibration-bin CSV path is required.")

    frames = []
    for csv_path in csv_paths:
        frame = pd.read_csv(csv_path)
        missing = sorted({"time", "bin", "bin_left", "bin_right", "n_samples", "accuracy", "confidence"}.difference(frame.columns))
        if missing:
            raise ValueError(f"{csv_path} is missing required columns: {missing}")
        if "decoder" not in frame.columns:
            frame["decoder"] = "overall"
        frame["source_file"] = csv_path.name
        frames.append(frame)

    bins = pd.concat(frames, ignore_index=True)
    group_columns = ["decoder", "time", "bin", "bin_left", "bin_right"]
    rows = []
    for keys, group in bins.groupby(group_columns, sort=True):
        n_samples = group["n_samples"].sum()
        if n_samples:
            weights = group["n_samples"] / n_samples
            accuracy = float((group["accuracy"].fillna(0.0) * weights).sum())
            confidence = float((group["confidence"].fillna(0.0) * weights).sum())
        else:
            accuracy = float("nan")
            confidence = float("nan")
        rows.append(
            {
                **dict(zip(group_columns, keys)),
                "n_samples": int(n_samples),
                "accuracy": accuracy,
                "confidence": confidence,
                "gap": accuracy - confidence if n_samples else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def build_calibration_report(
    summary_csv: Path,
    *,
    baseline_window: tuple[float, float] = (-0.1, 0.0),
    effect_window: tuple[float, float] = (0.1, 0.8),
) -> str:
    """Build a Markdown report that foregrounds calibration metrics."""
    summary = summarize_calibration_metrics(
        pd.read_csv(summary_csv),
        baseline_window=baseline_window,
        effect_window=effect_window,
    )
    lines = [
        "# RepTrace Calibration Report",
        "",
        f"- Summary CSV: `{summary_csv}`",
        f"- Baseline window: {_format_float(baseline_window[0])} to {_format_float(baseline_window[1])} s",
        f"- Effect window: {_format_float(effect_window[0])} to {_format_float(effect_window[1])} s",
        "",
        "| Decoder | Subjects | Effect ECE | Effect Brier | Effect log loss | Effect accuracy | Baseline accuracy | Best ECE time (s) | Best ECE | Accuracy at best ECE |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.decoder} | {row.n_subjects} | {_format_float(row.effect_ece_mean)} | {_format_float(row.effect_brier_mean)} | "
            f"{_format_float(row.effect_log_loss_mean)} | {_format_float(row.effect_accuracy_mean)} | {_format_float(row.baseline_accuracy_mean)} | "
            f"{_format_float(row.best_ece_time)} | {_format_float(row.best_ece)} | {_format_float(row.accuracy_at_best_ece)} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create calibration-focused summaries from RepTrace benchmark outputs."
    )
    parser.add_argument("summary_csv", type=Path)
    parser.add_argument("calibration_csv", nargs="*", help="Optional reliability-bin CSVs or glob patterns.")
    parser.add_argument("--out-report", type=Path)
    parser.add_argument("--out-bins", type=Path)
    parser.add_argument("--baseline-window", type=float, nargs=2, default=(-0.1, 0.0), metavar=("START", "STOP"))
    parser.add_argument("--effect-window", type=float, nargs=2, default=(0.1, 0.8), metavar=("START", "STOP"))
    args = parser.parse_args()

    report = build_calibration_report(
        args.summary_csv,
        baseline_window=tuple(args.baseline_window),
        effect_window=tuple(args.effect_window),
    )
    if args.out_report is None:
        print(report)
    else:
        args.out_report.parent.mkdir(parents=True, exist_ok=True)
        args.out_report.write_text(report, encoding="utf-8")
        print(f"Wrote calibration report: {args.out_report}")

    calibration_csvs = _expand_paths(args.calibration_csv)
    if args.out_bins is not None:
        aggregated = aggregate_reliability_bins(calibration_csvs)
        args.out_bins.parent.mkdir(parents=True, exist_ok=True)
        aggregated.to_csv(args.out_bins, index=False)
        print(f"Wrote aggregate reliability bins: {args.out_bins}")


if __name__ == "__main__":
    main()
