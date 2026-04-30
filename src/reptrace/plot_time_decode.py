from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from reptrace.results import METRIC_COLUMNS


def _summary_from_csv(results_csv: Path) -> pd.DataFrame:
    results = pd.read_csv(results_csv)
    if "time" not in results.columns:
        raise ValueError("Results CSV must contain a 'time' column.")

    if all(f"{metric}_mean" in results.columns for metric in METRIC_COLUMNS):
        summary = results.copy()
        for metric in METRIC_COLUMNS:
            if f"{metric}_sem" not in summary.columns:
                summary[f"{metric}_sem"] = 0.0
        return summary.sort_values("time")

    missing = [metric for metric in METRIC_COLUMNS if metric not in results.columns]
    if missing:
        raise ValueError(f"Results CSV is missing required metric columns: {missing}")

    grouped = results.groupby("time", as_index=False)
    summary = grouped[list(METRIC_COLUMNS)].mean()
    for metric in METRIC_COLUMNS:
        sem = grouped[metric].sem().rename(columns={metric: f"{metric}_sem"})
        summary = summary.merge(sem, on="time")
        summary = summary.rename(columns={metric: f"{metric}_mean"})
    return summary.sort_values("time")


def plot_time_decode_results(
    results_csv: Path,
    out_path: Path,
    *,
    metrics: tuple[str, ...] = METRIC_COLUMNS,
    chance: float | None = None,
    title: str | None = None,
) -> Path:
    """Plot time-resolved decoding metrics from a raw or aggregated CSV file."""
    unknown = [metric for metric in metrics if metric not in METRIC_COLUMNS]
    if unknown:
        raise ValueError(f"Unknown metrics: {unknown}")

    summary = _summary_from_csv(results_csv)
    n_metrics = len(metrics)
    n_cols = 2 if n_metrics > 1 else 1
    n_rows = (n_metrics + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 3.6 * n_rows), squeeze=False)
    axes_flat = axes.ravel()

    for index, metric in enumerate(metrics):
        ax = axes_flat[index]
        mean = summary[f"{metric}_mean"]
        sem = summary[f"{metric}_sem"].fillna(0.0)
        ax.plot(summary["time"], mean, label=metric)
        ax.fill_between(summary["time"], mean - sem, mean + sem, alpha=0.2)
        if metric == "accuracy" and chance is not None:
            ax.axhline(chance, color="0.4", linestyle="--", linewidth=1.0, label="chance")
        ax.axvline(0.0, color="0.6", linestyle=":", linewidth=1.0)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(metric)
        ax.set_title(metric.replace("_", " ").title())
        ax.legend(loc="best")

    for index in range(n_metrics, len(axes_flat)):
        axes_flat[index].axis("off")

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot time-resolved decoding metrics from a RepTrace result CSV."
    )
    parser.add_argument("results_csv", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=METRIC_COLUMNS,
        default=list(METRIC_COLUMNS),
    )
    parser.add_argument("--chance", type=float)
    parser.add_argument("--title")
    args = parser.parse_args()

    plot_time_decode_results(
        args.results_csv,
        out_path=args.out,
        metrics=tuple(args.metrics),
        chance=args.chance,
        title=args.title,
    )
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
