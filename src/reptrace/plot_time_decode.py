from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from reptrace.results import METRIC_COLUMNS, SUMMARY_GROUP_COLUMNS


def _group_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in SUMMARY_GROUP_COLUMNS if column in frame.columns]


def _group_label(group_name) -> str:
    if isinstance(group_name, tuple):
        return " / ".join(map(str, group_name))
    return str(group_name)


def _summary_from_csv(results_csv: Path) -> pd.DataFrame:
    results = pd.read_csv(results_csv)
    if "time" not in results.columns:
        raise ValueError("Results CSV must contain a 'time' column.")

    if all(f"{metric}_mean" in results.columns for metric in METRIC_COLUMNS):
        summary = results.copy()
        for metric in METRIC_COLUMNS:
            if f"{metric}_sem" not in summary.columns:
                summary[f"{metric}_sem"] = 0.0
        sort_columns = [*_group_columns(summary), "time"]
        return summary.sort_values(sort_columns or ["time"])

    missing = [metric for metric in METRIC_COLUMNS if metric not in results.columns]
    if missing:
        raise ValueError(f"Results CSV is missing required metric columns: {missing}")

    group_columns = _group_columns(results)
    grouped = results.groupby([*group_columns, "time"], as_index=False)
    summary = grouped[list(METRIC_COLUMNS)].mean()
    for metric in METRIC_COLUMNS:
        sem = grouped[metric].sem().rename(columns={metric: f"{metric}_sem"})
        summary = summary.merge(sem, on="time")
        summary = summary.rename(columns={metric: f"{metric}_mean"})
    return summary.sort_values([*group_columns, "time"] or ["time"])


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
    group_columns = _group_columns(summary)
    plot_groups = list(summary.groupby(group_columns, sort=True)) if group_columns else [(None, summary)]
    n_metrics = len(metrics)
    n_cols = 2 if n_metrics > 1 else 1
    n_rows = (n_metrics + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 3.6 * n_rows), squeeze=False)
    axes_flat = axes.ravel()

    for index, metric in enumerate(metrics):
        ax = axes_flat[index]
        for group_name, group in plot_groups:
            mean = group[f"{metric}_mean"]
            sem = group[f"{metric}_sem"].fillna(0.0)
            label = _group_label(group_name) if group_name is not None else metric
            ax.plot(group["time"], mean, label=label)
            ax.fill_between(group["time"], mean - sem, mean + sem, alpha=0.2)
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
