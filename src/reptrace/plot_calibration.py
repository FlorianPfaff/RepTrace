from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


def _display_label(row: pd.Series | dict) -> str:
    decoder = str(row["decoder"])
    emission_mode = row.get("emission_mode")
    return decoder if pd.isna(emission_mode) or emission_mode is None else f"{decoder} / {emission_mode}"


def _window(frame: pd.DataFrame, time_window: tuple[float, float] | None) -> pd.DataFrame:
    if time_window is None:
        return frame
    start, stop = time_window
    window = frame[(frame["time"] >= start) & (frame["time"] <= stop)]
    if window.empty:
        raise ValueError(f"No reliability-bin rows found in time window [{start}, {stop}].")
    return window


def summarize_reliability_curve(
    reliability_bins: pd.DataFrame,
    *,
    time_window: tuple[float, float] | None = None,
) -> pd.DataFrame:
    """Aggregate reliability bins over an optional time window for plotting."""
    required = {"time", "bin", "bin_left", "bin_right", "n_samples", "accuracy", "confidence"}
    missing = sorted(required.difference(reliability_bins.columns))
    if missing:
        raise ValueError(f"Reliability bins are missing required columns: {missing}")

    bins = _window(reliability_bins.copy(), time_window)
    if "decoder" not in bins.columns:
        bins["decoder"] = "overall"
    if "emission_mode" not in bins.columns:
        bins["emission_mode"] = "calibrated"

    rows = []
    group_columns = ["decoder", "emission_mode", "bin", "bin_left", "bin_right"]
    for keys, group in bins.groupby(group_columns, sort=True):
        n_samples = int(group["n_samples"].sum())
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
                "n_samples": n_samples,
                "accuracy": accuracy,
                "confidence": confidence,
                "gap": accuracy - confidence if n_samples else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def plot_reliability_diagram(
    reliability_bins_csv: Path,
    out_path: Path,
    *,
    time_window: tuple[float, float] | None = None,
    title: str | None = None,
) -> Path:
    """Plot a reliability diagram from aggregated reliability-bin CSV output."""
    curve = summarize_reliability_curve(pd.read_csv(reliability_bins_csv), time_window=time_window)
    groups = list(curve[["decoder", "emission_mode"]].drop_duplicates().itertuples(index=False, name=None))
    if not groups:
        raise ValueError("No reliability-bin rows available to plot.")

    n_groups = len(groups)
    n_cols = min(3, n_groups)
    n_rows = (n_groups + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 4.0 * n_rows), squeeze=False)
    axes_flat = axes.ravel()

    for index, (decoder, emission_mode) in enumerate(groups):
        ax = axes_flat[index]
        group = curve[(curve["decoder"] == decoder) & (curve["emission_mode"] == emission_mode) & (curve["n_samples"] > 0)].sort_values("bin")
        ax.plot([0.0, 1.0], [0.0, 1.0], color="0.5", linestyle="--", linewidth=1.0, label="perfect")
        ax.plot(group["confidence"], group["accuracy"], color="tab:blue", linewidth=1.5)
        sizes = 25 + 125 * group["n_samples"] / max(group["n_samples"].max(), 1)
        ax.scatter(group["confidence"], group["accuracy"], s=sizes, color="tab:blue", edgecolor="white", linewidth=0.6)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel("Mean confidence")
        ax.set_ylabel("Empirical accuracy")
        ax.set_title(_display_label({"decoder": decoder, "emission_mode": emission_mode}))
        ax.grid(True, color="0.9", linewidth=0.8)

    for index in range(n_groups, len(axes_flat)):
        axes_flat[index].axis("off")

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot reliability diagrams from RepTrace calibration bins.")
    parser.add_argument("reliability_bins_csv", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--time-window", type=float, nargs=2, metavar=("START", "STOP"))
    parser.add_argument("--title")
    args = parser.parse_args()

    out_path = plot_reliability_diagram(
        args.reliability_bins_csv,
        args.out,
        time_window=tuple(args.time_window) if args.time_window else None,
        title=args.title,
    )
    print(f"Wrote reliability diagram: {out_path}")


if __name__ == "__main__":
    main()
