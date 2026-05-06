from __future__ import annotations

import argparse
import itertools
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from reptrace.onset_detection import DEFAULT_THRESHOLD_QUANTILE, DEFAULT_THRESHOLD_WINDOW, THRESHOLD_METHODS
from reptrace.onset_workflow import DEFAULT_OBSERVATIONS_GLOB, _expand_task_dirs, run_onset_workflow


@dataclass(frozen=True)
class OnsetSensitivitySetting:
    """One onset-detection parameter setting to evaluate."""

    threshold_method: str
    threshold_quantile: float
    min_consecutive: int
    min_duration: float | None = None
    require_stable_prediction: bool = False

    @property
    def setting_id(self) -> str:
        method = self.threshold_method.replace("_", "")
        quantile = f"q{int(round(self.threshold_quantile * 1000)):04d}"
        consecutive = f"c{self.min_consecutive:02d}"
        duration = "dnone" if self.min_duration is None else f"d{int(round(self.min_duration * 1000)):04d}ms"
        stable = "stable" if self.require_stable_prediction else "anypred"
        return "_".join([method, quantile, consecutive, duration, stable])


@dataclass(frozen=True)
class OnsetSensitivityRun:
    """Top-level outputs from an onset sensitivity run."""

    out_dir: Path
    settings: list[OnsetSensitivitySetting]
    sensitivity_summary_csv: Path
    robustness_summary_csv: Path
    plot_path: Path | None


def build_sensitivity_settings(
    *,
    threshold_methods: list[str] | tuple[str, ...] = ("point",),
    threshold_quantiles: list[float] | tuple[float, ...] = (DEFAULT_THRESHOLD_QUANTILE,),
    min_consecutive_values: list[int] | tuple[int, ...] = (1,),
    min_duration_values: list[float | None] | tuple[float | None, ...] | None = None,
    stable_prediction_values: list[bool] | tuple[bool, ...] = (False,),
) -> list[OnsetSensitivitySetting]:
    """Return a deterministic grid of onset-detection settings."""

    if min_duration_values is None:
        min_duration_values = (None,)
    settings = []
    for threshold_method, threshold_quantile, min_consecutive, min_duration, stable_prediction in itertools.product(
        threshold_methods,
        threshold_quantiles,
        min_consecutive_values,
        min_duration_values,
        stable_prediction_values,
    ):
        if threshold_method not in THRESHOLD_METHODS:
            raise ValueError(f"threshold methods must be one of {THRESHOLD_METHODS}.")
        if not 0.0 <= threshold_quantile <= 1.0:
            raise ValueError("threshold quantiles must be between 0 and 1.")
        if min_consecutive < 1:
            raise ValueError("min_consecutive values must be at least 1.")
        if min_duration is not None and min_duration < 0:
            raise ValueError("min_duration values must be non-negative.")
        settings.append(
            OnsetSensitivitySetting(
                threshold_method=threshold_method,
                threshold_quantile=float(threshold_quantile),
                min_consecutive=int(min_consecutive),
                min_duration=None if min_duration is None else float(min_duration),
                require_stable_prediction=bool(stable_prediction),
            )
        )
    return settings


def _read_setting_summary(path: Path, setting: OnsetSensitivitySetting) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame.insert(0, "require_stable_prediction_setting", setting.require_stable_prediction)
    frame.insert(0, "min_duration_setting", np.nan if setting.min_duration is None else setting.min_duration)
    frame.insert(0, "min_consecutive_setting", setting.min_consecutive)
    frame.insert(0, "threshold_quantile_setting", setting.threshold_quantile)
    frame.insert(0, "threshold_method_setting", setting.threshold_method)
    frame.insert(0, "setting_id", setting.setting_id)
    return frame


def _present_group_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in ("task", "decoder", "emission_mode") if column in frame.columns]


def _iqr(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return float("nan")
    return float(clean.quantile(0.75) - clean.quantile(0.25))


def _range(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return float("nan")
    return float(clean.max() - clean.min())


def summarize_sensitivity(summary: pd.DataFrame) -> pd.DataFrame:
    """Summarize onset stability across the parameter grid."""

    if summary.empty:
        raise ValueError("Cannot summarize an empty sensitivity table.")
    group_columns = _present_group_columns(summary)
    if not group_columns:
        raise ValueError("Sensitivity table must contain at least one grouping column.")

    rows = []
    for keys, group in summary.groupby(group_columns, sort=True):
        key_values = keys if isinstance(keys, tuple) else (keys,)
        group_values = dict(zip(group_columns, key_values, strict=True))
        latencies = pd.to_numeric(group["post_detection_latency_median"], errors="coerce")
        rows.append(
            {
                **group_values,
                "n_settings": int(group["setting_id"].nunique()),
                "latency_median_across_settings": float(latencies.median()) if latencies.notna().any() else np.nan,
                "latency_iqr_across_settings": _iqr(latencies),
                "latency_range_across_settings": _range(latencies),
                "false_alarm_rate_mean": float(group["false_alarm_rate"].mean()),
                "false_alarm_rate_max": float(group["false_alarm_rate"].max()),
                "post_zero_detected_rate_mean": float(group["post_zero_detected_rate"].mean()),
                "post_zero_detected_rate_min": float(group["post_zero_detected_rate"].min()),
                "correct_detection_rate_mean": float(group["correct_detection_rate"].mean()),
                "correct_detection_rate_min": float(group["correct_detection_rate"].min()),
                "post_detection_run_length_median": float(group["post_detection_run_length_median"].median())
                if "post_detection_run_length_median" in group
                else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values(group_columns).reset_index(drop=True)


def plot_sensitivity_summary(summary: pd.DataFrame, out_path: Path) -> Path:
    """Plot onset latency and false-alarm sensitivity across settings."""

    if summary.empty:
        raise ValueError("Cannot plot an empty sensitivity table.")
    required = {"setting_id", "task", "post_detection_latency_median", "false_alarm_rate"}
    missing = sorted(required.difference(summary.columns))
    if missing:
        raise ValueError(f"Sensitivity summary is missing required columns for plotting: {missing}")

    setting_ids = list(dict.fromkeys(summary["setting_id"].astype(str)))
    setting_index = {setting_id: index for index, setting_id in enumerate(setting_ids)}
    frame = summary.copy()
    frame["setting_index"] = frame["setting_id"].astype(str).map(setting_index)
    group_columns = _present_group_columns(frame)

    fig, axes = plt.subplots(1, 2, figsize=(max(8.5, 0.65 * len(setting_ids)), 4.5))
    for keys, group in frame.groupby(group_columns, sort=True):
        key_values = keys if isinstance(keys, tuple) else (keys,)
        label = " / ".join(map(str, key_values))
        group = group.sort_values("setting_index")
        axes[0].plot(group["setting_index"], group["post_detection_latency_median"], marker="o", label=label)
        axes[1].plot(group["setting_index"], group["false_alarm_rate"], marker="o", label=label)

    axes[0].axhline(0.0, color="0.4", linewidth=1.0)
    axes[0].set_ylabel("Median post-zero onset latency (s)")
    axes[0].set_title("Latency sensitivity")
    axes[0].grid(axis="y", color="0.9", linewidth=0.8)

    axes[1].set_ylabel("False-alarm rate")
    axes[1].set_title("False-alarm sensitivity")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].grid(axis="y", color="0.9", linewidth=0.8)

    for ax in axes:
        ax.set_xticks(range(len(setting_ids)))
        ax.set_xticklabels(setting_ids, rotation=45, ha="right")
        ax.set_xlabel("Setting")
    axes[1].legend(loc="best", fontsize="small")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


# pylint: disable-next=too-many-arguments,too-many-locals
def run_onset_sensitivity(
    task_dirs: list[Path],
    *,
    out_dir: Path,
    observations_glob: str = DEFAULT_OBSERVATIONS_GLOB,
    threshold_window: tuple[float, float] = DEFAULT_THRESHOLD_WINDOW,
    threshold_methods: list[str] | tuple[str, ...] = ("point",),
    threshold_quantiles: list[float] | tuple[float, ...] = (DEFAULT_THRESHOLD_QUANTILE,),
    detection_start: float | None = None,
    min_consecutive_values: list[int] | tuple[int, ...] = (1,),
    min_duration_values: list[float | None] | tuple[float | None, ...] | None = None,
    include_stable_prediction: bool = False,
    score_column: str = "confidence",
    allow_missing: bool = False,
    plot_out: Path | None = None,
) -> OnsetSensitivityRun:
    """Run onset detection over a grid of threshold and persistence settings."""

    stable_values = (False, True) if include_stable_prediction else (False,)
    settings = build_sensitivity_settings(
        threshold_methods=threshold_methods,
        threshold_quantiles=threshold_quantiles,
        min_consecutive_values=min_consecutive_values,
        min_duration_values=min_duration_values,
        stable_prediction_values=stable_values,
    )
    if not settings:
        raise ValueError("At least one onset sensitivity setting is required.")

    out_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for setting in settings:
        setting_dir = out_dir / setting.setting_id
        workflow = run_onset_workflow(
            task_dirs,
            out_dir=setting_dir,
            observations_glob=observations_glob,
            threshold_window=threshold_window,
            threshold_method=setting.threshold_method,
            threshold_quantile=setting.threshold_quantile,
            score_column=score_column,
            detection_start=detection_start,
            min_consecutive=setting.min_consecutive,
            min_duration=setting.min_duration,
            require_stable_prediction=setting.require_stable_prediction,
            allow_missing=allow_missing,
            write_combined_events=False,
        )
        frames.append(_read_setting_summary(workflow.summary_all_csv, setting))

    sensitivity_summary = pd.concat(frames, ignore_index=True)
    sensitivity_summary_csv = out_dir / "onset_sensitivity_summary.csv"
    sensitivity_summary.to_csv(sensitivity_summary_csv, index=False)

    robustness_summary = summarize_sensitivity(sensitivity_summary)
    robustness_summary_csv = out_dir / "onset_sensitivity_robustness.csv"
    robustness_summary.to_csv(robustness_summary_csv, index=False)

    plot_path = None
    if plot_out is not None:
        plot_path = plot_sensitivity_summary(sensitivity_summary, plot_out)

    return OnsetSensitivityRun(
        out_dir=out_dir,
        settings=settings,
        sensitivity_summary_csv=sensitivity_summary_csv,
        robustness_summary_csv=robustness_summary_csv,
        plot_path=plot_path,
    )


def _duration_values(raw_values: list[float] | None) -> list[float | None] | None:
    if raw_values is None:
        return None
    return [float(value) for value in raw_values]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RepTrace onset sensitivity sweeps across task directories.")
    parser.add_argument("--task-dir", action="append", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--observations-glob", default=DEFAULT_OBSERVATIONS_GLOB)
    parser.add_argument(
        "--threshold-window",
        nargs=2,
        type=float,
        default=DEFAULT_THRESHOLD_WINDOW,
        metavar=("START", "STOP"),
    )
    parser.add_argument("--threshold-quantiles", nargs="+", type=float, default=[0.90, 0.95, 0.975])
    parser.add_argument("--threshold-methods", nargs="+", choices=THRESHOLD_METHODS, default=["point"])
    parser.add_argument("--detection-start", type=float)
    parser.add_argument("--min-consecutive-values", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--min-duration-values", nargs="*", type=float)
    parser.add_argument("--include-stable-prediction", action="store_true")
    parser.add_argument("--score-column", default="confidence")
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument("--plot-out", type=Path)
    args = parser.parse_args()

    run = run_onset_sensitivity(
        _expand_task_dirs(args.task_dir),
        out_dir=args.out_dir,
        observations_glob=args.observations_glob,
        threshold_window=tuple(args.threshold_window),
        threshold_methods=tuple(args.threshold_methods),
        threshold_quantiles=tuple(args.threshold_quantiles),
        detection_start=args.detection_start,
        min_consecutive_values=tuple(args.min_consecutive_values),
        min_duration_values=_duration_values(args.min_duration_values),
        include_stable_prediction=args.include_stable_prediction,
        score_column=args.score_column,
        allow_missing=args.allow_missing,
        plot_out=args.plot_out,
    )
    print(f"Wrote sensitivity summary: {run.sensitivity_summary_csv}")
    print(f"Wrote robustness summary: {run.robustness_summary_csv}")
    if run.plot_path is not None:
        print(f"Wrote sensitivity plot: {run.plot_path}")
    print(f"Evaluated {len(run.settings)} onset setting(s).")


if __name__ == "__main__":
    main()
