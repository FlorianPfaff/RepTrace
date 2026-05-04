from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

CONTROL_CONDITIONS = ("baseline_window", "shuffled_time", "shuffled_label")


def _condition_value(frame: pd.DataFrame, condition: str, column: str) -> float:
    row = frame.loc[frame["condition"] == condition]
    if row.empty:
        return float("nan")
    return float(row.iloc[0][column])


def _control_margin(frame: pd.DataFrame) -> float:
    observed = _condition_value(frame, "observed_effect", "persistence_gain_per_observation")
    controls = [
        _condition_value(frame, condition, "persistence_gain_per_observation")
        for condition in CONTROL_CONDITIONS
    ]
    controls = [value for value in controls if not np.isnan(value)]
    return observed - max(controls) if controls else float("nan")


def summarize_emission_mode(frame: pd.DataFrame) -> dict[str, float]:
    """Summarize temporal-model evidence for one emission mode."""
    observed_gain = _condition_value(frame, "observed_effect", "persistence_gain_per_observation")
    baseline_gain = _condition_value(frame, "baseline_window", "persistence_gain_per_observation")
    shuffled_time_gain = _condition_value(frame, "shuffled_time", "persistence_gain_per_observation")
    shuffled_label_gain = _condition_value(frame, "shuffled_label", "persistence_gain_per_observation")
    return {
        "observed_gain": observed_gain,
        "baseline_gain": baseline_gain,
        "effect_minus_baseline_gain": observed_gain - baseline_gain,
        "shuffled_time_gain": shuffled_time_gain,
        "shuffled_label_gain": shuffled_label_gain,
        "effect_minus_shuffled_time_gain": observed_gain - shuffled_time_gain,
        "effect_minus_shuffled_label_gain": observed_gain - shuffled_label_gain,
        "control_margin": _control_margin(frame),
        "shuffled_time_p": _condition_value(frame, "shuffled_time", "empirical_p_value"),
        "shuffled_label_p": _condition_value(frame, "shuffled_label", "empirical_p_value"),
        "best_stay_probability": _condition_value(frame, "observed_effect", "best_stay_probability"),
    }


def compare_emission_modes(summary: pd.DataFrame) -> pd.DataFrame:
    """Compare calibrated and uncalibrated temporal-model evidence by decoder."""
    required = {"decoder", "emission_mode", "condition", "persistence_gain_per_observation"}
    missing = sorted(required.difference(summary.columns))
    if missing:
        raise ValueError(f"Temporal-model summary is missing required columns: {missing}")

    rows = []
    for decoder, decoder_frame in summary.groupby("decoder", sort=True):
        modes = {mode: frame for mode, frame in decoder_frame.groupby("emission_mode", sort=True)}
        if "calibrated" not in modes or "uncalibrated" not in modes:
            continue
        calibrated = summarize_emission_mode(modes["calibrated"])
        uncalibrated = summarize_emission_mode(modes["uncalibrated"])
        delta_control_margin = calibrated["control_margin"] - uncalibrated["control_margin"]
        delta_effect_minus_baseline = calibrated["effect_minus_baseline_gain"] - uncalibrated["effect_minus_baseline_gain"]
        delta_observed_gain = calibrated["observed_gain"] - uncalibrated["observed_gain"]
        preferred = "calibrated" if delta_control_margin >= 0 else "uncalibrated"
        rows.append(
            {
                "decoder": decoder,
                "calibrated_observed_gain": calibrated["observed_gain"],
                "uncalibrated_observed_gain": uncalibrated["observed_gain"],
                "delta_observed_gain": delta_observed_gain,
                "calibrated_control_margin": calibrated["control_margin"],
                "uncalibrated_control_margin": uncalibrated["control_margin"],
                "delta_control_margin": delta_control_margin,
                "calibrated_effect_minus_baseline_gain": calibrated["effect_minus_baseline_gain"],
                "uncalibrated_effect_minus_baseline_gain": uncalibrated["effect_minus_baseline_gain"],
                "delta_effect_minus_baseline_gain": delta_effect_minus_baseline,
                "calibrated_shuffled_time_p": calibrated["shuffled_time_p"],
                "uncalibrated_shuffled_time_p": uncalibrated["shuffled_time_p"],
                "calibrated_shuffled_label_p": calibrated["shuffled_label_p"],
                "uncalibrated_shuffled_label_p": uncalibrated["shuffled_label_p"],
                "calibrated_best_stay_probability": calibrated["best_stay_probability"],
                "uncalibrated_best_stay_probability": uncalibrated["best_stay_probability"],
                "preferred_emission_mode": preferred,
            }
        )
    return pd.DataFrame(rows).sort_values("delta_control_margin", ascending=False).reset_index(drop=True)


def _format_float(value: float, digits: int = 4) -> str:
    return "nan" if pd.isna(value) else f"{value:.{digits}f}"


def build_emission_comparison_report(comparison: pd.DataFrame, *, summary_csv: Path) -> str:
    """Build a compact Markdown report for calibrated-vs-uncalibrated emissions."""
    lines = [
        "# RepTrace Emission Calibration Comparison",
        "",
        f"- Temporal-model summary: `{summary_csv}`",
        "",
        "Question: do calibrated probabilities produce cleaner state inference than",
        "uncalibrated score-derived emissions?",
        "",
        "The main comparison is the control margin: observed effect-window",
        "persistence gain minus the strongest baseline, shuffled-time, or",
        "shuffled-label control gain. Positive deltas favor calibrated emissions.",
        "",
    ]
    if comparison.empty:
        lines.append("No decoder had both calibrated and uncalibrated emission-mode rows.")
        lines.append("")
        return "\n".join(lines)

    lines.extend(
        [
            "| Decoder | Preferred | Delta control margin | Calibrated margin | Uncalibrated margin | Delta effect-baseline | Calibrated p(time) | Uncalibrated p(time) |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in comparison.itertuples(index=False):
        lines.append(
            f"| {row.decoder} | {row.preferred_emission_mode} | {_format_float(row.delta_control_margin)} | "
            f"{_format_float(row.calibrated_control_margin)} | {_format_float(row.uncalibrated_control_margin)} | "
            f"{_format_float(row.delta_effect_minus_baseline_gain)} | {_format_float(row.calibrated_shuffled_time_p)} | "
            f"{_format_float(row.uncalibrated_shuffled_time_p)} |"
        )
    lines.append("")
    return "\n".join(lines)


def compare_temporal_summary(
    summary_csv: Path,
    *,
    out_csv: Path | None = None,
    out_report: Path | None = None,
) -> tuple[pd.DataFrame, str | None]:
    """Compare calibrated and uncalibrated emission rows from a temporal-model summary CSV."""
    comparison = compare_emission_modes(pd.read_csv(summary_csv))
    if out_csv is not None:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        comparison.to_csv(out_csv, index=False)
    report = None
    if out_report is not None:
        report = build_emission_comparison_report(comparison, summary_csv=summary_csv)
        out_report.parent.mkdir(parents=True, exist_ok=True)
        out_report.write_text(report, encoding="utf-8")
    return comparison, report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare calibrated versus uncalibrated emissions in RepTrace temporal-model summaries."
    )
    parser.add_argument("summary_csv", type=Path)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--out-report", type=Path)
    args = parser.parse_args()

    comparison, _ = compare_temporal_summary(
        args.summary_csv,
        out_csv=args.out_csv,
        out_report=args.out_report,
    )
    print(f"Wrote emission comparison: {args.out_csv}")
    if args.out_report is not None:
        print(f"Wrote emission comparison report: {args.out_report}")
    print(comparison.to_string(index=False) if not comparison.empty else "No paired emission modes found.")


if __name__ == "__main__":
    main()
