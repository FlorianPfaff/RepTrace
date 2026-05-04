from __future__ import annotations

import argparse
import glob
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from reptrace.results import read_time_decode_results


METRIC_DIRECTIONS = {
    "effect_accuracy": "higher",
    "effect_minus_baseline": "higher",
    "baseline_abs_delta": "lower",
    "effect_log_loss": "lower",
    "effect_brier": "lower",
    "effect_ece": "lower",
}


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


def subject_decoder_metrics(
    csv_paths: list[Path],
    *,
    chance: float = 0.5,
    baseline_window: tuple[float, float] = (-0.1, 0.0),
    effect_window: tuple[float, float] = (0.1, 0.8),
) -> pd.DataFrame:
    """Return one row per subject and decoder with paired-test metrics."""
    results = read_time_decode_results(csv_paths)
    if "decoder" not in results.columns:
        raise ValueError("Subject CSVs must contain a 'decoder' column.")

    subject_time = (
        results.groupby(["decoder", "subject", "time"], as_index=False)[["accuracy", "log_loss", "brier", "ece"]]
        .mean()
        .sort_values(["decoder", "subject", "time"])
    )
    rows = []
    for (decoder, subject), frame in subject_time.groupby(["decoder", "subject"], sort=True):
        baseline_accuracy = _window_mean(frame, "accuracy", *baseline_window)
        effect_accuracy = _window_mean(frame, "accuracy", *effect_window)
        rows.append(
            {
                "decoder": str(decoder),
                "subject": str(subject),
                "baseline_accuracy": baseline_accuracy,
                "baseline_abs_delta": abs(baseline_accuracy - chance),
                "effect_accuracy": effect_accuracy,
                "effect_minus_baseline": effect_accuracy - baseline_accuracy,
                "effect_log_loss": _window_mean(frame, "log_loss", *effect_window),
                "effect_brier": _window_mean(frame, "brier", *effect_window),
                "effect_ece": _window_mean(frame, "ece", *effect_window),
            }
        )
    return pd.DataFrame(rows)


def sign_flip_p_value(
    differences: np.ndarray,
    *,
    n_permutations: int = 10_000,
    random_state: int = 13,
) -> float:
    """Return a two-sided paired sign-flip p-value for a mean difference."""
    if differences.ndim != 1:
        raise ValueError("differences must be one-dimensional.")
    if len(differences) < 2:
        raise ValueError("Need at least two paired subjects.")
    if n_permutations < 1:
        raise ValueError("n_permutations must be at least 1.")

    observed = abs(float(differences.mean()))
    n_subjects = len(differences)
    if 2**n_subjects <= n_permutations:
        signs = np.array(list(itertools.product([-1.0, 1.0], repeat=n_subjects)))
        null_means = signs @ differences / n_subjects
        return float((np.abs(null_means) >= observed).mean())

    rng = np.random.default_rng(random_state)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_permutations, n_subjects))
    null_means = signs @ differences / n_subjects
    return float((1.0 + (np.abs(null_means) >= observed).sum()) / (n_permutations + 1.0))


def paired_decoder_statistics(
    subject_metrics: pd.DataFrame,
    *,
    metrics: tuple[str, ...] = tuple(METRIC_DIRECTIONS),
    n_permutations: int = 10_000,
    random_state: int = 13,
) -> pd.DataFrame:
    """Compare decoders with subject-level paired sign-flip tests."""
    required = {"decoder", "subject", *metrics}
    missing = sorted(required.difference(subject_metrics.columns))
    if missing:
        raise ValueError(f"Subject metrics are missing required columns: {missing}")

    decoders = sorted(subject_metrics["decoder"].unique())
    if len(decoders) < 2:
        raise ValueError("Need at least two decoders for paired comparison.")

    rows = []
    for decoder_a, decoder_b in itertools.combinations(decoders, 2):
        left = subject_metrics[subject_metrics["decoder"] == decoder_a]
        right = subject_metrics[subject_metrics["decoder"] == decoder_b]
        paired = left.merge(right, on="subject", suffixes=("_a", "_b"))
        if len(paired) < 2:
            raise ValueError(f"Need at least two paired subjects for {decoder_a} vs {decoder_b}.")
        for metric in metrics:
            a_values = paired[f"{metric}_a"].to_numpy(dtype=float)
            b_values = paired[f"{metric}_b"].to_numpy(dtype=float)
            differences = a_values - b_values
            direction = METRIC_DIRECTIONS.get(metric, "higher")
            mean_a = float(a_values.mean())
            mean_b = float(b_values.mean())
            if direction == "lower":
                better = decoder_a if mean_a < mean_b else decoder_b
            else:
                better = decoder_a if mean_a > mean_b else decoder_b
            rows.append(
                {
                    "decoder_a": decoder_a,
                    "decoder_b": decoder_b,
                    "metric": metric,
                    "preferred_direction": direction,
                    "n_subjects": int(len(paired)),
                    "decoder_a_mean": mean_a,
                    "decoder_b_mean": mean_b,
                    "mean_difference_a_minus_b": float(differences.mean()),
                    "median_difference_a_minus_b": float(np.median(differences)),
                    "sign_flip_p": sign_flip_p_value(
                        differences,
                        n_permutations=n_permutations,
                        random_state=random_state,
                    ),
                    "better_decoder_by_mean": better,
                }
            )
    return pd.DataFrame(rows)


def build_paired_stats_report(
    statistics: pd.DataFrame,
    *,
    baseline_window: tuple[float, float] = (-0.1, 0.0),
    effect_window: tuple[float, float] = (0.1, 0.8),
    chance: float = 0.5,
) -> str:
    """Build a Markdown paired decoder statistics report."""
    lines = [
        "# RepTrace Paired Decoder Statistics",
        "",
        f"- Chance level: {_format_float(chance)}",
        f"- Baseline window: {_format_float(baseline_window[0])} to {_format_float(baseline_window[1])} s",
        f"- Effect window: {_format_float(effect_window[0])} to {_format_float(effect_window[1])} s",
        "- Test: two-sided paired sign-flip test over subjects",
        "",
        "| Decoder A | Decoder B | Metric | Preferred | Subjects | A mean | B mean | A minus B | Sign-flip p | Better by mean |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in statistics.itertuples(index=False):
        lines.append(
            f"| {row.decoder_a} | {row.decoder_b} | {row.metric} | {row.preferred_direction} | {row.n_subjects} | "
            f"{_format_float(row.decoder_a_mean)} | {_format_float(row.decoder_b_mean)} | "
            f"{_format_float(row.mean_difference_a_minus_b)} | {_format_float(row.sign_flip_p, digits=4)} | {row.better_decoder_by_mean} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run paired subject-level statistics for decoder comparisons.")
    parser.add_argument("csv", nargs="+", help="Subject result CSVs or glob patterns.")
    parser.add_argument("--out-csv", type=Path)
    parser.add_argument("--out-report", type=Path)
    parser.add_argument("--chance", type=float, default=0.5)
    parser.add_argument("--baseline-window", type=float, nargs=2, default=(-0.1, 0.0), metavar=("START", "STOP"))
    parser.add_argument("--effect-window", type=float, nargs=2, default=(0.1, 0.8), metavar=("START", "STOP"))
    parser.add_argument("--n-permutations", type=int, default=10_000)
    parser.add_argument("--random-state", type=int, default=13)
    args = parser.parse_args()

    baseline_window = tuple(args.baseline_window)
    effect_window = tuple(args.effect_window)
    subject_metrics = subject_decoder_metrics(
        _expand_paths(args.csv),
        chance=args.chance,
        baseline_window=baseline_window,
        effect_window=effect_window,
    )
    statistics = paired_decoder_statistics(
        subject_metrics,
        n_permutations=args.n_permutations,
        random_state=args.random_state,
    )
    if args.out_csv is not None:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        statistics.to_csv(args.out_csv, index=False)
        print(f"Wrote paired statistics CSV: {args.out_csv}")

    report = build_paired_stats_report(
        statistics,
        baseline_window=baseline_window,
        effect_window=effect_window,
        chance=args.chance,
    )
    if args.out_report is not None:
        args.out_report.parent.mkdir(parents=True, exist_ok=True)
        args.out_report.write_text(report, encoding="utf-8")
        print(f"Wrote paired statistics report: {args.out_report}")
    elif args.out_csv is None:
        print(report)


if __name__ == "__main__":
    main()
