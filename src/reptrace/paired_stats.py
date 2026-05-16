from __future__ import annotations

import argparse
import glob
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from reptrace.results import (
    DEFAULT_ECE_BINS,
    SUMMARY_GROUP_COLUMNS,
    read_time_decode_observations,
    read_time_decode_results,
    subject_time_metrics,
)


METRIC_DIRECTIONS = {
    "effect_accuracy": "higher",
    "effect_minus_baseline": "higher",
    "baseline_abs_delta": "lower",
    "effect_log_loss": "lower",
    "effect_brier": "lower",
    "effect_ece": "lower",
}
PAIRING_GROUP_DEFAULTS = {"emission_mode": "calibrated"}


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


def _normalise_emission_mode(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with an explicit, string-valued emission-mode column."""
    normalised = frame.copy()
    if "emission_mode" not in normalised.columns:
        normalised["emission_mode"] = PAIRING_GROUP_DEFAULTS["emission_mode"]
    normalised["emission_mode"] = normalised["emission_mode"].fillna(PAIRING_GROUP_DEFAULTS["emission_mode"]).astype(str)
    return normalised


def _as_tuple(value: object) -> tuple[object, ...]:
    if isinstance(value, tuple):
        return value
    return (value,)


def subject_decoder_metrics(
    csv_paths: list[Path],
    *,
    chance: float = 0.5,
    baseline_window: tuple[float, float] = (-0.1, 0.0),
    effect_window: tuple[float, float] = (0.1, 0.8),
    observation_csv_paths: list[Path] | None = None,
    observation_subject_column: str | None = None,
    ece_bins: int = DEFAULT_ECE_BINS,
) -> pd.DataFrame:
    """Return one row per subject, decoder, and emission mode with paired-test metrics.

    ``effect_ece`` is included only when probability observations are supplied,
    because ECE is nonlinear and must be recomputed from pooled held-out
    probabilities rather than averaged across folds.
    """
    results = read_time_decode_results(csv_paths)
    if "decoder" not in results.columns:
        raise ValueError("Subject CSVs must contain a 'decoder' column.")
    results = _normalise_emission_mode(results)

    observations = None
    metric_columns = ["accuracy", "log_loss", "brier"]
    if observation_csv_paths is not None:
        observations = read_time_decode_observations(
            observation_csv_paths,
            subject_column=observation_subject_column,
            result_csv_paths=csv_paths,
            results=results,
        )
        metric_columns.append("ece")

    group_columns = [column for column in SUMMARY_GROUP_COLUMNS if column in results.columns]
    subject_time_keys = [*group_columns, "subject", "time"]
    subject_group_keys = [*group_columns, "subject"]
    subject_time = subject_time_metrics(
        results,
        observations=observations,
        metric_columns=metric_columns,
        ece_bins=ece_bins,
    ).sort_values(subject_time_keys)

    rows = []
    for key, frame in subject_time.groupby(subject_group_keys, sort=True):
        group_values = dict(zip(subject_group_keys, _as_tuple(key)))
        baseline_accuracy = _window_mean(frame, "accuracy", *baseline_window)
        effect_accuracy = _window_mean(frame, "accuracy", *effect_window)
        row = {
            **{column: str(group_values[column]) for column in group_columns},
            "subject": str(group_values["subject"]),
            "baseline_accuracy": baseline_accuracy,
            "baseline_abs_delta": abs(baseline_accuracy - chance),
            "effect_accuracy": effect_accuracy,
            "effect_minus_baseline": effect_accuracy - baseline_accuracy,
            "effect_log_loss": _window_mean(frame, "log_loss", *effect_window),
            "effect_brier": _window_mean(frame, "brier", *effect_window),
        }
        if "ece" in frame.columns:
            row["effect_ece"] = _window_mean(frame, "ece", *effect_window)
        rows.append(row)
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
    metrics: tuple[str, ...] | None = None,
    n_permutations: int = 10_000,
    random_state: int = 13,
) -> pd.DataFrame:
    """Compare decoders with subject-level paired sign-flip tests.

    Decoder comparisons are stratified by emission mode so calibrated and
    uncalibrated subject metrics are never merged into the same paired test.
    When ``metrics`` is omitted, only metrics present in ``subject_metrics`` are
    tested; this prevents fold-averaged ECE from being tested by default.
    """
    if metrics is None:
        metrics = tuple(metric for metric in METRIC_DIRECTIONS if metric in subject_metrics.columns)
    if not metrics:
        raise ValueError("No paired-statistic metric columns are available.")

    required = {"decoder", "subject", *metrics}
    missing = sorted(required.difference(subject_metrics.columns))
    if missing:
        raise ValueError(f"Subject metrics are missing required columns: {missing}")

    subject_metrics = _normalise_emission_mode(subject_metrics)
    subject_metrics["decoder"] = subject_metrics["decoder"].astype(str)
    subject_metrics["subject"] = subject_metrics["subject"].astype(str)
    pairing_columns = ["emission_mode"]
    identity_columns = [*pairing_columns, "decoder", "subject"]
    duplicates = subject_metrics.duplicated(identity_columns, keep=False)
    if duplicates.any():
        duplicate_keys = subject_metrics.loc[duplicates, identity_columns].drop_duplicates().to_dict("records")
        raise ValueError(
            "Subject metrics must contain at most one row per emission mode, decoder, and subject. "
            f"Duplicate keys: {duplicate_keys}"
        )

    rows = []
    for emission_mode, group in subject_metrics.groupby("emission_mode", sort=True):
        decoders = sorted(group["decoder"].unique())
        if len(decoders) < 2:
            continue
        for decoder_a, decoder_b in itertools.combinations(decoders, 2):
            left = group[group["decoder"] == decoder_a]
            right = group[group["decoder"] == decoder_b]
            paired = left.merge(right, on=[*pairing_columns, "subject"], suffixes=("_a", "_b"))
            if len(paired) < 2:
                raise ValueError(
                    f"Need at least two paired subjects for {decoder_a} vs {decoder_b} "
                    f"in emission mode {emission_mode}."
                )
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
                        "emission_mode": str(emission_mode),
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

    if not rows:
        raise ValueError("Need at least two decoders within an emission mode for paired comparison.")
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
        "| Emission mode | Decoder A | Decoder B | Metric | Preferred | Subjects | A mean | B mean | A minus B | Sign-flip p | Better by mean |",
        "| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in statistics.itertuples(index=False):
        lines.append(
            f"| {row.emission_mode} | {row.decoder_a} | {row.decoder_b} | {row.metric} | {row.preferred_direction} | {row.n_subjects} | "
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
    parser.add_argument("--observations", nargs="+", help="Probability-observation CSVs used for exact effect_ece.")
    parser.add_argument("--observation-subject-column", help="Subject column for --observations.")
    parser.add_argument("--ece-bins", type=int, default=DEFAULT_ECE_BINS, help="Number of bins for exact observation-level ECE.")
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
        observation_csv_paths=_expand_paths(args.observations) if args.observations else None,
        observation_subject_column=args.observation_subject_column,
        ece_bins=args.ece_bins,
    )
    statistics = paired_decoder_statistics(subject_metrics, n_permutations=args.n_permutations, random_state=args.random_state)
    if args.out_csv is not None:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        statistics.to_csv(args.out_csv, index=False)
        print(f"Wrote paired statistics CSV: {args.out_csv}")

    report = build_paired_stats_report(statistics, baseline_window=baseline_window, effect_window=effect_window, chance=args.chance)
    if args.out_report is not None:
        args.out_report.parent.mkdir(parents=True, exist_ok=True)
        args.out_report.write_text(report, encoding="utf-8")
        print(f"Wrote paired statistics report: {args.out_report}")
    elif args.out_csv is None:
        print(report)


if __name__ == "__main__":
    main()
