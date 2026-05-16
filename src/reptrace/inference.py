from __future__ import annotations

import argparse
import glob
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
    "accuracy": "higher",
    "log_loss": "lower",
    "brier": "lower",
    "ece": "lower",
}
METRIC_DIRECTION_CHOICES = ("auto", "higher", "lower")


def _expand_paths(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if matches:
            paths.extend(Path(match) for match in matches)
        else:
            paths.append(Path(pattern))
    return paths


def _condition_filters(decoder: str | None, emission_mode: str | None) -> dict[str, str]:
    filters: dict[str, str] = {}
    if decoder is not None:
        filters["decoder"] = str(decoder)
    if emission_mode is not None:
        filters["emission_mode"] = str(emission_mode)
    return filters


def _display_values(series: pd.Series) -> list[str]:
    values = series.astype(object).where(series.notna(), "<NA>").astype(str).drop_duplicates()
    return sorted(values.tolist())


def _filter_results(results: pd.DataFrame, filters: dict[str, str]) -> pd.DataFrame:
    filtered = results
    for column, value in filters.items():
        if column not in filtered.columns:
            raise ValueError(f"Cannot filter by '{column}': column is not present in the result CSVs.")
        values = filtered[column].astype(object).where(filtered[column].notna(), "<NA>").astype(str)
        mask = values == value
        if not bool(mask.any()):
            available = ", ".join(_display_values(filtered[column])) or "<none>"
            raise ValueError(f"No rows match {column}={value!r}. Available values: {available}.")
        filtered = filtered.loc[mask].copy()
    return filtered


def _filter_available_conditions(frame: pd.DataFrame, filters: dict[str, str]) -> pd.DataFrame:
    filtered = frame
    for column, value in filters.items():
        if column not in filtered.columns:
            continue
        values = filtered[column].astype(object).where(filtered[column].notna(), "<NA>").astype(str)
        mask = values == value
        if not bool(mask.any()):
            available = ", ".join(_display_values(filtered[column])) or "<none>"
            raise ValueError(f"No observation rows match {column}={value!r}. Available values: {available}.")
        filtered = filtered.loc[mask].copy()
    return filtered


def _single_condition_values(results: pd.DataFrame, filters: dict[str, str]) -> dict[str, str]:
    condition_values: dict[str, str] = {}
    for column in SUMMARY_GROUP_COLUMNS:
        if column not in results.columns:
            continue
        values = _display_values(results[column])
        if len(values) > 1:
            option_name = column.replace("_", "-")
            raise ValueError(
                f"Result CSVs contain multiple {column} values ({', '.join(values)}). "
                f"Run inference for one condition at a time, for example with --{option_name}."
            )
        if values:
            condition_values[column] = filters.get(column, values[0])
    return condition_values


def _resolve_metric_direction(metric: str, metric_direction: str = "auto") -> str:
    normalized = metric_direction.lower().replace("_", "-")
    if normalized not in METRIC_DIRECTION_CHOICES:
        raise ValueError(f"metric_direction must be one of {METRIC_DIRECTION_CHOICES}.")
    if normalized == "auto":
        return METRIC_DIRECTIONS.get(metric, "higher")
    return normalized


def _signed_effects(matrix: pd.DataFrame, *, reference_value: float, metric_direction: str) -> pd.DataFrame:
    if metric_direction == "higher":
        return matrix - reference_value
    if metric_direction == "lower":
        return reference_value - matrix
    raise ValueError(f"Unsupported metric_direction: {metric_direction!r}")


def _subject_time_effects_and_conditions(
    csv_paths: list[Path],
    *,
    metric: str,
    chance: float,
    decoder: str | None,
    emission_mode: str | None,
    observation_csv_paths: list[Path] | None,
    observation_subject_column: str | None,
    ece_bins: int,
    metric_direction: str,
) -> tuple[pd.DataFrame, dict[str, str], str]:
    all_results = read_time_decode_results(csv_paths)
    if metric not in all_results.columns:
        raise ValueError(f"Metric column '{metric}' not found in results.")
    if metric == "ece" and observation_csv_paths is None:
        raise ValueError(
            "Exact ECE inference requires probability observation CSVs. "
            "Pass observation_csv_paths or the CLI --observations option."
        )

    filters = _condition_filters(decoder, emission_mode)
    results = _filter_results(all_results, filters)
    condition_values = _single_condition_values(results, filters)
    resolved_direction = _resolve_metric_direction(metric, metric_direction)

    observations = None
    if observation_csv_paths is not None:
        observations = read_time_decode_observations(
            observation_csv_paths,
            subject_column=observation_subject_column,
            result_csv_paths=csv_paths,
            results=all_results,
        )
        observations = _filter_available_conditions(observations, filters)

    subject_time = subject_time_metrics(
        results,
        observations=observations,
        metric_columns=[metric],
        ece_bins=ece_bins,
    ).sort_values(["subject", "time"])
    matrix = subject_time.pivot(index="subject", columns="time", values=metric).sort_index(axis=0).sort_index(axis=1)
    if matrix.isna().any().any():
        raise ValueError("All subjects must have results for the same time points.")
    return _signed_effects(matrix, reference_value=chance, metric_direction=resolved_direction), condition_values, resolved_direction


def subject_time_effects(
    csv_paths: list[Path],
    *,
    metric: str = "accuracy",
    chance: float = 0.5,
    decoder: str | None = None,
    emission_mode: str | None = None,
    observation_csv_paths: list[Path] | None = None,
    observation_subject_column: str | None = None,
    ece_bins: int = DEFAULT_ECE_BINS,
    metric_direction: str = "auto",
) -> pd.DataFrame:
    """Return a subject-by-time matrix of signed effects against a reference value.

    The sign is chosen so positive values mean better than the reference:
    higher-is-better metrics use metric minus reference, while lower-is-better
    metrics use reference minus metric. ECE inference uses pooled held-out
    probability observations. Pass observation_csv_paths when metric='ece';
    fold-averaged ECE is not used for inferential tests.
    """
    effects, _, _ = _subject_time_effects_and_conditions(
        csv_paths,
        metric=metric,
        chance=chance,
        decoder=decoder,
        emission_mode=emission_mode,
        observation_csv_paths=observation_csv_paths,
        observation_subject_column=observation_subject_column,
        ece_bins=ece_bins,
        metric_direction=metric_direction,
    )
    return effects


def _t_statistic(effects: np.ndarray) -> np.ndarray:
    if effects.shape[0] < 2:
        raise ValueError("Need at least two subjects for subject-level inference.")
    means = effects.mean(axis=0)
    sem = effects.std(axis=0, ddof=1) / np.sqrt(effects.shape[0])
    return np.divide(means, sem, out=np.zeros_like(means), where=sem > 0)


def _sign_flip_t_statistics(effects: np.ndarray, *, n_permutations: int, random_state: int) -> np.ndarray:
    if n_permutations < 1:
        raise ValueError("n_permutations must be at least 1.")
    rng = np.random.default_rng(random_state)
    n_subjects = effects.shape[0]
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_permutations, n_subjects))
    means = signs @ effects / n_subjects
    sum_squares = np.sum(effects**2, axis=0)
    variances = (sum_squares[None, :] - n_subjects * means**2) / (n_subjects - 1)
    sem = np.sqrt(np.maximum(variances, 0.0) / n_subjects)
    return np.divide(means, sem, out=np.zeros_like(means), where=sem > 0)


def _contiguous_clusters(mask: np.ndarray) -> list[tuple[int, int]]:
    clusters: list[tuple[int, int]] = []
    start: int | None = None
    for idx, value in enumerate(mask):
        if value and start is None:
            start = idx
        elif not value and start is not None:
            clusters.append((start, idx))
            start = None
    if start is not None:
        clusters.append((start, len(mask)))
    return clusters


def _cluster_masses(statistics: np.ndarray, threshold: np.ndarray) -> list[float]:
    masses = []
    for start, stop in _contiguous_clusters(statistics >= threshold):
        masses.append(float(statistics[start:stop].sum()))
    return masses


def _prepend_condition_columns(table: pd.DataFrame, condition_values: dict[str, str]) -> pd.DataFrame:
    if table.empty and not table.columns.tolist():
        table = pd.DataFrame(columns=[])
    for column, value in reversed(list(condition_values.items())):
        table.insert(0, column, value)
    return table


def sign_flip_time_inference(
    csv_paths: list[Path],
    *,
    metric: str = "accuracy",
    chance: float = 0.5,
    n_permutations: int = 10_000,
    random_state: int = 13,
    cluster_alpha: float = 0.05,
    decoder: str | None = None,
    emission_mode: str | None = None,
    observation_csv_paths: list[Path] | None = None,
    observation_subject_column: str | None = None,
    ece_bins: int = DEFAULT_ECE_BINS,
    metric_direction: str = "auto",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run one-sided subject-level sign-flip inference over time.

    The test uses fold-size-weighted subject time courses as independent
    samples. Pointwise p-values test whether the metric is better than the
    reference value ``chance``. Higher-is-better metrics use ``metric - chance``;
    lower-is-better metrics use ``chance - metric``. Cluster p-values use a
    max-cluster-mass correction over contiguous positive-effect time points.
    """
    if not 0 < cluster_alpha < 1:
        raise ValueError("cluster_alpha must be between 0 and 1.")

    effects, condition_values, resolved_direction = _subject_time_effects_and_conditions(
        csv_paths,
        metric=metric,
        chance=chance,
        decoder=decoder,
        emission_mode=emission_mode,
        observation_csv_paths=observation_csv_paths,
        observation_subject_column=observation_subject_column,
        ece_bins=ece_bins,
        metric_direction=metric_direction,
    )
    effect_values = effects.to_numpy(dtype=float)
    times = effects.columns.to_numpy(dtype=float)
    n_subjects = effect_values.shape[0]

    observed_statistic = _t_statistic(effect_values)
    observed_effect = effect_values.mean(axis=0)
    null_statistics = _sign_flip_t_statistics(effect_values, n_permutations=n_permutations, random_state=random_state)

    pointwise_p = (1.0 + (null_statistics >= observed_statistic[None, :]).sum(axis=0)) / (n_permutations + 1.0)
    cluster_threshold = np.quantile(null_statistics, 1.0 - cluster_alpha, axis=0)

    observed_clusters = _contiguous_clusters(observed_statistic >= cluster_threshold)
    max_null_masses = np.zeros(n_permutations)
    for idx, null_statistic in enumerate(null_statistics):
        masses = _cluster_masses(null_statistic, cluster_threshold)
        max_null_masses[idx] = max(masses) if masses else 0.0

    cluster_ids = np.full(len(times), -1, dtype=int)
    cluster_p_values = np.full(len(times), np.nan, dtype=float)
    cluster_rows = []
    for cluster_id, (start, stop) in enumerate(observed_clusters, start=1):
        cluster_statistic = observed_statistic[start:stop]
        cluster_mass = float(cluster_statistic.sum())
        cluster_p = float((1.0 + np.sum(max_null_masses >= cluster_mass)) / (n_permutations + 1.0))
        peak_offset = int(np.argmax(cluster_statistic))
        cluster_ids[start:stop] = cluster_id
        cluster_p_values[start:stop] = cluster_p
        cluster_rows.append(
            {
                "cluster_id": cluster_id,
                "start_time": float(times[start]),
                "stop_time": float(times[stop - 1]),
                "peak_time": float(times[start + peak_offset]),
                "n_timepoints": stop - start,
                "cluster_mass": cluster_mass,
                "peak_statistic": float(cluster_statistic[peak_offset]),
                "cluster_p": cluster_p,
            }
        )

    time_table = pd.DataFrame(
        {
            "time": times,
            "n_subjects": n_subjects,
            "metric_direction": resolved_direction,
            "reference_value": chance,
            f"{metric}_mean": observed_effect + chance if resolved_direction == "higher" else chance - observed_effect,
            "effect_mean": observed_effect,
            "statistic": observed_statistic,
            "pointwise_p": pointwise_p,
            "cluster_threshold": cluster_threshold,
            "cluster_id": cluster_ids,
            "cluster_p": cluster_p_values,
        }
    )
    cluster_table = pd.DataFrame(
        cluster_rows,
        columns=[
            "cluster_id",
            "start_time",
            "stop_time",
            "peak_time",
            "n_timepoints",
            "cluster_mass",
            "peak_statistic",
            "cluster_p",
        ],
    )
    cluster_table["metric_direction"] = resolved_direction
    cluster_table["reference_value"] = chance
    return _prepend_condition_columns(time_table, condition_values), _prepend_condition_columns(cluster_table, condition_values)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run subject-level sign-flip inference for time-resolved decoding results.")
    parser.add_argument("csv", nargs="+", help="Subject result CSVs or glob patterns.")
    parser.add_argument("--metric", default="accuracy")
    parser.add_argument("--chance", type=float, default=0.5, help="Reference value for the selected metric.")
    parser.add_argument("--metric-direction", choices=METRIC_DIRECTION_CHOICES, default="auto", help="Whether larger or smaller metric values are better; auto knows accuracy/log_loss/brier/ece.")
    parser.add_argument("--n-permutations", type=int, default=10_000)
    parser.add_argument("--random-state", type=int, default=13)
    parser.add_argument("--cluster-alpha", type=float, default=0.05)
    parser.add_argument("--decoder", help="Decoder value to analyze when the CSVs contain multiple decoders.")
    parser.add_argument("--emission-mode", help="Emission mode to analyze when the CSVs contain multiple emission modes.")
    parser.add_argument("--observations", nargs="+", help="Probability-observation CSVs used for exact ECE inference.")
    parser.add_argument("--observation-subject-column", help="Subject column for --observations.")
    parser.add_argument("--ece-bins", type=int, default=DEFAULT_ECE_BINS, help="Number of bins for exact observation-level ECE.")
    parser.add_argument("--out-time", type=Path, required=True)
    parser.add_argument("--out-clusters", type=Path, required=True)
    args = parser.parse_args()

    time_table, cluster_table = sign_flip_time_inference(
        _expand_paths(args.csv),
        metric=args.metric,
        chance=args.chance,
        n_permutations=args.n_permutations,
        random_state=args.random_state,
        cluster_alpha=args.cluster_alpha,
        decoder=args.decoder,
        emission_mode=args.emission_mode,
        observation_csv_paths=_expand_paths(args.observations) if args.observations else None,
        observation_subject_column=args.observation_subject_column,
        ece_bins=args.ece_bins,
        metric_direction=args.metric_direction,
    )
    args.out_time.parent.mkdir(parents=True, exist_ok=True)
    args.out_clusters.parent.mkdir(parents=True, exist_ok=True)
    time_table.to_csv(args.out_time, index=False)
    cluster_table.to_csv(args.out_clusters, index=False)
    print(f"Wrote time inference CSV: {args.out_time}")
    print(f"Wrote cluster inference CSV: {args.out_clusters}")
    if cluster_table.empty:
        print("No above-threshold clusters found.")
    else:
        best = cluster_table.loc[cluster_table["cluster_p"].idxmin()]
        print(f"Best cluster: {best['start_time']:.3f} to {best['stop_time']:.3f}s, p={best['cluster_p']:.4f}")


if __name__ == "__main__":
    main()
