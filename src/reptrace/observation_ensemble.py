from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss

from reptrace.metrics import brier_score_multiclass, expected_calibration_error
from reptrace.observation_schema import read_validated_probability_observations
from reptrace.observations import ProbabilityObservationTable, probability_columns, stable_hash

DEFAULT_BASELINE_WINDOW = (-0.35, -0.05)
DEFAULT_BASELINE_GROUP_COLUMNS = ("subject", "fold")
DEFAULT_DECODERS = ("logistic", "linear_svm")
DEFAULT_WEIGHTS = (0.5, 0.5)
DEFAULT_ENSEMBLE_DECODER = "baseline_debiased_logistic_linear_svm_ensemble"
DEFAULT_ENSEMBLE_EMISSION_MODE = "baseline_debiased_ensemble"
DEFAULT_MIN_PROBABILITY = 1e-12

_REQUIRED_VALUE_COLUMNS = ("time", "true_label")
_BASE_ALIGNMENT_COLUMNS = (
    "subject",
    "session",
    "stream_id",
    "fold",
    "split_id",
    "seed",
    "train_time",
    "test_time",
    "time",
    "window_start",
    "window_stop",
    "sample_index",
    "sequence_id",
    "true_label",
    "true_class",
    "group",
)
_METRIC_GROUP_COLUMNS = ("subject", "fold", "decoder", "emission_mode", "time", "window_start", "window_stop")


def _normalize_weights(weights: Sequence[float], n_decoders: int) -> np.ndarray:
    if len(weights) != n_decoders:
        raise ValueError(f"Expected {n_decoders} ensemble weights, got {len(weights)}.")
    values = np.asarray(weights, dtype=float)
    if not np.isfinite(values).all() or (values < 0).any() or float(values.sum()) <= 0.0:
        raise ValueError("Ensemble weights must be finite non-negative values with positive sum.")
    return values / values.sum()


def _class_suffixes(prob_columns: Sequence[str]) -> tuple[str, ...]:
    return tuple(column.removeprefix("prob_class_") for column in prob_columns)


def _class_columns_for_probabilities(frame: pd.DataFrame, prob_columns: Sequence[str]) -> tuple[str, ...]:
    return tuple(column for column in (f"class_{suffix}" for suffix in _class_suffixes(prob_columns)) if column in frame.columns)


def _label_values(prob_columns: Sequence[str]) -> tuple[int, ...]:
    suffixes = _class_suffixes(prob_columns)
    if not all(suffix.isdigit() for suffix in suffixes):
        return tuple(range(len(prob_columns)))
    return tuple(int(suffix) for suffix in suffixes)


def _label_positions(labels: Sequence[object] | np.ndarray | pd.Series, label_values: Sequence[int]) -> np.ndarray:
    numeric = pd.to_numeric(pd.Series(labels), errors="coerce")
    if numeric.isna().any():
        raise ValueError("true_label values must be numeric.")
    label_to_position = {int(label): position for position, label in enumerate(label_values)}
    positions = np.full(len(numeric), -1, dtype=int)
    for row_index, label in enumerate(numeric.astype(int).to_numpy()):
        position = label_to_position.get(int(label))
        if position is not None:
            positions[row_index] = position
    if bool((positions < 0).any()):
        missing = sorted(set(int(label) for label in numeric.astype(int).to_numpy() if int(label) not in label_to_position))
        raise ValueError(f"true_label values must index probability labels {list(label_values)}; missing labels: {missing[:5]}")
    return positions


def _alignment_columns(frame: pd.DataFrame, prob_columns: Sequence[str]) -> list[str]:
    class_columns = _class_columns_for_probabilities(frame, prob_columns)
    ordered = [column for column in (*_BASE_ALIGNMENT_COLUMNS, *class_columns) if column in frame.columns]
    if not ordered:
        raise ValueError("Observation rows do not contain any alignment columns.")
    return ordered


def _source_frames(
    observations: pd.DataFrame,
    *,
    decoders: Sequence[str],
    source_emission_mode: str | None,
) -> dict[str, pd.DataFrame]:
    missing = [column for column in ("decoder", *_REQUIRED_VALUE_COLUMNS) if column not in observations.columns]
    if missing:
        raise ValueError(f"Observation table is missing required columns: {missing}")
    working = observations.copy()
    working["decoder"] = working["decoder"].astype(str)
    if source_emission_mode is not None:
        if "emission_mode" not in working.columns:
            raise ValueError("source_emission_mode filtering requires an 'emission_mode' column.")
        working = working.loc[working["emission_mode"].astype(str) == source_emission_mode].copy()
        if working.empty:
            raise ValueError(f"No observation rows remain after filtering emission_mode == {source_emission_mode!r}.")

    sources: dict[str, pd.DataFrame] = {}
    for decoder in decoders:
        subset = working.loc[working["decoder"] == decoder].copy()
        if subset.empty:
            raise ValueError(f"No observations found for decoder {decoder!r}.")
        sources[decoder] = subset
    return sources


def _check_unique_alignment(subset: pd.DataFrame, keys: Sequence[str], decoder: str) -> None:
    duplicate_count = int(subset.duplicated(list(keys), keep=False).sum())
    if duplicate_count:
        examples = subset.loc[subset.duplicated(list(keys), keep=False), list(keys)].head(5).to_dict("records")
        raise ValueError(f"Decoder {decoder!r} has {duplicate_count} rows with duplicate alignment keys. Examples: {examples}")


def _align_probability_matrices(
    sources: dict[str, pd.DataFrame],
    *,
    prob_columns: Sequence[str],
    alignment_columns: Sequence[str],
) -> tuple[pd.DataFrame, list[np.ndarray]]:
    decoders = list(sources)
    base = sources[decoders[0]].copy().reset_index(drop=True)
    _check_unique_alignment(base, alignment_columns, decoders[0])
    aligned = base.loc[:, list(alignment_columns)].copy()
    matrices: list[np.ndarray] = []

    for decoder in decoders:
        subset = sources[decoder].copy()
        _check_unique_alignment(subset, alignment_columns, decoder)
        overlap = base.loc[:, list(alignment_columns)].merge(
            subset.loc[:, list(alignment_columns)].drop_duplicates(),
            on=list(alignment_columns),
            how="outer",
            indicator=True,
        )
        if not bool(overlap["_merge"].eq("both").all()):
            examples = overlap.loc[overlap["_merge"] != "both", [*alignment_columns, "_merge"]].head(5).to_dict("records")
            raise ValueError(f"Decoder {decoder!r} does not align one-to-one with decoder {decoders[0]!r}. Examples: {examples}")

        renamed = subset.loc[:, [*alignment_columns, *prob_columns]].rename(columns={column: f"{column}__{decoder}" for column in prob_columns})
        aligned = aligned.merge(renamed, on=list(alignment_columns), how="left", validate="one_to_one")
        decoder_prob_columns = [f"{column}__{decoder}" for column in prob_columns]
        if aligned.loc[:, decoder_prob_columns].isna().any().any():
            raise ValueError(f"Missing aligned probability values for decoder {decoder!r}.")
        probabilities = aligned.loc[:, decoder_prob_columns].to_numpy(dtype=float)
        if not np.isfinite(probabilities).all():
            raise ValueError(f"Probability values for decoder {decoder!r} must be finite.")
        matrices.append(probabilities)
    return base, matrices


def _baseline_offsets(
    base: pd.DataFrame,
    log_scores: np.ndarray,
    *,
    baseline_window: tuple[float, float] | None,
    baseline_group_columns: Sequence[str],
) -> tuple[np.ndarray, int]:
    if baseline_window is None:
        return np.zeros_like(log_scores), 0
    start, stop = (float(baseline_window[0]), float(baseline_window[1]))
    if stop < start:
        raise ValueError("baseline_window stop must be greater than or equal to start.")
    times = pd.to_numeric(base["time"], errors="coerce")
    if times.isna().any():
        raise ValueError("Observation column 'time' must be numeric for baseline debiasing.")
    baseline_mask = ((times >= start) & (times <= stop)).to_numpy(dtype=bool)
    n_baseline = int(baseline_mask.sum())
    if n_baseline == 0:
        raise ValueError(f"No observations fall inside baseline window [{start}, {stop}].")

    global_offset = log_scores[baseline_mask].mean(axis=0)
    offsets = np.tile(global_offset, (len(base), 1))
    group_columns = [column for column in baseline_group_columns if column in base.columns]
    if not group_columns:
        return offsets, n_baseline

    score_columns = [f"__baseline_score_{class_index}" for class_index in range(log_scores.shape[1])]
    baseline_scores = pd.concat(
        [
            base.loc[baseline_mask, group_columns].reset_index(drop=True),
            pd.DataFrame(log_scores[baseline_mask], columns=score_columns),
        ],
        axis=1,
    )
    group_offsets = baseline_scores.groupby(group_columns, dropna=False, sort=False)[score_columns].mean().reset_index()
    row_offsets = base.loc[:, group_columns].reset_index(drop=True).merge(group_offsets, on=group_columns, how="left", sort=False)
    offset_values = row_offsets.loc[:, score_columns].to_numpy(dtype=float)
    has_group_offset = np.isfinite(offset_values).all(axis=1)
    offsets[has_group_offset] = offset_values[has_group_offset]
    return offsets, n_baseline


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def ensemble_probability_observations(
    observations: pd.DataFrame,
    *,
    decoders: Sequence[str] = DEFAULT_DECODERS,
    weights: Sequence[float] = DEFAULT_WEIGHTS,
    source_emission_mode: str | None = "calibrated",
    baseline_window: tuple[float, float] | None = DEFAULT_BASELINE_WINDOW,
    baseline_group_columns: Sequence[str] = DEFAULT_BASELINE_GROUP_COLUMNS,
    min_probability: float = DEFAULT_MIN_PROBABILITY,
    output_decoder: str = DEFAULT_ENSEMBLE_DECODER,
    output_emission_mode: str = DEFAULT_ENSEMBLE_EMISSION_MODE,
) -> pd.DataFrame:
    """Combine calibrated decoder observation rows with baseline-debiased log-probability ensembling."""
    if len(decoders) < 2:
        raise ValueError("At least two source decoders are required for an ensemble.")
    if min_probability <= 0.0 or min_probability >= 1.0:
        raise ValueError("min_probability must lie in (0, 1).")

    prob_columns = probability_columns(observations)
    if not prob_columns:
        raise ValueError("Observation table must contain prob_class_* columns.")
    normalized_weights = _normalize_weights(weights, len(decoders))
    sources = _source_frames(observations, decoders=decoders, source_emission_mode=source_emission_mode)
    alignment_columns = _alignment_columns(observations, prob_columns)
    base, probability_matrices = _align_probability_matrices(sources, prob_columns=prob_columns, alignment_columns=alignment_columns)

    log_scores = np.zeros_like(probability_matrices[0], dtype=float)
    for weight, probabilities in zip(normalized_weights, probability_matrices):
        log_scores += float(weight) * np.log(np.clip(probabilities, min_probability, 1.0))

    offsets, n_baseline = _baseline_offsets(
        base,
        log_scores,
        baseline_window=baseline_window,
        baseline_group_columns=baseline_group_columns,
    )
    probabilities = _softmax(log_scores - offsets)

    output = base.copy()
    label_values = _label_values(prob_columns)
    predicted_positions = probabilities.argmax(axis=1)
    predicted_labels = np.asarray([label_values[position] for position in predicted_positions], dtype=int)
    true_labels = pd.to_numeric(output["true_label"], errors="coerce")
    if true_labels.isna().any():
        raise ValueError("true_label values must be numeric.")
    true_labels_array = true_labels.astype(int).to_numpy()
    label_to_position = {label: position for position, label in enumerate(label_values)}
    true_probabilities = np.full(len(output), np.nan, dtype=float)
    for row_index, true_label in enumerate(true_labels_array):
        position = label_to_position.get(int(true_label))
        if position is not None:
            true_probabilities[row_index] = probabilities[row_index, position]

    class_columns = _class_columns_for_probabilities(output, prob_columns)
    predicted_classes: list[str] = []
    for row_index, predicted_label in enumerate(predicted_labels):
        class_column = f"class_{predicted_label}"
        if class_column in output.columns:
            predicted_classes.append(str(output.loc[row_index, class_column]))
        elif class_columns:
            predicted_classes.append(str(output.loc[row_index, class_columns[predicted_positions[row_index]]]))
        else:
            predicted_classes.append(str(predicted_label))

    for column_index, column in enumerate(prob_columns):
        output[column] = probabilities[:, column_index]
    output["decoder"] = output_decoder
    output["backend"] = "ensemble"
    output["emission_mode"] = output_emission_mode
    output["predicted_label"] = predicted_labels
    output["predicted_class"] = predicted_classes
    output["probability_true_class"] = true_probabilities
    output["confidence"] = probabilities.max(axis=1)
    output["is_correct"] = predicted_labels == true_labels_array
    output["calibration_fold"] = "baseline_window" if baseline_window is not None else ""
    output["source_decoders"] = "|".join(decoders)
    output["source_emission_mode"] = "" if source_emission_mode is None else source_emission_mode
    output["ensemble_weights"] = "|".join(f"{weight:.12g}" for weight in normalized_weights)
    output["baseline_window_start"] = "" if baseline_window is None else float(baseline_window[0])
    output["baseline_window_stop"] = "" if baseline_window is None else float(baseline_window[1])
    output["baseline_group_columns"] = "|".join(column for column in baseline_group_columns if column in output.columns)
    output["n_baseline_observations"] = n_baseline
    output["model_hash"] = stable_hash(
        {
            "backend": "ensemble",
            "decoders": list(decoders),
            "weights": [float(weight) for weight in normalized_weights],
            "source_emission_mode": source_emission_mode,
            "baseline_window": baseline_window,
            "baseline_group_columns": [column for column in baseline_group_columns if column in observations.columns],
            "min_probability": min_probability,
        }
    )
    return ProbabilityObservationTable(output).standardized(defaults={"backend": "ensemble", "decoder": output_decoder, "emission_mode": output_emission_mode}).frame


def summarize_ensemble_metrics(observations: pd.DataFrame, *, ece_bins: int = 10) -> pd.DataFrame:
    """Summarize ensemble observation rows as time-resolved result metrics."""
    if ece_bins < 1:
        raise ValueError("ece_bins must be positive.")
    prob_columns = probability_columns(observations)
    if "true_label" not in observations.columns or not prob_columns:
        raise ValueError("Ensemble observations must contain true_label and prob_class_* columns.")
    label_values = _label_values(prob_columns)
    group_columns = [column for column in _METRIC_GROUP_COLUMNS if column in observations.columns]
    rows: list[dict[str, object]] = []
    for group_key, group in observations.groupby(group_columns, dropna=False, sort=True):
        if len(group_columns) == 1 and not isinstance(group_key, tuple):
            group_key = (group_key,)
        probabilities = group.loc[:, list(prob_columns)].to_numpy(dtype=float)
        true_labels = pd.to_numeric(group["true_label"], errors="coerce")
        if true_labels.isna().any():
            raise ValueError("true_label values must be numeric.")
        true_label_values = true_labels.astype(int).to_numpy()
        true_positions = _label_positions(true_label_values, label_values)
        prediction_positions = probabilities.argmax(axis=1)
        predicted_label_values = np.asarray([label_values[position] for position in prediction_positions], dtype=int)
        row = dict(zip(group_columns, group_key))
        row.update(
            {
                "accuracy": accuracy_score(true_label_values, predicted_label_values),
                "log_loss": log_loss(true_label_values, probabilities, labels=list(label_values)),
                "brier": brier_score_multiclass(probabilities, true_positions),
                "ece": expected_calibration_error(probabilities, true_positions, n_bins=ece_bins),
                "n_train": "",
                "n_test": int(len(group)),
                "n_classes": int(len(prob_columns)),
                "class_names": "|".join(str(group.iloc[0].get(f"class_{label}", label)) for label in label_values),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _parse_weights(weights: Sequence[float] | None, decoders: Sequence[str]) -> tuple[float, ...]:
    if weights is not None:
        return tuple(float(weight) for weight in weights)
    if tuple(decoders) == DEFAULT_DECODERS:
        return DEFAULT_WEIGHTS
    return tuple(1.0 for _ in decoders)


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point for baseline-debiased probability ensembling."""
    parser = argparse.ArgumentParser(description="Create a baseline-debiased log-probability ensemble from RepTrace observation CSVs.")
    parser.add_argument("observation_csv", nargs="+", type=Path, help="Observation CSVs or glob patterns containing source decoder probability rows.")
    parser.add_argument("--out", type=Path, required=True, help="CSV path for the ensembled probability observations.")
    parser.add_argument("--metrics-out", type=Path, help="Optional time-resolved metrics CSV computed from the ensembled observations.")
    parser.add_argument("--decoder", action="append", dest="decoders", help="Source decoder to ensemble. May be repeated; defaults to logistic and linear_svm.")
    parser.add_argument("--weight", action="append", type=float, dest="weights", help="Source decoder weight. May be repeated in the same order as --decoder.")
    parser.add_argument("--source-emission-mode", default="calibrated", help="Source emission_mode to use before ensembling. Defaults to calibrated.")
    parser.add_argument("--no-source-emission-filter", action="store_true", help="Use all source emission modes instead of filtering by --source-emission-mode.")
    parser.add_argument("--baseline-window", nargs=2, type=float, metavar=("START", "STOP"), default=DEFAULT_BASELINE_WINDOW)
    parser.add_argument("--no-baseline-debiasing", action="store_true", help="Disable pre-stimulus log-probability offset removal.")
    parser.add_argument("--baseline-group-column", action="append", dest="baseline_group_columns", help="Column used for baseline-offset grouping. May be repeated; defaults to subject and fold.")
    parser.add_argument("--min-probability", type=float, default=DEFAULT_MIN_PROBABILITY, help="Lower clipping bound before taking log probabilities.")
    parser.add_argument("--output-decoder", default=DEFAULT_ENSEMBLE_DECODER)
    parser.add_argument("--output-emission-mode", default=DEFAULT_ENSEMBLE_EMISSION_MODE)
    parser.add_argument("--ece-bins", type=int, default=10, help="Number of ECE bins for --metrics-out.")
    parser.add_argument("--probability-tolerance", type=float, default=1e-3, help="Input probability row-sum tolerance.")
    args = parser.parse_args(argv)

    try:
        decoders = tuple(args.decoders or DEFAULT_DECODERS)
        weights = _parse_weights(args.weights, decoders)
        source_emission_mode = None if args.no_source_emission_filter else args.source_emission_mode
        baseline_window = None if args.no_baseline_debiasing else tuple(float(value) for value in args.baseline_window)
        baseline_group_columns = tuple(args.baseline_group_columns or DEFAULT_BASELINE_GROUP_COLUMNS)

        observations = read_validated_probability_observations(
            args.observation_csv,
            profile="generic",
            probability_tolerance=args.probability_tolerance,
            require_normalized=True,
        )
        ensemble = ensemble_probability_observations(
            observations,
            decoders=decoders,
            weights=weights,
            source_emission_mode=source_emission_mode,
            baseline_window=baseline_window,
            baseline_group_columns=baseline_group_columns,
            min_probability=args.min_probability,
            output_decoder=args.output_decoder,
            output_emission_mode=args.output_emission_mode,
        )
        ProbabilityObservationTable(ensemble).to_csv(args.out)
        print(f"Wrote ensemble observations: {args.out}")
        if args.metrics_out is not None:
            metrics = summarize_ensemble_metrics(ensemble, ece_bins=args.ece_bins)
            args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
            metrics.to_csv(args.metrics_out, index=False)
            print(f"Wrote ensemble metrics: {args.metrics_out}")
    except Exception as exc:
        print(f"Observation ensembling failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
