from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd

EPSILON = 1e-12
MODEL_GROUP_COLUMNS = ("decoder", "emission_mode")


def _expand_paths(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if matches:
            paths.extend(Path(match) for match in matches)
        else:
            paths.append(Path(pattern))
    return paths


def probability_columns(frame: pd.DataFrame) -> list[str]:
    """Return probability-vector columns in class-index order."""
    columns = [column for column in frame.columns if column.startswith("prob_class_")]
    if not columns:
        raise ValueError("Observation CSVs must contain probability columns named 'prob_class_*'.")

    def sort_key(column: str) -> tuple[int, str]:
        suffix = column.removeprefix("prob_class_")
        return (int(suffix), suffix) if suffix.isdigit() else (10_000, suffix)

    return sorted(columns, key=sort_key)


def _class_names(frame: pd.DataFrame, prob_columns: list[str]) -> list[str]:
    names = []
    for index, column in enumerate(prob_columns):
        class_column = f"class_{column.removeprefix('prob_class_')}"
        if class_column in frame.columns and frame[class_column].notna().any():
            names.append(str(frame[class_column].dropna().iloc[0]))
        else:
            names.append(str(index))
    return names


def read_probability_observations(csv_paths: list[Path]) -> pd.DataFrame:
    """Read held-out probability observation CSVs emitted by RepTrace."""
    if not csv_paths:
        raise ValueError("At least one observation CSV path is required.")

    frames = []
    for csv_path in csv_paths:
        frame = pd.read_csv(csv_path)
        missing = [column for column in ("time",) if column not in frame.columns]
        if missing:
            raise ValueError(f"{csv_path} is missing required columns: {missing}")
        probability_columns(frame)
        if "sequence_id" not in frame.columns:
            if "sample_index" not in frame.columns:
                raise ValueError(f"{csv_path} is missing 'sequence_id' or 'sample_index'.")
            frame["sequence_id"] = frame["sample_index"]
        if "subject" not in frame.columns:
            frame["subject"] = csv_path.stem
        if "decoder" not in frame.columns:
            frame["decoder"] = "decoder"
        if "emission_mode" not in frame.columns:
            frame["emission_mode"] = "calibrated"
        frame["subject"] = frame["subject"].astype(str)
        frame["decoder"] = frame["decoder"].astype(str)
        frame["emission_mode"] = frame["emission_mode"].astype(str)
        frame["source_file"] = csv_path.name
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _normalize_probabilities(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(probabilities, dtype=float), EPSILON, 1.0)
    row_sums = clipped.sum(axis=1, keepdims=True)
    return clipped / row_sums


def _filter_time_window(frame: pd.DataFrame, time_window: tuple[float, float] | None) -> pd.DataFrame:
    if time_window is None:
        return frame.copy()
    start, stop = time_window
    return frame.loc[(frame["time"] >= start) & (frame["time"] <= stop)].copy()


def _sequence_key_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in ("subject", "fold", "sequence_id") if column in frame.columns]


def _model_group_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in MODEL_GROUP_COLUMNS if column in frame.columns]


def _sequences_from_frame(frame: pd.DataFrame, prob_columns: list[str]) -> list[np.ndarray]:
    key_columns = _sequence_key_columns(frame)
    sequences = []
    for _, sequence_frame in frame.sort_values([*key_columns, "time"]).groupby(key_columns, sort=True):
        probabilities = _normalize_probabilities(sequence_frame[prob_columns].to_numpy())
        if len(probabilities) > 1:
            sequences.append(probabilities)
    if not sequences:
        raise ValueError("Need at least one sequence with two or more time points for temporal modeling.")
    return sequences


def _logsumexp(values: np.ndarray, axis: int | None = None) -> np.ndarray:
    max_value = np.max(values, axis=axis, keepdims=True)
    stable = np.exp(values - max_value)
    result = max_value + np.log(stable.sum(axis=axis, keepdims=True))
    if axis is None:
        return np.asarray(result.squeeze())
    return np.squeeze(result, axis=axis)


def _log_transition(n_states: int, stay_probability: float) -> np.ndarray:
    if n_states < 2:
        raise ValueError("Need at least two states.")
    stay = float(np.clip(stay_probability, EPSILON, 1.0 - EPSILON))
    switch = (1.0 - stay) / (n_states - 1)
    transition = np.full((n_states, n_states), switch)
    np.fill_diagonal(transition, stay)
    return np.log(np.clip(transition, EPSILON, 1.0))


def _sequence_log_likelihood(probabilities: np.ndarray, log_transition: np.ndarray) -> float:
    n_states = probabilities.shape[1]
    log_emissions = np.log(_normalize_probabilities(probabilities))
    log_alpha = np.full(n_states, -np.log(n_states)) + log_emissions[0]
    for time_index in range(1, len(probabilities)):
        log_alpha = log_emissions[time_index] + _logsumexp(log_alpha[:, None] + log_transition, axis=0)
    return float(_logsumexp(log_alpha))


def _total_log_likelihood(sequences: list[np.ndarray], stay_probability: float) -> float:
    n_states = sequences[0].shape[1]
    log_transition = _log_transition(n_states, stay_probability)
    return float(sum(_sequence_log_likelihood(sequence, log_transition) for sequence in sequences))


def _stay_grid(n_states: int, grid_size: int) -> np.ndarray:
    if grid_size < 2:
        raise ValueError("stay_grid_size must be at least 2.")
    return np.linspace(1.0 / n_states, 0.995, grid_size)


def fit_sticky_switching_model(sequences: list[np.ndarray], *, stay_grid_size: int = 200) -> dict[str, float]:
    """Fit a sticky switching model by grid-searching the state persistence."""
    n_states = sequences[0].shape[1]
    if any(sequence.shape[1] != n_states for sequence in sequences):
        raise ValueError("All probability sequences must have the same number of states.")

    grid = _stay_grid(n_states, stay_grid_size)
    log_likelihoods = np.array([_total_log_likelihood(sequences, stay_probability) for stay_probability in grid])
    best_index = int(np.argmax(log_likelihoods))
    n_observations = int(sum(len(sequence) for sequence in sequences))
    log_likelihood = float(log_likelihoods[best_index])
    uniform_log_likelihood_per_observation = -float(np.log(n_states))
    log_likelihood_per_observation = log_likelihood / n_observations
    return {
        "n_sequences": float(len(sequences)),
        "n_observations": float(n_observations),
        "n_states": float(n_states),
        "best_stay_probability": float(grid[best_index]),
        "log_likelihood": log_likelihood,
        "log_likelihood_per_observation": log_likelihood_per_observation,
        "uniform_log_likelihood_per_observation": uniform_log_likelihood_per_observation,
        "persistence_gain_per_observation": log_likelihood_per_observation - uniform_log_likelihood_per_observation,
    }


def _shuffle_time(sequences: list[np.ndarray], rng: np.random.Generator) -> list[np.ndarray]:
    shuffled = []
    for sequence in sequences:
        order = rng.permutation(len(sequence))
        shuffled.append(sequence[order])
    return shuffled


def _shuffle_probability_labels(sequences: list[np.ndarray], rng: np.random.Generator) -> list[np.ndarray]:
    shuffled = []
    for sequence in sequences:
        permuted = sequence.copy()
        for row_index in range(len(permuted)):
            permuted[row_index] = permuted[row_index, rng.permutation(permuted.shape[1])]
        shuffled.append(permuted)
    return shuffled


def _fit_control(
    sequences: list[np.ndarray],
    *,
    control: str,
    n_permutations: int,
    random_seed: int,
    stay_grid_size: int,
) -> list[dict[str, float]]:
    rng = np.random.default_rng(random_seed)
    rows = []
    for _ in range(n_permutations):
        if control == "shuffled_time":
            control_sequences = _shuffle_time(sequences, rng)
        elif control == "shuffled_label":
            control_sequences = _shuffle_probability_labels(sequences, rng)
        else:
            raise ValueError(f"Unknown control '{control}'.")
        rows.append(fit_sticky_switching_model(control_sequences, stay_grid_size=stay_grid_size))
    return rows


def _model_row(group_values: dict[str, str], condition: str, fit: dict[str, float], *, empirical_p_value: float | None = None) -> dict[str, float | str | None]:
    return {
        **group_values,
        "condition": condition,
        "n_sequences": int(fit["n_sequences"]),
        "n_observations": int(fit["n_observations"]),
        "n_states": int(fit["n_states"]),
        "best_stay_probability": fit["best_stay_probability"],
        "best_stay_probability_sd": None,
        "log_likelihood": fit["log_likelihood"],
        "log_likelihood_per_observation": fit["log_likelihood_per_observation"],
        "log_likelihood_per_observation_sd": None,
        "uniform_log_likelihood_per_observation": fit["uniform_log_likelihood_per_observation"],
        "persistence_gain_per_observation": fit["persistence_gain_per_observation"],
        "persistence_gain_per_observation_sd": None,
        "empirical_p_value": empirical_p_value,
    }


def _control_row(
    group_values: dict[str, str],
    condition: str,
    fits: list[dict[str, float]],
    *,
    observed_gain: float,
) -> dict[str, float | str | None]:
    frame = pd.DataFrame(fits)
    gains = frame["persistence_gain_per_observation"].to_numpy()
    empirical_p_value = (1.0 + float(np.sum(gains >= observed_gain))) / (len(gains) + 1.0)
    return {
        **group_values,
        "condition": condition,
        "n_sequences": int(frame["n_sequences"].iloc[0]),
        "n_observations": int(frame["n_observations"].iloc[0]),
        "n_states": int(frame["n_states"].iloc[0]),
        "best_stay_probability": float(frame["best_stay_probability"].mean()),
        "best_stay_probability_sd": float(frame["best_stay_probability"].std(ddof=1)) if len(frame) > 1 else 0.0,
        "log_likelihood": float(frame["log_likelihood"].mean()),
        "log_likelihood_per_observation": float(frame["log_likelihood_per_observation"].mean()),
        "log_likelihood_per_observation_sd": float(frame["log_likelihood_per_observation"].std(ddof=1)) if len(frame) > 1 else 0.0,
        "uniform_log_likelihood_per_observation": float(frame["uniform_log_likelihood_per_observation"].mean()),
        "persistence_gain_per_observation": float(gains.mean()),
        "persistence_gain_per_observation_sd": float(gains.std(ddof=1)) if len(frame) > 1 else 0.0,
        "empirical_p_value": empirical_p_value,
    }


def _forward_backward(probabilities: np.ndarray, stay_probability: float) -> np.ndarray:
    probabilities = _normalize_probabilities(probabilities)
    n_states = probabilities.shape[1]
    log_emissions = np.log(probabilities)
    log_transition = _log_transition(n_states, stay_probability)

    log_alpha = np.empty_like(log_emissions)
    log_alpha[0] = np.full(n_states, -np.log(n_states)) + log_emissions[0]
    for time_index in range(1, len(probabilities)):
        log_alpha[time_index] = log_emissions[time_index] + _logsumexp(log_alpha[time_index - 1][:, None] + log_transition, axis=0)

    log_beta = np.zeros_like(log_emissions)
    for time_index in range(len(probabilities) - 2, -1, -1):
        log_beta[time_index] = _logsumexp(log_transition + log_emissions[time_index + 1] + log_beta[time_index + 1], axis=1)

    log_normalizer = _logsumexp(log_alpha[-1])
    posterior = np.exp(log_alpha + log_beta - log_normalizer)
    return posterior / posterior.sum(axis=1, keepdims=True)


def _viterbi_path(probabilities: np.ndarray, stay_probability: float) -> np.ndarray:
    probabilities = _normalize_probabilities(probabilities)
    n_states = probabilities.shape[1]
    log_emissions = np.log(probabilities)
    log_transition = _log_transition(n_states, stay_probability)

    scores = np.empty_like(log_emissions)
    backpointers = np.zeros_like(log_emissions, dtype=int)
    scores[0] = np.full(n_states, -np.log(n_states)) + log_emissions[0]
    for time_index in range(1, len(probabilities)):
        candidates = scores[time_index - 1][:, None] + log_transition
        backpointers[time_index] = np.argmax(candidates, axis=0)
        scores[time_index] = log_emissions[time_index] + np.max(candidates, axis=0)

    path = np.zeros(len(probabilities), dtype=int)
    path[-1] = int(np.argmax(scores[-1]))
    for time_index in range(len(probabilities) - 2, -1, -1):
        path[time_index] = backpointers[time_index + 1, path[time_index + 1]]
    return path


def build_state_trace(frame: pd.DataFrame, *, stay_probability: float, class_names: list[str], prob_columns: list[str]) -> pd.DataFrame:
    """Decode posterior and Viterbi state traces for observed probability sequences."""
    key_columns = _sequence_key_columns(frame)
    rows = []
    for key, sequence_frame in frame.sort_values([*key_columns, "time"]).groupby(key_columns, sort=True):
        key_values = key if isinstance(key, tuple) else (key,)
        metadata = dict(zip(key_columns, key_values, strict=True))
        probabilities = _normalize_probabilities(sequence_frame[prob_columns].to_numpy())
        posterior = _forward_backward(probabilities, stay_probability)
        viterbi = _viterbi_path(probabilities, stay_probability)
        for row_index, (_, observation) in enumerate(sequence_frame.iterrows()):
            state = int(viterbi[row_index])
            row = {
                **metadata,
                "decoder": str(observation["decoder"]),
                "emission_mode": str(observation["emission_mode"]) if "emission_mode" in observation else "calibrated",
                "time": float(observation["time"]),
                "viterbi_state": state,
                "viterbi_class": class_names[state],
                "viterbi_posterior": float(posterior[row_index, state]),
            }
            for optional_column in ("source_file", "sample_index", "true_class", "predicted_class"):
                if optional_column in observation:
                    row[optional_column] = observation[optional_column]
            for state_index, class_name in enumerate(class_names):
                row[f"state_{state_index}"] = class_name
                row[f"posterior_state_{state_index}"] = float(posterior[row_index, state_index])
            rows.append(row)
    return pd.DataFrame(rows)


def fit_temporal_models(
    observation_csvs: list[Path],
    *,
    effect_window: tuple[float, float] = (0.1, 0.8),
    baseline_window: tuple[float, float] = (-0.1, 0.0),
    n_permutations: int = 100,
    random_seed: int = 13,
    stay_grid_size: int = 200,
    out_summary: Path | None = None,
    out_states: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Fit sticky switching models to probability observation CSVs and controls."""
    observations = read_probability_observations(observation_csvs)
    prob_columns = probability_columns(observations)
    rows = []
    state_frames = []

    group_columns = _model_group_columns(observations)
    for keys, decoder_frame in observations.groupby(group_columns, sort=True):
        key_values = keys if isinstance(keys, tuple) else (keys,)
        group_values = dict(zip(group_columns, map(str, key_values), strict=True))
        class_names = _class_names(decoder_frame, prob_columns)
        effect_frame = _filter_time_window(decoder_frame, effect_window)
        effect_sequences = _sequences_from_frame(effect_frame, prob_columns)
        observed_fit = fit_sticky_switching_model(effect_sequences, stay_grid_size=stay_grid_size)
        rows.append(_model_row(group_values, "observed_effect", observed_fit))

        baseline_frame = _filter_time_window(decoder_frame, baseline_window)
        if not baseline_frame.empty:
            try:
                baseline_sequences = _sequences_from_frame(baseline_frame, prob_columns)
            except ValueError:
                baseline_sequences = []
            if baseline_sequences:
                baseline_fit = fit_sticky_switching_model(baseline_sequences, stay_grid_size=stay_grid_size)
                rows.append(_model_row(group_values, "baseline_window", baseline_fit))

        if n_permutations > 0:
            for offset, control in enumerate(("shuffled_time", "shuffled_label")):
                control_fits = _fit_control(
                    effect_sequences,
                    control=control,
                    n_permutations=n_permutations,
                    random_seed=random_seed + offset,
                    stay_grid_size=stay_grid_size,
                )
                rows.append(_control_row(group_values, control, control_fits, observed_gain=observed_fit["persistence_gain_per_observation"]))

        if out_states is not None:
            state_frames.append(
                build_state_trace(
                    effect_frame,
                    stay_probability=observed_fit["best_stay_probability"],
                    class_names=class_names,
                    prob_columns=prob_columns,
                )
            )

    summary = pd.DataFrame(rows)
    if out_summary is not None:
        out_summary.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(out_summary, index=False)

    states = pd.concat(state_frames, ignore_index=True) if state_frames else None
    if out_states is not None and states is not None:
        out_states.parent.mkdir(parents=True, exist_ok=True)
        states.to_csv(out_states, index=False)
    return summary, states


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit conservative sticky switching models to RepTrace probability observation CSVs."
    )
    parser.add_argument("observation_csv", nargs="+", help="Observation CSVs or glob patterns emitted by --observations-out/--observation-dir.")
    parser.add_argument("--out-summary", type=Path, required=True)
    parser.add_argument("--out-states", type=Path)
    parser.add_argument("--effect-window", nargs=2, type=float, default=(0.1, 0.8), metavar=("START", "STOP"))
    parser.add_argument("--baseline-window", nargs=2, type=float, default=(-0.1, 0.0), metavar=("START", "STOP"))
    parser.add_argument("--n-permutations", type=int, default=100)
    parser.add_argument("--random-seed", type=int, default=13)
    parser.add_argument("--stay-grid-size", type=int, default=200)
    args = parser.parse_args()

    paths = _expand_paths(args.observation_csv)
    summary, states = fit_temporal_models(
        paths,
        effect_window=tuple(args.effect_window),
        baseline_window=tuple(args.baseline_window),
        n_permutations=args.n_permutations,
        random_seed=args.random_seed,
        stay_grid_size=args.stay_grid_size,
        out_summary=args.out_summary,
        out_states=args.out_states,
    )
    print(f"Wrote temporal model summary: {args.out_summary}")
    if args.out_states is not None and states is not None:
        print(f"Wrote state traces: {args.out_states}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
