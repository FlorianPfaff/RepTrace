from __future__ import annotations

import pytest

from reptrace.temporal_model import fit_sticky_switching_model, fit_temporal_models

from .synthetic import synthetic_probability_observations, write_observation_csvs


@pytest.mark.performance
def test_sticky_switching_grid_search_perf(benchmark) -> None:
    observations = synthetic_probability_observations(
        n_subjects=1,
        n_sequences_per_subject=192,
        n_times=72,
        n_classes=4,
    )
    prob_columns = [f"prob_class_{index}" for index in range(4)]
    sequences = [
        sequence_frame.sort_values("time")[prob_columns].to_numpy()
        for _, sequence_frame in observations.groupby("sequence_id", sort=True)
    ]

    def run():
        return fit_sticky_switching_model(sequences, stay_grid_size=100)

    fit = benchmark.pedantic(run, rounds=1, iterations=1)
    assert fit["n_sequences"] == 192
    assert fit["n_states"] == 4
    assert 0.25 <= fit["best_stay_probability"] <= 0.995


@pytest.mark.performance
def test_fit_temporal_models_perf(tmp_path, benchmark) -> None:
    observations = synthetic_probability_observations(
        n_subjects=3,
        n_sequences_per_subject=64,
        n_times=72,
        n_classes=4,
    )
    observation_csvs = write_observation_csvs(observations, tmp_path / "observations")

    def run():
        return fit_temporal_models(
            observation_csvs,
            effect_window=(0.1, 0.7),
            baseline_window=(-0.3, -0.05),
            n_permutations=3,
            random_seed=13,
            stay_grid_size=60,
            out_summary=None,
            out_states=None,
        )

    summary, states = benchmark.pedantic(run, rounds=1, iterations=1)
    assert states is None
    assert not summary.empty
    assert {"condition", "best_stay_probability", "persistence_gain_per_observation"}.issubset(summary.columns)
