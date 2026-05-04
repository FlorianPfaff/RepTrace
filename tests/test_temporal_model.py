from pathlib import Path

import pandas as pd

from reptrace.temporal_model import fit_temporal_models, probability_columns, read_probability_observations


def _observation_frame() -> pd.DataFrame:
    rows = []
    for sequence_id in range(20):
        for time, p0 in [(-0.08, 0.55), (-0.04, 0.45), (0.10, 0.90), (0.20, 0.86), (0.30, 0.14), (0.40, 0.10)]:
            p1 = 1.0 - p0
            predicted_label = 0 if p0 >= p1 else 1
            rows.append(
                {
                    "subject": "sub-01",
                    "fold": sequence_id % 2,
                    "decoder": "logistic",
                    "time": time,
                    "window_start": time - 0.01,
                    "window_stop": time + 0.01,
                    "sample_index": sequence_id,
                    "sequence_id": sequence_id,
                    "true_label": predicted_label,
                    "true_class": "left" if predicted_label == 0 else "right",
                    "predicted_label": predicted_label,
                    "predicted_class": "left" if predicted_label == 0 else "right",
                    "probability_true_class": max(p0, p1),
                    "confidence": max(p0, p1),
                    "class_0": "left",
                    "class_1": "right",
                    "prob_class_0": p0,
                    "prob_class_1": p1,
                }
            )
    return pd.DataFrame(rows)


def test_read_probability_observations_adds_source_file(tmp_path: Path):
    csv_path = tmp_path / "observations.csv"
    _observation_frame().to_csv(csv_path, index=False)

    observations = read_probability_observations([csv_path])

    assert probability_columns(observations) == ["prob_class_0", "prob_class_1"]
    assert observations["source_file"].unique().tolist() == ["observations.csv"]


def test_fit_temporal_models_compares_observed_to_controls(tmp_path: Path):
    csv_path = tmp_path / "observations.csv"
    summary_path = tmp_path / "temporal_summary.csv"
    states_path = tmp_path / "states.csv"
    _observation_frame().to_csv(csv_path, index=False)

    summary, states = fit_temporal_models(
        [csv_path],
        effect_window=(0.1, 0.4),
        baseline_window=(-0.1, 0.0),
        n_permutations=25,
        random_seed=7,
        stay_grid_size=30,
        out_summary=summary_path,
        out_states=states_path,
    )

    observed = summary.loc[summary["condition"] == "observed_effect"].iloc[0]
    shuffled_time = summary.loc[summary["condition"] == "shuffled_time"].iloc[0]
    shuffled_label = summary.loc[summary["condition"] == "shuffled_label"].iloc[0]

    assert summary_path.exists()
    assert states_path.exists()
    assert observed["best_stay_probability"] > 0.5
    assert observed["persistence_gain_per_observation"] > shuffled_time["persistence_gain_per_observation"]
    assert observed["persistence_gain_per_observation"] > shuffled_label["persistence_gain_per_observation"]
    assert shuffled_time["empirical_p_value"] <= 0.2
    assert shuffled_label["empirical_p_value"] <= 0.2
    assert states is not None
    assert {"viterbi_state", "viterbi_class", "posterior_state_0", "posterior_state_1"}.issubset(states.columns)
    assert states[["posterior_state_0", "posterior_state_1"]].sum(axis=1).round(6).eq(1.0).all()
