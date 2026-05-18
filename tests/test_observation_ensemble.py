from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from reptrace.observation_ensemble import (
    DEFAULT_ENSEMBLE_DECODER,
    ensemble_probability_observations,
    main,
    summarize_ensemble_metrics,
)


def _source_observations() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    decoder_specs = {
        "logistic": {
            "baseline": (0.70, 0.30),
            "effect": (0.80, 0.20),
        },
        "linear_svm": {
            "baseline": (0.80, 0.20),
            "effect": (0.95, 0.05),
        },
    }
    for decoder, probabilities_by_window in decoder_specs.items():
        for sample_index, true_label in enumerate([0, 1]):
            rows.append(
                {
                    "subject": "sub-01",
                    "fold": 0,
                    "split_id": "split-0",
                    "seed": 13,
                    "decoder": decoder,
                    "backend": "sklearn",
                    "emission_mode": "calibrated",
                    "train_time": -0.20,
                    "test_time": -0.20,
                    "time": -0.20,
                    "window_start": -0.21,
                    "window_stop": -0.19,
                    "sample_index": sample_index,
                    "sequence_id": sample_index,
                    "true_label": true_label,
                    "true_class": "zero" if true_label == 0 else "one",
                    "class_0": "zero",
                    "class_1": "one",
                    "prob_class_0": probabilities_by_window["baseline"][0],
                    "prob_class_1": probabilities_by_window["baseline"][1],
                }
            )
        for sample_index in [0, 1]:
            rows.append(
                {
                    "subject": "sub-01",
                    "fold": 0,
                    "split_id": "split-0",
                    "seed": 13,
                    "decoder": decoder,
                    "backend": "sklearn",
                    "emission_mode": "calibrated",
                    "train_time": 0.10,
                    "test_time": 0.10,
                    "time": 0.10,
                    "window_start": 0.09,
                    "window_stop": 0.11,
                    "sample_index": sample_index,
                    "sequence_id": sample_index,
                    "true_label": 0,
                    "true_class": "zero",
                    "class_0": "zero",
                    "class_1": "one",
                    "prob_class_0": probabilities_by_window["effect"][0],
                    "prob_class_1": probabilities_by_window["effect"][1],
                }
            )
    return pd.DataFrame(rows)


def test_ensemble_probability_observations_baseline_debiases_bias() -> None:
    ensemble = ensemble_probability_observations(
        _source_observations(),
        baseline_window=(-0.25, -0.15),
    )

    assert ensemble["decoder"].unique().tolist() == [DEFAULT_ENSEMBLE_DECODER]
    assert ensemble["backend"].unique().tolist() == ["ensemble"]
    assert ensemble["source_decoders"].unique().tolist() == ["logistic|linear_svm"]
    assert ensemble["n_baseline_observations"].unique().tolist() == [2]
    assert np.allclose(ensemble[["prob_class_0", "prob_class_1"]].sum(axis=1), 1.0)

    baseline = ensemble.loc[ensemble["time"] == -0.20]
    effect = ensemble.loc[ensemble["time"] == 0.10]
    assert np.allclose(baseline["prob_class_0"], 0.5)
    assert effect["prob_class_0"].gt(0.70).all()
    assert effect["predicted_label"].tolist() == [0, 0]
    assert effect["probability_true_class"].gt(0.70).all()


def test_summarize_ensemble_metrics_returns_time_resolved_rows() -> None:
    ensemble = ensemble_probability_observations(
        _source_observations(),
        baseline_window=(-0.25, -0.15),
    )

    metrics = summarize_ensemble_metrics(ensemble)

    assert metrics["time"].tolist() == [-0.20, 0.10]
    assert metrics["accuracy"].tolist() == [0.5, 1.0]
    assert metrics["n_test"].tolist() == [2, 2]
    assert metrics["class_names"].tolist() == ["zero|one", "zero|one"]


def test_ensemble_cli_writes_observations_and_metrics(tmp_path: Path) -> None:
    source_path = tmp_path / "source_observations.csv"
    ensemble_path = tmp_path / "ensemble_observations.csv"
    metrics_path = tmp_path / "ensemble_metrics.csv"
    _source_observations().to_csv(source_path, index=False)

    exit_code = main(
        [
            str(source_path),
            "--out",
            str(ensemble_path),
            "--metrics-out",
            str(metrics_path),
            "--baseline-window",
            "-0.25",
            "-0.15",
        ]
    )

    assert exit_code == 0
    ensemble = pd.read_csv(ensemble_path)
    metrics = pd.read_csv(metrics_path)
    assert ensemble["decoder"].unique().tolist() == [DEFAULT_ENSEMBLE_DECODER]
    assert metrics["accuracy"].tolist() == [0.5, 1.0]


def test_ensemble_rejects_misaligned_source_rows() -> None:
    observations = _source_observations()
    misaligned = observations.drop(observations.loc[observations["decoder"] == "linear_svm"].index[-1])

    with pytest.raises(ValueError, match="does not align one-to-one"):
        ensemble_probability_observations(
            misaligned,
            baseline_window=(-0.25, -0.15),
        )
