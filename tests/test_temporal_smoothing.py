from pathlib import Path

import pandas as pd

from reptrace.temporal_smoothing import smooth_probability_observations


def _noisy_observation_frame() -> pd.DataFrame:
    rows = []
    for sequence_id in range(16):
        for time, p0 in [(0.10, 0.92), (0.20, 0.88), (0.30, 0.42), (0.40, 0.86), (0.50, 0.90)]:
            p1 = 1.0 - p0
            predicted_label = 0 if p0 >= p1 else 1
            rows.append(
                {
                    "subject": "sub-01",
                    "fold": sequence_id % 4,
                    "split_id": "stratified-kfold-4",
                    "seed": 13,
                    "decoder": "logistic",
                    "backend": "sklearn",
                    "emission_mode": "calibrated",
                    "train_time": time,
                    "test_time": time,
                    "time": time,
                    "window_start": time - 0.01,
                    "window_stop": time + 0.01,
                    "sample_index": sequence_id,
                    "sequence_id": sequence_id,
                    "session": "ses-01",
                    "true_label": 0,
                    "true_class": "left",
                    "predicted_label": predicted_label,
                    "predicted_class": "left" if predicted_label == 0 else "right",
                    "probability_true_class": p0,
                    "confidence": max(p0, p1),
                    "is_correct": predicted_label == 0,
                    "class_0": "left",
                    "class_1": "right",
                    "prob_class_0": p0,
                    "prob_class_1": p1,
                    "model_hash": "base-model",
                }
            )
    return pd.DataFrame(rows)


def test_temporal_smoothing_exports_posteriors_and_metrics(tmp_path: Path):
    csv_path = tmp_path / "observations.csv"
    out_observations = tmp_path / "smoothed_observations.csv"
    out_metrics = tmp_path / "smoothed_metrics.csv"
    _noisy_observation_frame().to_csv(csv_path, index=False)

    smoothed, metrics = smooth_probability_observations(
        [csv_path],
        fit_window=(0.1, 0.5),
        stay_grid_size=40,
        out_observations=out_observations,
        out_metrics=out_metrics,
    )

    noisy_time = smoothed.loc[smoothed["time"].eq(0.30)]
    assert out_observations.exists()
    assert out_metrics.exists()
    assert smoothed[["prob_class_0", "prob_class_1"]].sum(axis=1).round(6).eq(1.0).all()
    assert smoothed["emission_mode"].unique().tolist() == ["calibrated_temporal_posterior"]
    assert smoothed["base_emission_mode"].unique().tolist() == ["calibrated"]
    assert smoothed["temporal_smoothing_method"].unique().tolist() == ["sticky_forward_backward"]
    assert noisy_time["prob_class_0"].min() > 0.5
    assert noisy_time["is_correct"].all()

    metric_row = metrics.loc[metrics["time"].eq(0.30)].iloc[0]
    assert metric_row["accuracy"] == 1.0
    assert metric_row["emission_mode"] == "calibrated_temporal_posterior"
    assert "temporal_smoothing_stay_probability" in metrics.columns
