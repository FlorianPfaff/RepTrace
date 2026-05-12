from __future__ import annotations

from pathlib import Path

import pandas as pd


DEFAULT_TRACE = [(-0.1, 0.55), (0.0, 0.56), (0.1, 0.91), (0.2, 0.90), (0.3, 0.89)]
FALSE_ALARM_TRACE = [(-0.2, 0.50), (-0.1, 0.95), (0.0, 0.55), (0.1, 0.56), (0.2, 0.57)]


def write_observations(
    task_dir: Path,
    *,
    subject: str = "sub-01",
    n_sequences: int = 3,
    trace: list[tuple[float, float]] | None = None,
) -> Path:
    observation_dir = task_dir / "observations"
    observation_dir.mkdir(parents=True)
    rows = []
    trace = DEFAULT_TRACE if trace is None else trace
    for sequence_id in range(n_sequences):
        true_label = sequence_id % 2
        for time, confidence in trace:
            predicted_label = true_label if confidence > 0.8 else 1 - true_label
            rows.append(
                {
                    "subject": subject,
                    "fold": sequence_id % 2,
                    "decoder": "logistic",
                    "emission_mode": "calibrated",
                    "time": time,
                    "window_start": time - 0.01,
                    "window_stop": time + 0.01,
                    "sample_index": sequence_id,
                    "sequence_id": sequence_id,
                    "true_label": true_label,
                    "true_class": f"class-{true_label}",
                    "predicted_label": predicted_label,
                    "predicted_class": f"class-{predicted_label}",
                    "probability_true_class": confidence if predicted_label == true_label else 1.0 - confidence,
                    "confidence": confidence,
                    "class_0": "class-0",
                    "class_1": "class-1",
                    "prob_class_0": confidence if predicted_label == 0 else 1.0 - confidence,
                    "prob_class_1": confidence if predicted_label == 1 else 1.0 - confidence,
                }
            )
    path = observation_dir / f"{subject}_observations.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path
