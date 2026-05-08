from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def synthetic_probability_observations(
    *,
    n_subjects: int = 3,
    n_sequences_per_subject: int = 64,
    n_times: int = 72,
    n_classes: int = 4,
    seed: int = 13,
) -> pd.DataFrame:
    """Build deterministic held-out probability observations for benchmark tests.

    The generated frame follows the observation schema emitted by RepTrace time
    decoding: each row is one sequence/time observation with class-probability
    columns, class labels, fold metadata, and onset-friendly confidence scores.
    """

    rng = np.random.default_rng(seed)
    times = np.linspace(-0.35, 0.75, n_times)
    rows: list[dict[str, object]] = []

    for subject_index in range(n_subjects):
        subject = f"sub-{subject_index + 1:02d}"
        for sequence_id in range(n_sequences_per_subject):
            true_label = int(sequence_id % n_classes)
            for time in times:
                concentration = np.full(n_classes, 3.0)
                if time >= 0.12:
                    concentration[true_label] = 12.0
                elif time >= 0.0:
                    concentration[true_label] = 6.0

                probabilities = rng.dirichlet(concentration)
                predicted_label = int(np.argmax(probabilities))
                row: dict[str, object] = {
                    "subject": subject,
                    "fold": sequence_id % 5,
                    "sequence_id": sequence_id,
                    "sample_index": sequence_id,
                    "decoder": "logistic",
                    "emission_mode": "calibrated",
                    "time": float(time),
                    "window_start": float(time - 0.01),
                    "window_stop": float(time + 0.01),
                    "true_label": true_label,
                    "true_class": f"class_{true_label}",
                    "predicted_label": predicted_label,
                    "predicted_class": f"class_{predicted_label}",
                    "probability_true_class": float(probabilities[true_label]),
                    "confidence": float(probabilities[predicted_label]),
                    "is_correct": predicted_label == true_label,
                }
                for class_index, probability in enumerate(probabilities):
                    row[f"class_{class_index}"] = f"class_{class_index}"
                    row[f"prob_class_{class_index}"] = float(probability)
                rows.append(row)

    return pd.DataFrame(rows)


def write_observation_csvs(frame: pd.DataFrame, directory: Path) -> list[Path]:
    """Write one synthetic probability-observation CSV per subject."""

    directory.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for subject, subject_frame in frame.groupby("subject", sort=True):
        path = directory / f"{subject}_observations.csv"
        subject_frame.to_csv(path, index=False)
        paths.append(path)
    return paths
