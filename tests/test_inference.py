from pathlib import Path

import pandas as pd
import pytest

from reptrace.inference import sign_flip_time_inference, subject_time_effects


def _write_subject_csv(path: Path, subject: str, effects: list[float]) -> None:
    times = [-0.05, 0.05, 0.15, 0.25]
    rows = []
    for fold in [0, 1]:
        for time, effect in zip(times, effects, strict=True):
            rows.append(
                {
                    "subject": subject,
                    "fold": fold,
                    "time": time,
                    "accuracy": 0.5 + effect + fold * 0.002,
                    "log_loss": 0.7,
                    "brier": 0.5,
                    "ece": 0.1,
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False)


def test_subject_time_effects_averages_folds(tmp_path: Path):
    csv_path = tmp_path / "sub-01_time_decode.csv"
    _write_subject_csv(csv_path, "sub-01", [0.0, 0.1, 0.2, 0.3])

    effects = subject_time_effects([csv_path], chance=0.5)

    assert effects.index.tolist() == ["sub-01"]
    assert effects.round(3).iloc[0].tolist() == [0.001, 0.101, 0.201, 0.301]


def test_subject_time_effects_weights_folds_by_test_size(tmp_path: Path):
    csv_path = tmp_path / "sub-01_time_decode.csv"
    pd.DataFrame(
        {
            "subject": ["sub-01", "sub-01"],
            "fold": [0, 1],
            "time": [0.1, 0.1],
            "accuracy": [1.0, 0.0],
            "log_loss": [0.1, 1.0],
            "brier": [0.1, 0.9],
            "ece": [0.0, 1.0],
            "n_test": [9, 1],
        }
    ).to_csv(csv_path, index=False)

    effects = subject_time_effects([csv_path], chance=0.5)

    assert effects.loc["sub-01", 0.1] == pytest.approx(0.4)


def test_subject_time_effects_rejects_mixed_decoders_without_filter(tmp_path: Path):
    csv_path = tmp_path / "sub-01_time_decode.csv"
    frame = pd.DataFrame(
        {
            "subject": ["sub-01", "sub-01"],
            "fold": [0, 0],
            "time": [0.1, 0.1],
            "decoder": ["logistic", "linear_svm"],
            "accuracy": [0.6, 0.8],
            "log_loss": [0.7, 0.7],
            "brier": [0.5, 0.5],
            "ece": [0.1, 0.1],
        }
    )
    frame.to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="multiple decoder values"):
        subject_time_effects([csv_path])


def test_subject_time_effects_filters_decoder_and_emission_mode(tmp_path: Path):
    csv_path = tmp_path / "sub-01_time_decode.csv"
    rows = []
    for decoder, emission_mode, accuracy in [
        ("logistic", "calibrated", 0.7),
        ("logistic", "uncalibrated", 0.6),
        ("linear_svm", "calibrated", 0.9),
        ("linear_svm", "uncalibrated", 0.8),
    ]:
        rows.append(
            {
                "subject": "sub-01",
                "fold": 0,
                "time": 0.1,
                "decoder": decoder,
                "emission_mode": emission_mode,
                "accuracy": accuracy,
                "log_loss": 0.7,
                "brier": 0.5,
                "ece": 0.1,
            }
        )
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    effects = subject_time_effects([csv_path], chance=0.5, decoder="linear_svm", emission_mode="uncalibrated")

    assert effects.loc["sub-01", 0.1] == pytest.approx(0.3)


def test_sign_flip_time_inference_finds_cluster(tmp_path: Path):
    csv_paths = []
    for idx in range(8):
        csv_path = tmp_path / f"sub-{idx + 1:02d}_time_decode.csv"
        _write_subject_csv(
            csv_path,
            f"sub-{idx + 1:02d}",
            [0.0, 0.07 + idx * 0.004, 0.10 + idx * 0.004, 0.09 + idx * 0.004],
        )
        csv_paths.append(csv_path)

    time_table, cluster_table = sign_flip_time_inference(
        csv_paths,
        n_permutations=2048,
        random_state=7,
        cluster_alpha=0.05,
    )

    assert len(time_table) == 4
    assert not cluster_table.empty
    assert cluster_table["cluster_p"].min() < 0.05
    assert cluster_table.loc[cluster_table["cluster_p"].idxmin(), "start_time"] == 0.05
    assert time_table["emission_mode"].unique().tolist() == ["calibrated"]
