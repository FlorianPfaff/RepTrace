from __future__ import annotations

import mne
import numpy as np
import pandas as pd
import pytest

from reptrace.mne_time_decode import run_time_resolved_decode


def _write_synthetic_epochs(tmp_path) -> tuple[object, object]:
    rng = np.random.default_rng(13)
    n_epochs = 48
    n_channels = 8
    n_times = 60
    sfreq = 100.0

    labels = np.array(["animate", "inanimate"] * (n_epochs // 2))
    data = rng.normal(0.0, 0.5, size=(n_epochs, n_channels, n_times))
    signal_window = slice(30, 42)
    data[labels == "animate", 0:2, signal_window] += 0.8
    data[labels == "inanimate", 2:4, signal_window] += 0.8

    info = mne.create_info(
        ch_names=[f"MEG{index:03d}" for index in range(n_channels)],
        sfreq=sfreq,
        ch_types="mag",
    )
    metadata = pd.DataFrame(
        {
            "condition": labels,
            "run": np.repeat(np.arange(6), n_epochs // 6),
        }
    )
    epochs = mne.EpochsArray(data, info, tmin=-0.2, metadata=metadata, verbose="error")
    epochs_path = tmp_path / "synthetic-epo.fif"
    epochs.save(epochs_path, overwrite=True, verbose="error")
    return epochs_path, metadata


@pytest.mark.performance
def test_mne_time_resolved_decode_perf(tmp_path, benchmark) -> None:
    epochs_path, _ = _write_synthetic_epochs(tmp_path)
    out_path = tmp_path / "time_decode.csv"
    observations_path = tmp_path / "observations.csv"

    def run():
        return run_time_resolved_decode(
            epochs_path=epochs_path,
            label_column="condition",
            group_column="run",
            out_path=out_path,
            picks="data",
            tmin=-0.1,
            tmax=0.35,
            window_ms=40.0,
            step_ms=40.0,
            n_splits=3,
            max_iter=200,
            decoder="logistic",
            emission_mode="calibrated",
            observation_out_path=observations_path,
            subject="synthetic",
        )

    results = benchmark.pedantic(run, rounds=1, iterations=1)
    assert not results.empty
    assert {"time", "accuracy", "brier", "ece"}.issubset(results.columns)
    assert observations_path.exists()
