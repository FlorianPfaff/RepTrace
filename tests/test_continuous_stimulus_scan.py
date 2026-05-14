from __future__ import annotations

from pathlib import Path

import mne
import numpy as np
import pandas as pd

from reptrace.continuous_stimulus_scan import label_event_table, run_continuous_stimulus_scan
from reptrace.observation_schema import validate_probability_observations


def _write_raw(path: Path, events: pd.DataFrame, *, sfreq: float = 100.0, duration: float = 9.0) -> None:
    rng = np.random.default_rng(13)
    data = rng.normal(scale=0.01, size=(2, int(sfreq * duration)))
    for row in events.to_dict(orient="records"):
        start = int(round((float(row["onset"]) + 0.10) * sfreq))
        stop = int(round((float(row["onset"]) + 0.20) * sfreq))
        if row["stimulus_class"] == "A":
            data[0, start:stop] += 4.0
            data[1, start:stop] -= 1.0
        else:
            data[0, start:stop] -= 4.0
            data[1, start:stop] += 1.0
    info = mne.create_info(["EEG001", "EEG002"], sfreq=sfreq, ch_types="eeg")
    raw = mne.io.RawArray(data, info, verbose="error")
    raw.save(path, overwrite=True, verbose="error")


def test_label_event_table_maps_source_patterns() -> None:
    events = pd.DataFrame(
        [
            {"onset": 1.0, "kind": "apple"},
            {"onset": 2.0, "kind": "noise"},
            {"onset": 3.0, "kind": "other"},
        ]
    )

    labeled = label_event_table(
        events,
        source_column="kind",
        positive_pattern="apple",
        negative_pattern="noise",
        positive_label="fruit",
        negative_label="baseline",
    )

    assert labeled["stimulus_class"].tolist() == ["fruit", "baseline"]


def test_continuous_stimulus_scan_trains_scans_and_summarizes_events(tmp_path: Path) -> None:
    train_events = pd.DataFrame(
        [
            {"onset": 1.0, "stimulus_class": "A"},
            {"onset": 2.0, "stimulus_class": "B"},
            {"onset": 3.0, "stimulus_class": "A"},
            {"onset": 4.0, "stimulus_class": "B"},
            {"onset": 5.0, "stimulus_class": "A"},
            {"onset": 6.0, "stimulus_class": "B"},
        ]
    )
    scan_events = pd.DataFrame(
        [
            {"onset": 2.0, "stimulus_class": "A"},
            {"onset": 4.0, "stimulus_class": "B"},
            {"onset": 6.0, "stimulus_class": "A"},
        ]
    )
    train_raw = tmp_path / "train_raw.fif"
    scan_raw = tmp_path / "scan_raw.fif"
    _write_raw(train_raw, train_events)
    _write_raw(scan_raw, scan_events)

    result = run_continuous_stimulus_scan(
        train_raw=train_raw,
        train_events=train_events,
        scan_raw=scan_raw,
        scan_events=scan_events,
        out_dir=tmp_path / "scan_results",
        train_window=(0.10, 0.20),
        picks="eeg",
        decoder="logistic",
        max_iter=1000,
        scan_step=0.05,
        scan_start=0.0,
        scan_stop=8.0,
        target_classes=["A"],
        threshold_window=(0.0, 1.0),
        detection_window=(1.0, 8.0),
        min_consecutive=1,
        merge_gap=0.10,
        refractory=0.50,
        match_tolerance=0.20,
        annotation_latency=0.15,
    )

    assert not result.observations.empty
    assert {
        "subject",
        "stream_id",
        "fold",
        "split_id",
        "seed",
        "decoder",
        "backend",
        "emission_mode",
        "train_time",
        "test_time",
        "time",
        "sequence_id",
        "calibration_fold",
        "preprocessing_hash",
        "model_hash",
        "prob_class_0",
        "prob_class_1",
    }.issubset(result.observations.columns)
    assert result.observations["backend"].unique().tolist() == ["sklearn"]
    assert result.observations["seed"].unique().tolist() == [13]
    assert result.observations["test_time"].round(6).tolist() == result.observations["time"].round(6).tolist()
    assert result.observations["sequence_id"].str.contains(":").all()
    assert result.observations["preprocessing_hash"].str.len().eq(16).all()
    assert result.observations["model_hash"].str.len().eq(16).all()
    assert validate_probability_observations(result.observations, profile="stimulus-detection").is_valid
    assert result.annotations["stimulus_class"].tolist() == ["A", "A"]
    assert set(result.events["stimulus_class"]) == {"A"}
    assert result.summary.iloc[0]["n_annotations"] == 2
    assert result.summary.iloc[0]["true_positive_count"] == 2
    assert result.summary.iloc[0]["precision"] > 0.0
    assert result.summary.iloc[0]["recall"] == 1.0
    assert (tmp_path / "scan_results" / "stream_observations.csv").is_file()
    assert (tmp_path / "scan_results" / "stimulus_summary.csv").is_file()
