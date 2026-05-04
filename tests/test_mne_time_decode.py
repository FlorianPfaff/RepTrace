from pathlib import Path

import numpy as np
import pandas as pd

from reptrace.mne_time_decode import run_time_resolved_decode


class FakeEpochs:
    def __init__(self, data: np.ndarray, times: np.ndarray, metadata: pd.DataFrame):
        self._data = data
        self.times = times
        self.metadata = metadata

    def __len__(self) -> int:
        return self._data.shape[0]

    def copy(self):
        return FakeEpochs(self._data.copy(), self.times.copy(), self.metadata.copy())

    def pick(self, picks: str):
        return self

    def crop(self, tmin: float | None = None, tmax: float | None = None):
        keep = np.ones(len(self.times), dtype=bool)
        if tmin is not None:
            keep &= self.times >= tmin
        if tmax is not None:
            keep &= self.times <= tmax
        self.times = self.times[keep]
        self._data = self._data[:, :, keep]
        return self

    def __getitem__(self, keep):
        return FakeEpochs(self._data[keep], self.times.copy(), self.metadata.loc[keep].reset_index(drop=True))

    def get_data(self, copy: bool = False):
        return self._data.copy() if copy else self._data


def test_run_time_resolved_decode_writes_probability_observations(tmp_path: Path, monkeypatch):
    rng = np.random.default_rng(13)
    labels = np.array(["animate", "inanimate"] * 4)
    data = rng.normal(size=(8, 1, 5))
    data[labels == "animate", 0, :] += 0.5
    metadata = pd.DataFrame({"condition": labels, "session": ["a", "a", "b", "b", "c", "c", "d", "d"]})
    epochs = FakeEpochs(data, np.array([0.00, 0.01, 0.02, 0.03, 0.04]), metadata)
    monkeypatch.setattr("reptrace.mne_time_decode.mne.read_epochs", lambda *args, **kwargs: epochs)

    out = tmp_path / "decode.csv"
    observations_out = tmp_path / "observations.csv"

    run_time_resolved_decode(
        epochs_path=tmp_path / "sub-01_epo.fif",
        label_column="condition",
        out_path=out,
        n_splits=2,
        window_ms=20,
        step_ms=20,
        max_iter=2000,
        observation_out_path=observations_out,
        subject="sub-01",
        emission_mode="both",
    )

    observations = pd.read_csv(observations_out)

    assert len(observations) == 32
    assert {
        "subject",
        "fold",
        "decoder",
        "emission_mode",
        "time",
        "sample_index",
        "sequence_id",
        "true_class",
        "predicted_class",
        "probability_true_class",
        "confidence",
        "prob_class_0",
        "prob_class_1",
    }.issubset(observations.columns)
    assert observations["subject"].unique().tolist() == ["sub-01"]
    assert sorted(observations["emission_mode"].unique().tolist()) == ["calibrated", "uncalibrated"]
    assert observations[["prob_class_0", "prob_class_1"]].sum(axis=1).round(6).tolist() == [1.0] * 32
