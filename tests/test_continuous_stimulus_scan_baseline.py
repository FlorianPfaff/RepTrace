from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.preprocessing import LabelEncoder

import reptrace.continuous_stimulus_scan as continuous_scan
from reptrace import _continuous_stimulus_scan_extensions as extensions


class FakeRaw:
    def __init__(self, data: np.ndarray, sfreq: float):
        self._data = data
        self.info = {"sfreq": sfreq}
        self.n_times = data.shape[1]
        self.times = np.arange(self.n_times, dtype=float) / sfreq
        self.first_samp = 0
        self._sfreq = sfreq

    def pick(self, picks: list[str]):
        assert picks == ["MEG001"]
        return self

    def time_as_index(self, times, use_rounding: bool = True):
        values = np.asarray(times, dtype=float) * self._sfreq
        if use_rounding:
            return np.rint(values).astype(int)
        return values.astype(int)

    def get_data(self, *, start: int, stop: int):
        return self._data[:, start:stop]


def test_continuous_scan_applies_epoch_baseline_to_stream_features(tmp_path: Path, monkeypatch) -> None:
    raw = FakeRaw(np.array([[10.0, 20.0, 30.0, 40.0, 50.0, 60.0]]), sfreq=10.0)
    monkeypatch.setattr(continuous_scan.mne.io, "read_raw_fif", lambda *args, **kwargs: raw)

    captured: dict[str, np.ndarray] = {}

    def fake_predict_emission_probabilities(model, features: np.ndarray, *, emission_mode: str) -> np.ndarray:
        captured["features"] = features.copy()
        return np.array([[0.25, 0.75]])

    monkeypatch.setattr(continuous_scan, "predict_emission_probabilities", fake_predict_emission_probabilities)

    model = object()
    extensions._remember_model_preprocessing(model, train_window=(-0.2, 0.1), baseline=(-0.2, -0.1))
    encoder = LabelEncoder().fit(["negative", "positive"])

    observations = continuous_scan._scan_raw_probabilities(
        scan_raw=tmp_path / "scan_raw.fif",
        model=model,
        encoder=encoder,
        channel_names=["MEG001"],
        n_window_samples=4,
        segments=[continuous_scan.ScanSegment("stream", start=0.2, stop=0.6, output_origin=0.0)],
        scan_step=0.4,
        decoder="logistic",
        emission_mode="calibrated",
        subject=None,
        demean_window=False,
    )

    assert observations["predicted_class"].tolist() == ["positive"]
    # The scanned raw samples are [30, 40, 50, 60]. With train_window=(-0.2, 0.1)
    # and baseline=(-0.2, -0.1), the first two samples define the baseline mean 35.
    np.testing.assert_allclose(captured["features"], np.array([[-5.0, 5.0, 15.0, 25.0]]))
