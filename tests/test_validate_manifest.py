from pathlib import Path

import pandas as pd

from reptrace.validate_manifest import validate_manifest, validation_report_frame


class FakeEpochs:
    def __init__(self, n_epochs: int, metadata: pd.DataFrame | None = None):
        self._n_epochs = n_epochs
        self.metadata = metadata

    def __len__(self):
        return self._n_epochs


def test_validate_manifest_reports_missing_files(tmp_path: Path):
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "subject,epochs,metadata_csv,label_column\n"
        "sub-01,missing-epo.fif,missing.csv,condition\n",
        encoding="utf-8",
    )

    validations = validate_manifest(manifest)

    assert not validations[0].ok
    assert "epochs file does not exist" in " ".join(validations[0].messages)
    assert "metadata_csv does not exist" in " ".join(validations[0].messages)


def test_validate_manifest_accepts_events_metadata(tmp_path: Path, monkeypatch):
    epochs_path = tmp_path / "sub-01_epo.fif"
    epochs_path.write_text("placeholder", encoding="utf-8")
    events_path = tmp_path / "sub-01_events.csv"
    events_path.write_text(
        "category,run\n"
        "face,1\n"
        "chair,1\n"
        "person,2\n"
        "car,2\n",
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "subject,epochs,events_csv,source_column,positive_pattern,label_column,positive_label,negative_label,group_column,n_splits\n"
        "sub-01,sub-01_epo.fif,sub-01_events.csv,category,face|person,condition,face,object,run,2\n",
        encoding="utf-8",
    )

    monkeypatch.setattr("reptrace.validate_manifest.mne.read_epochs", lambda *args, **kwargs: FakeEpochs(4))

    validations = validate_manifest(manifest)

    assert validations[0].ok


def test_validate_manifest_reports_label_and_group_issues(tmp_path: Path, monkeypatch):
    epochs_path = tmp_path / "sub-01_epo.fif"
    epochs_path.write_text("placeholder", encoding="utf-8")
    metadata_path = tmp_path / "sub-01_metadata.csv"
    metadata_path.write_text(
        "condition,run\n"
        "face,1\n"
        "face,1\n"
        "object,1\n",
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "subject,epochs,metadata_csv,label_column,group_column,n_splits\n"
        "sub-01,sub-01_epo.fif,sub-01_metadata.csv,condition,run,2\n",
        encoding="utf-8",
    )

    monkeypatch.setattr("reptrace.validate_manifest.mne.read_epochs", lambda *args, **kwargs: FakeEpochs(3))

    validations = validate_manifest(manifest)

    assert not validations[0].ok
    messages = " ".join(validations[0].messages)
    assert "smallest class" in messages
    assert "fewer than n_splits" in messages


def test_validation_report_frame_is_tabular():
    frame = validation_report_frame([])

    assert list(frame.columns) == []
