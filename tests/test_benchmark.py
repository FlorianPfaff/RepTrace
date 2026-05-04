from pathlib import Path

import pandas as pd

from reptrace.benchmark import run_benchmark_manifest


def _fake_decode(**kwargs):
    out_path = kwargs["out_path"]
    frame = pd.DataFrame(
        {
            "fold": [0, 1],
            "decoder": [kwargs.get("decoder", "logistic"), kwargs.get("decoder", "logistic")],
            "time": [0.1, 0.1],
            "accuracy": [0.6, 0.8],
            "log_loss": [0.5, 0.4],
            "brier": [0.3, 0.2],
            "ece": [0.1, 0.2],
        }
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_path, index=False)
    return frame


def test_run_benchmark_manifest_runs_subjects_and_aggregates(tmp_path: Path, monkeypatch):
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "subject,epochs,metadata_csv,label_column,group_column\n"
        "sub-01,data/sub-01_epo.fif,data/sub-01_metadata.csv,condition,run\n"
        "sub-02,data/sub-02_epo.fif,data/sub-02_metadata.csv,condition,run\n",
        encoding="utf-8",
    )
    calls = []

    def fake_decode(**kwargs):
        calls.append(kwargs)
        return _fake_decode(**kwargs)

    monkeypatch.setattr("reptrace.benchmark.run_time_resolved_decode", fake_decode)

    run = run_benchmark_manifest(manifest, out_dir=tmp_path / "results")

    assert len(calls) == 2
    assert [path.name for path in run.result_csvs] == ["sub-01_time_decode.csv", "sub-02_time_decode.csv"]
    assert run.aggregate_csv == tmp_path / "results" / "summary.csv"
    assert run.aggregate_csv.exists()
    assert pd.read_csv(run.result_csvs[0])["subject"].tolist() == ["sub-01", "sub-01"]


def test_run_benchmark_manifest_prepares_metadata_from_events(tmp_path: Path, monkeypatch):
    events = tmp_path / "data" / "sub-01_events.csv"
    events.parent.mkdir()
    events.write_text("category\nface\nchair\n", encoding="utf-8")
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "subject,epochs,events_csv,source_column,positive_pattern,label_column,positive_label,negative_label\n"
        "sub-01,data/sub-01_epo.fif,data/sub-01_events.csv,category,face,condition,face,object\n",
        encoding="utf-8",
    )
    calls = []

    def fake_decode(**kwargs):
        calls.append(kwargs)
        return _fake_decode(**kwargs)

    monkeypatch.setattr("reptrace.benchmark.run_time_resolved_decode", fake_decode)

    run_benchmark_manifest(manifest, out_dir=tmp_path / "results")

    metadata_csv = calls[0]["metadata_csv"]
    assert metadata_csv == tmp_path / "results" / "metadata" / "sub-01_metadata.csv"
    assert pd.read_csv(metadata_csv)["condition"].tolist() == ["face", "object"]


def test_run_benchmark_manifest_supports_decoder_column(tmp_path: Path, monkeypatch):
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "subject,epochs,metadata_csv,label_column,decoder\n"
        "sub-01,data/sub-01_epo.fif,data/sub-01_metadata.csv,condition,logistic\n"
        "sub-01,data/sub-01_epo.fif,data/sub-01_metadata.csv,condition,lda\n",
        encoding="utf-8",
    )
    calls = []

    def fake_decode(**kwargs):
        calls.append(kwargs)
        return _fake_decode(**kwargs)

    monkeypatch.setattr("reptrace.benchmark.run_time_resolved_decode", fake_decode)

    run = run_benchmark_manifest(manifest, out_dir=tmp_path / "results")

    assert [call["decoder"] for call in calls] == ["logistic", "lda"]
    assert [path.name for path in run.result_csvs] == ["sub-01_logistic_time_decode.csv", "sub-01_lda_time_decode.csv"]
    summary = pd.read_csv(run.aggregate_csv)
    assert summary["decoder"].tolist() == ["lda", "logistic"]


def test_run_benchmark_manifest_passes_calibration_output_paths(tmp_path: Path, monkeypatch):
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "subject,epochs,metadata_csv,label_column,decoder,calibration_bins\n"
        "sub-01,data/sub-01_epo.fif,data/sub-01_metadata.csv,condition,logistic,5\n",
        encoding="utf-8",
    )
    calls = []

    def fake_decode(**kwargs):
        calls.append(kwargs)
        return _fake_decode(**kwargs)

    monkeypatch.setattr("reptrace.benchmark.run_time_resolved_decode", fake_decode)

    run = run_benchmark_manifest(
        manifest,
        out_dir=tmp_path / "results",
        calibration_dir=tmp_path / "results" / "calibration",
    )

    assert calls[0]["calibration_out_path"] == tmp_path / "results" / "calibration" / "sub-01_logistic_calibration_bins.csv"
    assert calls[0]["calibration_bins"] == 5
    assert run.calibration_csvs == [calls[0]["calibration_out_path"]]


def test_run_benchmark_manifest_resume_skips_complete_existing_rows(tmp_path: Path, monkeypatch):
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "subject,epochs,metadata_csv,label_column\n"
        "sub-01,data/sub-01_epo.fif,data/sub-01_metadata.csv,condition\n"
        "sub-02,data/sub-02_epo.fif,data/sub-02_metadata.csv,condition\n",
        encoding="utf-8",
    )
    existing = tmp_path / "results" / "sub-01_time_decode.csv"
    existing.parent.mkdir()
    pd.DataFrame(
        {
            "subject": ["sub-01"],
            "fold": [0],
            "decoder": ["logistic"],
            "time": [0.1],
            "accuracy": [0.6],
            "log_loss": [0.5],
            "brier": [0.3],
            "ece": [0.1],
        }
    ).to_csv(existing, index=False)
    calls = []

    def fake_decode(**kwargs):
        calls.append(kwargs)
        return _fake_decode(**kwargs)

    monkeypatch.setattr("reptrace.benchmark.run_time_resolved_decode", fake_decode)

    run = run_benchmark_manifest(manifest, out_dir=tmp_path / "results", resume=True)

    assert len(calls) == 1
    assert calls[0]["out_path"].name == "sub-02_time_decode.csv"
    assert run.skipped_existing == 1
    assert [path.name for path in run.result_csvs] == ["sub-01_time_decode.csv", "sub-02_time_decode.csv"]
    assert pd.read_csv(run.aggregate_csv)["n_subjects"].max() == 2


def test_run_benchmark_manifest_resume_reruns_when_calibration_is_missing(tmp_path: Path, monkeypatch):
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "subject,epochs,metadata_csv,label_column,decoder\n"
        "sub-01,data/sub-01_epo.fif,data/sub-01_metadata.csv,condition,logistic\n",
        encoding="utf-8",
    )
    existing = tmp_path / "results" / "sub-01_logistic_time_decode.csv"
    existing.parent.mkdir()
    pd.DataFrame(
        {
            "subject": ["sub-01"],
            "fold": [0],
            "decoder": ["logistic"],
            "time": [0.1],
            "accuracy": [0.6],
            "log_loss": [0.5],
            "brier": [0.3],
            "ece": [0.1],
        }
    ).to_csv(existing, index=False)
    calls = []

    def fake_decode(**kwargs):
        calls.append(kwargs)
        return _fake_decode(**kwargs)

    monkeypatch.setattr("reptrace.benchmark.run_time_resolved_decode", fake_decode)

    run = run_benchmark_manifest(
        manifest,
        out_dir=tmp_path / "results",
        calibration_dir=tmp_path / "results" / "calibration",
        resume=True,
    )

    assert len(calls) == 1
    assert run.skipped_existing == 0
    assert calls[0]["calibration_out_path"] == tmp_path / "results" / "calibration" / "sub-01_logistic_calibration_bins.csv"
