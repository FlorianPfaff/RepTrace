from pathlib import Path

import pandas as pd

from reptrace.benchmark import run_benchmark_manifest


def _fake_decode(**kwargs):
    out_path = kwargs["out_path"]
    temporal_train_window = kwargs.get("temporal_train_window")
    temporal_mode = "train_window_ensemble" if temporal_train_window is not None else "same_time"
    temporal_train_start = "" if temporal_train_window is None else temporal_train_window[0]
    temporal_train_stop = "" if temporal_train_window is None else temporal_train_window[1]
    frame = pd.DataFrame(
        {
            "fold": [0, 1],
            "decoder": [kwargs.get("decoder", "logistic"), kwargs.get("decoder", "logistic")],
            "emission_mode": [kwargs.get("emission_mode", "calibrated"), kwargs.get("emission_mode", "calibrated")],
            "feature_preprocessor": [kwargs.get("feature_preprocessor", "none"), kwargs.get("feature_preprocessor", "none")],
            "pca_components": ["" if kwargs.get("pca_components") is None else kwargs.get("pca_components")] * 2,
            "tuned_hyperparameters": [kwargs.get("tune_hyperparameters", False)] * 2,
            "tuning_cv_splits": [kwargs.get("tuning_cv_splits", "") if kwargs.get("tune_hyperparameters", False) else ""] * 2,
            "tuning_scoring": [kwargs.get("tuning_scoring", "") if kwargs.get("tune_hyperparameters", False) else ""] * 2,
            "tuning_c_grid": [kwargs.get("tuning_c_grid", "") if kwargs.get("tune_hyperparameters", False) else ""] * 2,
            "temporal_mode": [temporal_mode, temporal_mode],
            "temporal_train_window_start": [temporal_train_start, temporal_train_start],
            "temporal_train_window_stop": [temporal_train_stop, temporal_train_stop],
            "time": [0.1, 0.1],
            "accuracy": [0.6, 0.8],
            "log_loss": [0.5, 0.4],
            "brier": [0.3, 0.2],
            "ece": [0.1, 0.2],
        }
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_path, index=False)
    observation_out_path = kwargs.get("observation_out_path")
    if observation_out_path is not None:
        observation_out_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {
                "subject": [kwargs.get("subject", "sub-01")],
                "fold": [0],
                "decoder": [kwargs.get("decoder", "logistic")],
                "emission_mode": [kwargs.get("emission_mode", "calibrated")],
                "feature_preprocessor": [kwargs.get("feature_preprocessor", "none")],
                "pca_components": ["" if kwargs.get("pca_components") is None else kwargs.get("pca_components")],
                "tuned_hyperparameters": [kwargs.get("tune_hyperparameters", False)],
                "tuning_cv_splits": [kwargs.get("tuning_cv_splits", "") if kwargs.get("tune_hyperparameters", False) else ""],
                "tuning_scoring": [kwargs.get("tuning_scoring", "") if kwargs.get("tune_hyperparameters", False) else ""],
                "tuning_c_grid": [kwargs.get("tuning_c_grid", "") if kwargs.get("tune_hyperparameters", False) else ""],
                "temporal_mode": [temporal_mode],
                "temporal_train_window_start": [temporal_train_start],
                "temporal_train_window_stop": [temporal_train_stop],
                "time": [0.1],
                "sample_index": [0],
                "sequence_id": [0],
                "true_label": [1],
                "true_class": ["positive"],
                "predicted_label": [1],
                "predicted_class": ["positive"],
                "probability_true_class": [0.7],
                "confidence": [0.7],
                "prob_class_0": [0.3],
                "prob_class_1": [0.7],
            }
        ).to_csv(observation_out_path, index=False)
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


def test_run_benchmark_manifest_passes_observation_output_paths(tmp_path: Path, monkeypatch):
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "subject,epochs,metadata_csv,label_column,decoder\n"
        "sub-01,data/sub-01_epo.fif,data/sub-01_metadata.csv,condition,logistic\n",
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
        observation_dir=tmp_path / "results" / "observations",
    )

    assert calls[0]["observation_out_path"] == tmp_path / "results" / "observations" / "sub-01_logistic_observations.csv"
    assert calls[0]["subject"] == "sub-01"
    assert run.observation_csvs == [calls[0]["observation_out_path"]]
    assert pd.read_csv(run.observation_csvs[0])["prob_class_1"].tolist() == [0.7]


def test_run_benchmark_manifest_can_compare_temporal_smoothing(tmp_path: Path, monkeypatch):
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "subject,epochs,metadata_csv,label_column,decoder\n"
        "sub-01,data/sub-01_epo.fif,data/sub-01_metadata.csv,condition,logistic\n",
        encoding="utf-8",
    )

    def fake_decode(**kwargs):
        out_path = kwargs["out_path"]
        subject = kwargs.get("subject", "sub-01")
        rows = []
        observation_rows = []
        for fold in (0, 1):
            for time, p0 in ((0.1, 0.9), (0.2, 0.45)):
                p1 = 1.0 - p0
                rows.append(
                    {
                        "subject": subject,
                        "fold": fold,
                        "decoder": kwargs.get("decoder", "logistic"),
                        "emission_mode": "calibrated",
                        "time": time,
                        "accuracy": float(p0 > p1),
                        "log_loss": 0.5,
                        "brier": 0.25,
                        "ece": 0.1,
                        "n_test": 2,
                    }
                )
                for sequence_id in (0, 1):
                    observation_rows.append(
                        {
                            "subject": subject,
                            "fold": fold,
                            "decoder": kwargs.get("decoder", "logistic"),
                            "emission_mode": "calibrated",
                            "time": time,
                            "sample_index": sequence_id,
                            "sequence_id": sequence_id,
                            "true_label": 0,
                            "true_class": "animate",
                            "predicted_label": 0 if p0 > p1 else 1,
                            "predicted_class": "animate" if p0 > p1 else "inanimate",
                            "probability_true_class": p0,
                            "confidence": max(p0, p1),
                            "prob_class_0": p0,
                            "prob_class_1": p1,
                        }
                    )
        frame = pd.DataFrame(rows)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(out_path, index=False)
        observation_out_path = kwargs["observation_out_path"]
        observation_out_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(observation_rows).to_csv(observation_out_path, index=False)
        return frame

    monkeypatch.setattr("reptrace.benchmark.run_time_resolved_decode", fake_decode)

    run = run_benchmark_manifest(
        manifest,
        out_dir=tmp_path / "results",
        temporal_smoothing_dir=tmp_path / "results" / "temporal_smoothing",
        temporal_smoothing_fit_window=(0.1, 0.2),
        temporal_smoothing_stay_grid_size=20,
    )

    assert run.observation_csvs == [tmp_path / "results" / "observations" / "sub-01_logistic_observations.csv"]
    assert run.smoothed_observation_csv is not None and run.smoothed_observation_csv.exists()
    assert run.smoothed_metric_csv is not None and run.smoothed_metric_csv.exists()
    summary = pd.read_csv(run.aggregate_csv)
    assert sorted(summary["emission_mode"].unique().tolist()) == ["calibrated", "calibrated_temporal_posterior"]
    assert sorted(summary["temporal_smoothing_method"].unique().tolist()) == ["none", "sticky_forward_backward"]


def test_run_benchmark_manifest_supports_emission_mode_column(tmp_path: Path, monkeypatch):
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "subject,epochs,metadata_csv,label_column,decoder,emission_mode\n"
        "sub-01,data/sub-01_epo.fif,data/sub-01_metadata.csv,condition,linear_svm,calibrated\n"
        "sub-01,data/sub-01_epo.fif,data/sub-01_metadata.csv,condition,linear_svm,uncalibrated\n",
        encoding="utf-8",
    )
    calls = []

    def fake_decode(**kwargs):
        calls.append(kwargs)
        return _fake_decode(**kwargs)

    monkeypatch.setattr("reptrace.benchmark.run_time_resolved_decode", fake_decode)

    run = run_benchmark_manifest(manifest, out_dir=tmp_path / "results")

    assert [call["emission_mode"] for call in calls] == ["calibrated", "uncalibrated"]
    assert [path.name for path in run.result_csvs] == [
        "sub-01_linear_svm_calibrated_time_decode.csv",
        "sub-01_linear_svm_uncalibrated_time_decode.csv",
    ]
    summary = pd.read_csv(run.aggregate_csv)
    assert sorted(summary["emission_mode"].unique().tolist()) == ["calibrated", "uncalibrated"]


def test_run_benchmark_manifest_supports_tuned_pca_whiten_variant(tmp_path: Path, monkeypatch):
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "subject,epochs,metadata_csv,label_column,decoder,feature_preprocessor,pca_components,tune_hyperparameters,tuning_cv_splits,tuning_scoring,tuning_c_grid\n"
        "sub-01,data/sub-01_epo.fif,data/sub-01_metadata.csv,condition,logistic,pca-whiten,0.95,true,2,balanced_accuracy,\"0.1,1,10\"\n",
        encoding="utf-8",
    )
    calls = []

    def fake_decode(**kwargs):
        calls.append(kwargs)
        return _fake_decode(**kwargs)

    monkeypatch.setattr("reptrace.benchmark.run_time_resolved_decode", fake_decode)

    run = run_benchmark_manifest(manifest, out_dir=tmp_path / "results")

    assert calls[0]["feature_preprocessor"] == "pca_whiten"
    assert calls[0]["pca_components"] == "0.95"
    assert calls[0]["tune_hyperparameters"] is True
    assert calls[0]["tuning_cv_splits"] == 2
    assert calls[0]["tuning_scoring"] == "balanced_accuracy"
    assert calls[0]["tuning_c_grid"] == "0.1,1,10"
    assert run.result_csvs[0].name == "sub-01_logistic_pca_whiten_pca0p95_tuned_balanced_accuracy_time_decode.csv"
    summary = pd.read_csv(run.aggregate_csv)
    assert summary["feature_preprocessor"].unique().tolist() == ["pca_whiten"]
    assert summary["tuned_hyperparameters"].unique().tolist() == [True]


def test_run_benchmark_manifest_supports_tuned_temporal_train_window(tmp_path: Path, monkeypatch):
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "subject,epochs,metadata_csv,label_column,decoder,tune_hyperparameters,tuning_cv_splits,tuning_scoring,tuning_c_grid,temporal_train_window_start,temporal_train_window_stop\n"
        "sub-01,data/sub-01_epo.fif,data/sub-01_metadata.csv,condition,logistic,true,2,balanced_accuracy,\"0.1,1,10\",0.12,0.25\n",
        encoding="utf-8",
    )
    calls = []

    def fake_decode(**kwargs):
        calls.append(kwargs)
        return _fake_decode(**kwargs)

    monkeypatch.setattr("reptrace.benchmark.run_time_resolved_decode", fake_decode)

    run = run_benchmark_manifest(manifest, out_dir=tmp_path / "results")

    assert calls[0]["tune_hyperparameters"] is True
    assert calls[0]["temporal_train_window"] == (0.12, 0.25)
    assert run.result_csvs[0].name == "sub-01_logistic_tuned_balanced_accuracy_trainwin0p12_0p25_time_decode.csv"
    summary = pd.read_csv(run.aggregate_csv)
    assert summary["temporal_mode"].unique().tolist() == ["train_window_ensemble"]
    assert summary["temporal_train_window_start"].unique().tolist() == [0.12]
    assert summary["tuned_hyperparameters"].unique().tolist() == [True]


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


def test_run_benchmark_manifest_resume_reruns_when_observations_are_missing(tmp_path: Path, monkeypatch):
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
        observation_dir=tmp_path / "results" / "observations",
        resume=True,
    )

    assert len(calls) == 1
    assert run.skipped_existing == 0
    assert calls[0]["observation_out_path"] == tmp_path / "results" / "observations" / "sub-01_logistic_observations.csv"
