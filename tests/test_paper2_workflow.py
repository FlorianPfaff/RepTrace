from pathlib import Path

import pandas as pd

from reptrace.benchmark import BenchmarkRun
from reptrace.paper2_workflow import prepare_paper2_manifest, run_paper2_workflow
from reptrace.validate_manifest import ManifestValidation


def test_prepare_paper2_manifest_rewrites_data_root_and_adds_decoders(tmp_path: Path):
    source = tmp_path / "source.csv"
    source.write_text(
        "subject,epochs,events_csv,source_column,positive_pattern,label_column\n"
        "sub-01,../data/nod/sub-01_epo.fif,../data/nod/sub-01_events.csv,stim_is_animate,True,condition\n"
        "sub-02,../data/nod/sub-02_epo.fif,../data/nod/sub-02_events.csv,stim_is_animate,True,condition\n",
        encoding="utf-8",
    )

    prepared = prepare_paper2_manifest(
        source,
        tmp_path / "prepared.csv",
        data_root=tmp_path / "nod",
        decoders=("logistic", "linear_svm"),
        expected_subjects=2,
    )

    assert prepared["subject"].tolist() == ["sub-01", "sub-02", "sub-01", "sub-02"]
    assert prepared["decoder"].tolist() == ["logistic", "logistic", "linear_svm", "linear_svm"]
    assert prepared["epochs"].str.endswith("_epo.fif").all()
    assert prepared["events_csv"].str.startswith(str((tmp_path / "nod").resolve())).all()


def _fake_observations(subject: str, decoder: str) -> pd.DataFrame:
    rows = []
    times = [-0.08, -0.04, 0.10, 0.20, 0.30, 0.40]
    for emission_mode, effect_probability in [("calibrated", 0.88), ("uncalibrated", 0.70)]:
        for sequence_id in range(10):
            true_label = sequence_id % 2
            true_class = "left" if true_label == 0 else "right"
            for time in times:
                probability_true = 0.52 if time < 0.0 else effect_probability
                if true_label == 0:
                    p0, p1 = probability_true, 1.0 - probability_true
                else:
                    p0, p1 = 1.0 - probability_true, probability_true
                predicted_label = 0 if p0 >= p1 else 1
                rows.append(
                    {
                        "subject": subject,
                        "fold": sequence_id % 2,
                        "decoder": decoder,
                        "emission_mode": emission_mode,
                        "time": time,
                        "window_start": time - 0.01,
                        "window_stop": time + 0.01,
                        "sample_index": sequence_id,
                        "sequence_id": sequence_id,
                        "true_label": true_label,
                        "true_class": true_class,
                        "predicted_label": predicted_label,
                        "predicted_class": "left" if predicted_label == 0 else "right",
                        "probability_true_class": probability_true,
                        "confidence": max(p0, p1),
                        "class_0": "left",
                        "class_1": "right",
                        "prob_class_0": p0,
                        "prob_class_1": p1,
                    }
                )
    return pd.DataFrame(rows)


def _fake_benchmark(manifest_csv: Path, *, out_dir: Path, aggregate_out: Path, plot_out: Path, calibration_dir: Path, observation_dir: Path, **kwargs) -> BenchmarkRun:
    manifest = pd.read_csv(manifest_csv)
    result_csvs = []
    observation_csvs = []
    for row in manifest.itertuples(index=False):
        subject = str(row.subject)
        decoder = str(row.decoder)
        result_csv = out_dir / f"{subject}_{decoder}_time_decode.csv"
        result_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {
                "subject": [subject, subject],
                "fold": [0, 0],
                "decoder": [decoder, decoder],
                "emission_mode": ["calibrated", "uncalibrated"],
                "time": [0.1, 0.1],
                "accuracy": [0.7, 0.65],
                "log_loss": [0.5, 0.6],
                "brier": [0.2, 0.25],
                "ece": [0.08, 0.14],
            }
        ).to_csv(result_csv, index=False)
        result_csvs.append(result_csv)

        observation_csv = observation_dir / f"{subject}_{decoder}_observations.csv"
        observation_csv.parent.mkdir(parents=True, exist_ok=True)
        _fake_observations(subject, decoder).to_csv(observation_csv, index=False)
        observation_csvs.append(observation_csv)

    aggregate_out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "decoder": ["linear_svm", "linear_svm"],
            "emission_mode": ["calibrated", "uncalibrated"],
            "time": [0.1, 0.1],
            "n_subjects": [1, 1],
            "accuracy_mean": [0.7, 0.65],
            "accuracy_sem": [0.0, 0.0],
            "log_loss_mean": [0.5, 0.6],
            "log_loss_sem": [0.0, 0.0],
            "brier_mean": [0.2, 0.25],
            "brier_sem": [0.0, 0.0],
            "ece_mean": [0.08, 0.14],
            "ece_sem": [0.0, 0.0],
        }
    ).to_csv(aggregate_out, index=False)
    plot_out.parent.mkdir(parents=True, exist_ok=True)
    plot_out.write_bytes(b"fake plot")
    return BenchmarkRun(result_csvs=result_csvs, aggregate_csv=aggregate_out, plot_path=plot_out, calibration_csvs=[], observation_csvs=observation_csvs)


def test_run_paper2_workflow_builds_compact_outputs_and_exports(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        "reptrace.paper2_workflow.validate_manifest",
        lambda manifest_csv: [ManifestValidation(subject="sub-01", ok=True, messages=[])],
    )
    monkeypatch.setattr("reptrace.paper2_workflow.run_benchmark_manifest", _fake_benchmark)

    run = run_paper2_workflow(
        out_dir=tmp_path / "paper2",
        paper_export_dir=tmp_path / "paper_repo" / "results" / "paper2_temporal_state_inference",
        task_ids=("nod_animate",),
        decoders=("linear_svm",),
        n_permutations=3,
        stay_grid_size=12,
        max_subjects=1,
        command_line="python -m reptrace.paper2_workflow --task nod_animate",
    )

    summary = pd.read_csv(run.paper2_summary_csv)
    assert run.paper2_figure.exists()
    assert run.paper2_report.exists()
    assert run.command_log.exists()
    assert summary["task"].tolist() == ["nod_animate"]
    assert summary["decoder"].tolist() == ["linear_svm"]
    assert "delta_control_margin" in summary.columns
    assert run.exported_artifacts
    assert not any(path.name == "state_trace.csv" for path in run.exported_artifacts)
    assert not any("observations" in path.parts for path in run.exported_artifacts)
