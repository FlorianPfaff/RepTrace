from pathlib import Path

import pandas as pd

from reptrace.onset_sensitivity import (
    build_sensitivity_settings,
    run_onset_sensitivity,
    summarize_sensitivity,
)


def _write_observations(task_dir: Path, *, subject: str = "sub-01") -> Path:
    observation_dir = task_dir / "observations"
    observation_dir.mkdir(parents=True)
    rows = []
    for sequence_id in range(3):
        true_label = sequence_id % 2
        for time, confidence in [(-0.1, 0.55), (0.0, 0.56), (0.1, 0.91), (0.2, 0.90), (0.3, 0.89)]:
            predicted_label = true_label if confidence > 0.8 else 1 - true_label
            rows.append(
                {
                    "subject": subject,
                    "fold": sequence_id % 2,
                    "decoder": "logistic",
                    "emission_mode": "calibrated",
                    "time": time,
                    "window_start": time - 0.01,
                    "window_stop": time + 0.01,
                    "sample_index": sequence_id,
                    "sequence_id": sequence_id,
                    "true_label": true_label,
                    "true_class": f"class-{true_label}",
                    "predicted_label": predicted_label,
                    "predicted_class": f"class-{predicted_label}",
                    "probability_true_class": confidence if predicted_label == true_label else 1.0 - confidence,
                    "confidence": confidence,
                    "class_0": "class-0",
                    "class_1": "class-1",
                    "prob_class_0": confidence if predicted_label == 0 else 1.0 - confidence,
                    "prob_class_1": confidence if predicted_label == 1 else 1.0 - confidence,
                }
            )
    path = observation_dir / f"{subject}_observations.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_build_sensitivity_settings_crosses_parameters():
    settings = build_sensitivity_settings(
        threshold_quantiles=(0.9, 0.95),
        min_consecutive_values=(1, 2),
        stable_prediction_values=(False, True),
    )

    assert len(settings) == 8
    assert settings[0].setting_id == "point_q0900_c01_dnone_anypred"
    assert settings[-1].setting_id == "point_q0950_c02_dnone_stable"


def test_build_sensitivity_settings_can_compare_threshold_methods():
    settings = build_sensitivity_settings(
        threshold_methods=("point", "max_run"),
        threshold_quantiles=(0.95,),
    )

    assert [setting.threshold_method for setting in settings] == ["point", "max_run"]
    assert [setting.setting_id for setting in settings] == [
        "point_q0950_c01_dnone_anypred",
        "maxrun_q0950_c01_dnone_anypred",
    ]


def test_summarize_sensitivity_reports_latency_spread():
    frame = pd.DataFrame(
        {
            "setting_id": ["a", "b", "c"],
            "task": ["task", "task", "task"],
            "decoder": ["logistic", "logistic", "logistic"],
            "emission_mode": ["calibrated", "calibrated", "calibrated"],
            "post_detection_latency_median": [0.1, 0.2, 0.4],
            "false_alarm_rate": [0.0, 0.1, 0.2],
            "post_zero_detected_rate": [1.0, 0.9, 0.8],
            "correct_detection_rate": [0.8, 0.7, 0.6],
            "post_detection_run_length_median": [2.0, 3.0, 4.0],
        }
    )

    summary = summarize_sensitivity(frame)

    assert len(summary) == 1
    row = summary.iloc[0]
    assert row["n_settings"] == 3
    assert row["latency_median_across_settings"] == 0.2
    assert row["latency_range_across_settings"] == 0.30000000000000004
    assert row["false_alarm_rate_max"] == 0.2
    assert row["post_zero_detected_rate_min"] == 0.8


def test_run_onset_sensitivity_writes_summaries_and_plot(tmp_path: Path):
    task_a = tmp_path / "nod_animate_all"
    task_b = tmp_path / "nod_canine_device_all"
    _write_observations(task_a, subject="sub-01")
    _write_observations(task_b, subject="sub-02")

    run = run_onset_sensitivity(
        [task_a, task_b],
        out_dir=tmp_path / "onset_sensitivity_all",
        threshold_window=(-0.1, 0.0),
        threshold_methods=("point", "max_run"),
        threshold_quantiles=(0.8, 0.9),
        detection_start=0.0,
        min_consecutive_values=(1, 2),
        include_stable_prediction=True,
        plot_out=tmp_path / "onset_sensitivity_all" / "sensitivity.png",
    )

    assert len(run.settings) == 16
    assert run.sensitivity_summary_csv.exists()
    assert run.robustness_summary_csv.exists()
    assert run.plot_path is not None
    assert run.plot_path.exists()

    sensitivity = pd.read_csv(run.sensitivity_summary_csv)
    robustness = pd.read_csv(run.robustness_summary_csv)
    assert set(sensitivity["task"]) == {"nod_animate_all", "nod_canine_device_all"}
    assert set(robustness["task"]) == {"nod_animate_all", "nod_canine_device_all"}
    assert sensitivity["setting_id"].nunique() == 16
    assert robustness["n_settings"].min() == 16


def test_run_onset_sensitivity_allows_missing_tasks(tmp_path: Path):
    task_a = tmp_path / "nod_animate_all"
    _write_observations(task_a)

    run = run_onset_sensitivity(
        [task_a, tmp_path / "missing_task"],
        out_dir=tmp_path / "sensitivity",
        threshold_window=(-0.1, 0.0),
        threshold_quantiles=(0.8,),
        allow_missing=True,
    )

    assert run.sensitivity_summary_csv.exists()
    sensitivity = pd.read_csv(run.sensitivity_summary_csv)
    assert set(sensitivity["task"]) == {"nod_animate_all"}
