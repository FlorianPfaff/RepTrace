from pathlib import Path

import pandas as pd

from reptrace.onset_workflow import plot_onset_summary, run_onset_workflow, run_task_onset_detection


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


def test_run_task_onset_detection_writes_task_outputs(tmp_path: Path):
    task_dir = tmp_path / "nod_animate_all"
    _write_observations(task_dir)

    output, events, summary = run_task_onset_detection(
        task_dir,
        out_dir=tmp_path / "onsets",
        threshold_window=(-0.1, 0.0),
        threshold_quantile=0.8,
        detection_start=0.0,
        min_consecutive=2,
    )

    assert output.task == "nod_animate_all"
    assert output.events_csv.exists()
    assert output.summary_csv.exists()
    assert output.n_events == 3
    assert output.n_summary_rows == 1
    assert set(events["task"]) == {"nod_animate_all"}
    assert set(summary["task"]) == {"nod_animate_all"}
    assert summary["post_zero_detected_count"].iloc[0] == 3


def test_run_onset_workflow_combines_tasks_and_writes_plot(tmp_path: Path):
    task_a = tmp_path / "nod_animate_all"
    task_b = tmp_path / "nod_canine_device_all"
    _write_observations(task_a, subject="sub-01")
    _write_observations(task_b, subject="sub-02")

    run = run_onset_workflow(
        [task_a, task_b],
        out_dir=tmp_path / "onset_detection_all",
        threshold_window=(-0.1, 0.0),
        threshold_quantile=0.8,
        detection_start=0.0,
        min_consecutive=2,
        write_combined_events=True,
        plot_out=tmp_path / "onset_detection_all" / "onset_summary.png",
    )

    assert run.summary_all_csv.exists()
    assert run.events_all_csv is not None
    assert run.events_all_csv.exists()
    assert run.plot_path is not None
    assert run.plot_path.exists()
    summary = pd.read_csv(run.summary_all_csv)
    assert set(summary["task"]) == {"nod_animate_all", "nod_canine_device_all"}


def test_run_onset_workflow_allows_missing_tasks(tmp_path: Path):
    task_a = tmp_path / "nod_animate_all"
    missing = tmp_path / "missing_task"
    _write_observations(task_a)

    run = run_onset_workflow(
        [task_a, missing],
        out_dir=tmp_path / "onsets",
        threshold_window=(-0.1, 0.0),
        threshold_quantile=0.8,
        allow_missing=True,
    )

    assert run.summary_all_csv.exists()
    assert len(run.task_outputs) == 1


def test_plot_onset_summary_rejects_empty_frame(tmp_path: Path):
    try:
        plot_onset_summary(pd.DataFrame(), tmp_path / "plot.png")
    except ValueError as exc:
        assert "empty onset summary" in str(exc)
    else:
        raise AssertionError("Expected ValueError for empty onset summary.")
