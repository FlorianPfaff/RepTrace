from pathlib import Path

import pandas as pd

from reptrace.onset_workflow import plot_onset_summary, run_onset_workflow, run_task_onset_detection
from onset_test_helpers import FALSE_ALARM_TRACE, write_observations


def test_run_task_onset_detection_writes_task_outputs(tmp_path: Path):
    task_dir = tmp_path / "nod_animate_all"
    write_observations(task_dir)

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
    write_observations(task_a, subject="sub-01")
    write_observations(task_b, subject="sub-02")

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


def test_run_onset_workflow_detection_window_controls_false_alarm_mode(tmp_path: Path):
    task_dir = tmp_path / "ds000117_faces_sub01"
    write_observations(task_dir, trace=FALSE_ALARM_TRACE)

    post_only = run_onset_workflow(
        [task_dir],
        out_dir=tmp_path / "post_only",
        threshold_window=(-0.2, -0.2),
        threshold_quantile=0.5,
        min_consecutive=1,
    )
    full_scan = run_onset_workflow(
        [task_dir],
        out_dir=tmp_path / "full_scan",
        threshold_window=(-0.2, -0.2),
        threshold_quantile=0.5,
        detection_window=(-0.2, 0.3),
        min_consecutive=1,
    )

    post_summary = pd.read_csv(post_only.summary_all_csv)
    full_summary = pd.read_csv(full_scan.summary_all_csv)

    assert post_summary["false_alarm_count"].iloc[0] == 0
    assert full_summary["false_alarm_count"].iloc[0] == 3
    assert full_summary["false_alarm_rate"].iloc[0] == 1.0


def test_run_onset_workflow_allows_missing_tasks(tmp_path: Path):
    task_a = tmp_path / "nod_animate_all"
    missing = tmp_path / "missing_task"
    write_observations(task_a)

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
