from pathlib import Path

import pandas as pd

from reptrace.semantic_stages import analyze_semantic_stages, posterior_columns, read_state_traces


def _state_trace_frame() -> pd.DataFrame:
    rows = []
    times = [-0.05, 0.10, 0.20, 0.30, 0.50]
    for sequence_id in range(12):
        for semantic_class in ("animate", "inanimate"):
            for time in times:
                if time < 0.0:
                    posterior_true = 0.52
                elif time <= 0.30:
                    posterior_true = 0.82
                else:
                    posterior_true = 0.57
                if semantic_class == "animate":
                    posterior_state_0 = posterior_true
                    posterior_state_1 = 1.0 - posterior_true
                    viterbi_class = "animate" if posterior_true >= 0.6 else "inanimate"
                    viterbi_state = 0 if viterbi_class == "animate" else 1
                else:
                    posterior_state_0 = 1.0 - posterior_true
                    posterior_state_1 = posterior_true
                    viterbi_class = "inanimate" if posterior_true >= 0.6 else "animate"
                    viterbi_state = 1 if viterbi_class == "inanimate" else 0
                rows.append(
                    {
                        "subject": "sub-01",
                        "fold": sequence_id % 2,
                        "sequence_id": f"{semantic_class}-{sequence_id}",
                        "decoder": "logistic",
                        "time": time,
                        "sample_index": sequence_id,
                        "true_class": semantic_class,
                        "predicted_class": semantic_class,
                        "viterbi_state": viterbi_state,
                        "viterbi_class": viterbi_class,
                        "viterbi_posterior": max(posterior_state_0, posterior_state_1),
                        "state_0": "animate",
                        "state_1": "inanimate",
                        "posterior_state_0": posterior_state_0,
                        "posterior_state_1": posterior_state_1,
                    }
                )
    return pd.DataFrame(rows)


def test_read_state_traces_detects_posterior_columns(tmp_path: Path):
    state_csv = tmp_path / "state_trace.csv"
    _state_trace_frame().to_csv(state_csv, index=False)

    traces = read_state_traces([state_csv])

    assert posterior_columns(traces) == ["posterior_state_0", "posterior_state_1"]
    assert traces["source_file"].unique().tolist() == ["state_trace.csv"]


def test_analyze_semantic_stages_detects_category_conditioned_segments(tmp_path: Path):
    state_csv = tmp_path / "state_trace.csv"
    time_csv = tmp_path / "stage_time.csv"
    stages_csv = tmp_path / "stages.csv"
    report_md = tmp_path / "stage_report.md"
    _state_trace_frame().to_csv(state_csv, index=False)

    time_summary, stages, report = analyze_semantic_stages(
        [state_csv],
        posterior_threshold=0.7,
        match_threshold=0.7,
        min_duration=0.15,
        out_time=time_csv,
        out_stages=stages_csv,
        out_report=report_md,
    )

    assert time_csv.exists()
    assert stages_csv.exists()
    assert report_md.exists()
    assert report is not None and "do semantic categories unfold" in report
    assert sorted(stages["semantic_class"].tolist()) == ["animate", "inanimate"]
    assert stages["start_time"].round(3).tolist() == [0.1, 0.1]
    assert stages["stop_time"].round(3).tolist() == [0.3, 0.3]
    assert stages["mean_posterior_true_class"].round(3).tolist() == [0.82, 0.82]
    assert set(time_summary["true_class"]) == {"animate", "inanimate"}
