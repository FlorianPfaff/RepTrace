from pathlib import Path

import numpy as np
import pandas as pd

from reptrace.onset_validation import OnsetChunk, parse_chunk_spec, run_onset_chunk_validation, summarize_onset_chunks


def _chunk_observations() -> pd.DataFrame:
    rows = []
    traces = {
        0: [(-0.20, 0.50), (-0.10, 0.52), (0.08, 0.93), (0.16, 0.91), (0.30, 0.82)],
        1: [(-0.20, 0.51), (-0.10, 0.53), (0.08, 0.60), (0.16, 0.62), (0.30, 0.92)],
        2: [(-0.20, 0.54), (-0.10, 0.55), (0.08, 0.58), (0.16, 0.59), (0.30, 0.60)],
    }
    for sequence_id, trace in traces.items():
        true_label = sequence_id % 2
        for time, confidence in trace:
            predicted_label = true_label if confidence >= 0.80 else 1 - true_label
            probabilities = np.array([0.0, 0.0])
            probabilities[predicted_label] = confidence
            probabilities[1 - predicted_label] = 1.0 - confidence
            rows.append(
                {
                    "subject": "sub-01",
                    "fold": sequence_id % 2,
                    "decoder": "logistic",
                    "emission_mode": "calibrated",
                    "time": time,
                    "window_start": time - 0.01,
                    "window_stop": time + 0.01,
                    "sequence_id": sequence_id,
                    "true_label": true_label,
                    "predicted_label": predicted_label,
                    "confidence": confidence,
                    "prob_class_0": probabilities[0],
                    "prob_class_1": probabilities[1],
                }
            )
    return pd.DataFrame(rows)


def test_parse_chunk_spec_accepts_expected_response():
    chunk = parse_chunk_spec("early:0.05:0.20:positive")

    assert chunk == OnsetChunk("early", 0.05, 0.20, "positive")


def test_summarize_onset_chunks_reports_pre_and_post_windows():
    events, summary = summarize_onset_chunks(
        _chunk_observations(),
        chunks=(
            OnsetChunk("pre", -0.25, -0.05, "null"),
            OnsetChunk("early", 0.05, 0.20, "positive"),
            OnsetChunk("late", 0.20, 0.40, "positive"),
        ),
        threshold_window=(-0.25, -0.05),
        threshold_quantile=0.90,
        threshold_method="max_run",
        min_consecutive=1,
        require_stable_prediction=False,
    )

    by_chunk = summary.set_index("chunk")

    assert set(summary["chunk"]) == {"pre", "early", "late"}
    assert events["chunk"].nunique() == 3
    assert by_chunk.loc["pre", "detected_rate"] < by_chunk.loc["early", "detected_rate"]
    assert by_chunk.loc["late", "detected_rate"] > by_chunk.loc["pre", "detected_rate"]
    assert by_chunk.loc["pre", "chunk_expected_response"] == "null"


def test_run_onset_chunk_validation_writes_outputs(tmp_path: Path):
    observations_path = tmp_path / "observations.csv"
    events_path = tmp_path / "chunk_events.csv"
    summary_path = tmp_path / "chunk_summary.csv"
    _chunk_observations().to_csv(observations_path, index=False)

    events, summary = run_onset_chunk_validation(
        [observations_path],
        chunks=(OnsetChunk("late", 0.20, 0.40, "positive"),),
        threshold_window=(-0.25, -0.05),
        threshold_quantile=0.90,
        min_consecutive=1,
        require_stable_prediction=False,
        out_events=events_path,
        out_summary=summary_path,
    )

    assert events_path.exists()
    assert summary_path.exists()
    assert len(events) == 3
    assert len(summary) == 1
    written = pd.read_csv(summary_path)
    assert written["chunk"].iloc[0] == "late"
