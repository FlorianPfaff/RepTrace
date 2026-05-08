from __future__ import annotations

import pytest

from reptrace.onset_detection import annotate_threshold_crossings, detect_onsets

from .synthetic import synthetic_probability_observations


@pytest.mark.performance
def test_onset_detection_max_run_perf(benchmark) -> None:
    observations = synthetic_probability_observations(
        n_subjects=4,
        n_sequences_per_subject=96,
        n_times=96,
        n_classes=4,
    )

    def run():
        return detect_onsets(
            observations,
            threshold_window=(-0.35, -0.05),
            detection_window=(0.0, 0.75),
            threshold_quantile=0.95,
            score_column="confidence",
            threshold_method="max_run",
            min_consecutive=3,
            min_duration=0.02,
            require_stable_prediction=True,
        )

    events = benchmark.pedantic(run, rounds=1, iterations=1)
    assert len(events) == 4 * 96
    assert {"detected", "detection_time", "score_threshold"}.issubset(events.columns)


@pytest.mark.performance
def test_threshold_annotation_max_run_perf(benchmark) -> None:
    observations = synthetic_probability_observations(
        n_subjects=4,
        n_sequences_per_subject=96,
        n_times=96,
        n_classes=4,
    )

    def run():
        return annotate_threshold_crossings(
            observations,
            threshold_window=(-0.35, -0.05),
            threshold_quantile=0.95,
            score_column="confidence",
            threshold_method="max_run",
            min_consecutive=3,
            min_duration=0.02,
            require_stable_prediction=True,
        )

    thresholded = benchmark.pedantic(run, rounds=1, iterations=1)
    assert len(thresholded) == len(observations)
    assert {"onset_score", "score_threshold", "above_threshold"}.issubset(thresholded.columns)
