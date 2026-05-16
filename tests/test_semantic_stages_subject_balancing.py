import numpy as np
import pandas as pd

from reptrace.semantic_stages import summarize_category_timecourse, summarize_dominant_timecourse


def test_category_timecourse_is_subject_balanced_not_trial_weighted():
    # Subject s1 has three trials and subject s2 has one. A trial-weighted mean
    # would be 0.75; the intended subject-balanced mean is mean([1.0, 0.0]) = 0.5.
    state_traces = pd.DataFrame(
        {
            "decoder": ["d"] * 4,
            "emission_mode": ["calibrated"] * 4,
            "subject": ["s1", "s1", "s1", "s2"],
            "sequence_id": [0, 1, 2, 0],
            "time": [0.0, 0.0, 0.0, 0.0],
            "true_class": ["A", "A", "A", "A"],
            "viterbi_class": ["A", "A", "A", "B"],
            "viterbi_posterior": [1.0, 1.0, 1.0, 0.0],
            "posterior_state_0": [1.0, 1.0, 1.0, 0.0],
            "state_0": ["A", "A", "A", "A"],
        }
    )

    summary, state_names = summarize_category_timecourse(state_traces)

    assert state_names == ["A"]
    assert len(summary) == 1
    row = summary.iloc[0]
    assert row.n_observations == 4
    assert row.n_subjects == 2
    assert row.n_sequences == 4
    assert row.posterior_true_class_mean == 0.5
    assert row.posterior_true_class_sem == 0.5
    assert row.viterbi_match_rate == 0.5
    assert row.viterbi_posterior_mean == 0.5


def test_dominant_timecourse_is_subject_balanced_not_trial_weighted():
    # s1 has more trials favouring A, but subjects are balanced and s2's stronger
    # evidence makes B dominant after averaging within subject first.
    state_traces = pd.DataFrame(
        {
            "decoder": ["d"] * 4,
            "emission_mode": ["calibrated"] * 4,
            "subject": ["s1", "s1", "s1", "s2"],
            "sequence_id": [0, 1, 2, 0],
            "time": [0.0, 0.0, 0.0, 0.0],
            "viterbi_class": ["A", "A", "A", "B"],
            "posterior_state_0": [0.8, 0.8, 0.8, 0.1],
            "posterior_state_1": [0.2, 0.2, 0.2, 0.9],
            "state_0": ["A", "A", "A", "A"],
            "state_1": ["B", "B", "B", "B"],
        }
    )

    summary = summarize_dominant_timecourse(state_traces)

    assert len(summary) == 1
    row = summary.iloc[0]
    assert row.true_class == "B"
    assert row.n_observations == 4
    assert row.n_subjects == 2
    assert row.n_sequences == 4
    assert np.isclose(row.posterior_true_class_mean, 0.55)
    assert np.isclose(row.posterior_true_class_sem, 0.35)
    assert np.isclose(row.viterbi_match_rate, 0.5)
    assert np.isclose(row.viterbi_posterior_mean, 0.55)
    assert np.isclose(row.posterior_margin, 0.10)
