import numpy as np

from reptrace.decoding import make_decoder
from reptrace.decoding.temporal_generalization import (
    TemporalFeatureWindow,
    compute_temporal_generalization_matrix,
    summarize_temporal_generalization_matrix,
    summarize_train_time_stability,
)


def _window(center, values, labels):
    return TemporalFeatureWindow(
        center=center,
        start=center - 0.05,
        stop=center + 0.05,
        features=np.asarray(values, dtype=float).reshape(-1, 1),
        labels=np.asarray(labels, dtype=int),
    )


def test_temporal_generalization_matrix_scores_all_train_test_pairs():
    train_windows = [
        _window(0.1, [-2.0, -1.0, 1.0, 2.0], [0, 0, 1, 1]),
        _window(0.0, [-2.0, -1.0, 1.0, 2.0], [1, 1, 0, 0]),
    ]
    test_windows = [
        _window(0.0, [-1.5, 1.5], [1, 0]),
        _window(0.1, [-1.5, 1.5], [0, 1]),
    ]

    rows = compute_temporal_generalization_matrix(
        train_windows,
        test_windows,
        fit_model=lambda window: make_decoder("logistic", max_iter=2000).fit(window.features, window.labels),
        predict_labels=lambda model, window: model.predict(window.features),
        chance_accuracy=0.5,
        metadata={"participant": "S01", "decoder": "logistic"},
    )

    assert rows["train_window_center_s"].tolist() == [0.0, 0.0, 0.1, 0.1]
    assert rows["test_window_center_s"].tolist() == [0.0, 0.1, 0.0, 0.1]
    assert rows["is_diagonal"].tolist() == [True, False, False, True]
    assert rows["participant"].tolist() == ["S01"] * 4
    assert rows["decoder"].tolist() == ["logistic"] * 4
    assert rows["chance_accuracy"].tolist() == [0.5] * 4
    assert rows.loc[rows["is_diagonal"], "accuracy"].tolist() == [1.0, 1.0]
    assert rows.loc[~rows["is_diagonal"], "accuracy"].tolist() == [0.0, 0.0]


def test_temporal_generalization_includes_model_metadata():
    train_windows = [_window(0.0, [-1.0, 1.0], [0, 1])]
    test_windows = [_window(0.0, [-1.0, 1.0], [0, 1])]

    rows = compute_temporal_generalization_matrix(
        train_windows,
        test_windows,
        fit_model=lambda window: {"classes": len(np.unique(window.labels))},
        predict_labels=lambda _model, window: window.labels,
        model_metadata=lambda model: {"n_model_classes": model["classes"]},
    )

    assert rows.loc[0, "n_model_classes"] == 2
    assert rows.loc[0, "accuracy"] == 1.0


def test_summarize_temporal_generalization_matrix_groups_rows():
    rows = compute_temporal_generalization_matrix(
        [
            _window(0.0, [-1.0, 1.0], [0, 1]),
            _window(0.1, [-1.0, 1.0], [0, 1]),
        ],
        [_window(0.0, [-1.0, 1.0], [0, 1])],
        fit_model=lambda window: window,
        predict_labels=lambda _model, window: window.labels,
        chance_accuracy=0.5,
        metadata={"decoder": "toy"},
    )

    summary = summarize_temporal_generalization_matrix(
        rows,
        group_columns=("decoder", "test_window_center_s"),
    )

    assert summary.to_dict("records") == [
        {
            "decoder": "toy",
            "test_window_center_s": 0.0,
            "n_rows": 2,
            "accuracy_mean": 1.0,
            "accuracy_median": 1.0,
            "accuracy_std": 0.0,
            "accuracy_sem": 0.0,
            "percent_mean": 100.0,
            "percent_median": 100.0,
            "percent_std": 0.0,
            "percent_sem": 0.0,
            "chance_accuracy": 0.5,
            "chance_percent": 50.0,
            "above_chance_count": 2,
            "is_diagonal": False,
        }
    ]


def test_summarize_train_time_stability_ranks_cross_time_performance():
    rows = compute_temporal_generalization_matrix(
        [
            _window(0.0, [-2.0, -1.0, 1.0, 2.0], [0, 0, 1, 1]),
            _window(0.1, [-2.0, -1.0, 1.0, 2.0], [0, 0, 1, 1]),
        ],
        [
            _window(0.0, [-1.5, 1.5], [0, 1]),
            _window(0.1, [-1.5, 1.5], [1, 0]),
        ],
        fit_model=lambda window: window,
        predict_labels=lambda model, window: np.asarray(model.labels[: len(window.labels)]),
        chance_accuracy=0.5,
        metadata={"decoder": "toy"},
    )

    summary = summarize_train_time_stability(rows, group_columns=("decoder",))

    assert summary["decoder"].tolist() == ["toy", "toy"]
    assert summary["train_window_center_s"].tolist() == [0.0, 0.1]
    assert summary["n_test_windows"].tolist() == [2, 2]
    assert summary["accuracy_mean"].tolist() == [0.5, 0.5]
    assert summary["accuracy_diagonal_mean"].tolist() == [1.0, 0.0]
    assert summary["accuracy_off_diagonal_mean"].tolist() == [0.0, 1.0]
    assert summary["stability_rank"].tolist() == [1, 1]
