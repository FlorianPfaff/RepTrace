import numpy as np

from reptrace.decoding import make_decoder
from reptrace.decoding.temporal_generalization import TemporalFeatureWindow
from reptrace.decoding.time_resolved import compute_time_resolved_decoding


def _window(center, values, labels):
    return TemporalFeatureWindow(
        center=center,
        start=center - 0.05,
        stop=center + 0.05,
        features=np.asarray(values, dtype=float).reshape(-1, 1),
        labels=np.asarray(labels, dtype=int),
    )


def test_time_resolved_decoding_scores_matched_windows():
    train_windows = [
        _window(0.1, [-2.0, -1.0, 1.0, 2.0], [0, 0, 1, 1]),
        _window(0.0, [-2.0, -1.0, 1.0, 2.0], [0, 0, 1, 1]),
    ]
    test_windows = [
        _window(0.0, [-1.5, 1.5], [0, 1]),
        _window(0.1, [-1.5, 1.5], [0, 1]),
    ]

    scores, predictions = compute_time_resolved_decoding(
        train_windows,
        test_windows,
        fit_model=lambda window: make_decoder("logistic", max_iter=2000).fit(window.features, window.labels),
        predict_labels=lambda model, window: model.predict(window.features),
        chance_accuracy=0.5,
        metadata={"participant": "S01", "decoder": "logistic"},
        model_metadata=lambda _model: {"model_name": "toy"},
        prediction_centers=(0.0,),
    )

    assert scores["window_center_s"].tolist() == [0.0, 0.1]
    assert scores["accuracy"].tolist() == [1.0, 1.0]
    assert scores["chance_accuracy"].tolist() == [0.5, 0.5]
    assert scores["participant"].tolist() == ["S01", "S01"]
    assert predictions["model_name"].tolist() == ["toy", "toy"]
    assert predictions["window_center_s"].tolist() == [0.0, 0.0]
    assert predictions["sample_index"].tolist() == [0, 1]
    assert predictions["correct"].tolist() == [True, True]


def test_time_resolved_decoding_accepts_extra_metadata_callbacks():
    train_windows = [_window(0.0, [-1.0, 1.0], [0, 1])]
    test_windows = [_window(0.0, [-1.0, 1.0], [0, 1])]

    scores, predictions = compute_time_resolved_decoding(
        train_windows,
        test_windows,
        fit_model=lambda window: {"classes": len(np.unique(window.labels))},
        predict_labels=lambda _model, window: window.labels,
        model_metadata=lambda model: {"n_model_classes": model["classes"]},
        score_metadata=lambda _model, _train, _test, _pred: {"custom_score": 3.0},
        prediction_centers=(0.0,),
        prediction_metadata=lambda _window, index, _true, _pred: {"trial": index + 1},
    )

    assert scores.loc[0, "n_model_classes"] == 2
    assert scores.loc[0, "custom_score"] == 3.0
    assert predictions["trial"].tolist() == [1, 2]


def test_time_resolved_decoding_rejects_missing_test_window():
    train_windows = [_window(0.0, [-1.0, 1.0], [0, 1])]
    test_windows = [_window(0.1, [-1.0, 1.0], [0, 1])]

    try:
        compute_time_resolved_decoding(
            train_windows,
            test_windows,
            fit_model=lambda window: window,
            predict_labels=lambda _model, window: window.labels,
        )
    except ValueError as exc:
        assert "Missing test window" in str(exc)
    else:
        raise AssertionError("Expected missing test window to raise ValueError.")
