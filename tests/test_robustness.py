import pandas as pd
import pytest

from reptrace.decoding.robustness import (
    RobustnessCondition,
    annotate_condition_rows,
    run_participant_robustness_conditions,
    run_robustness_conditions,
)


def test_annotate_condition_rows_adds_control_metadata_and_overwrites_stale_values():
    condition = RobustnessCondition("pca_50", "PCA 50", {"components_pca": 50})

    rows = annotate_condition_rows(
        [
            {
                "control": "old",
                "participant": 1,
                "accuracy": 0.75,
            }
        ],
        condition,
    )

    assert rows == [
        {
            "control": "pca_50",
            "control_label": "PCA 50",
            "participant": 1,
            "accuracy": 0.75,
        }
    ]


def test_annotate_condition_rows_accepts_dataframes():
    condition = RobustnessCondition("default", "Default")
    frame = pd.DataFrame({"participant": [1, 2], "accuracy": [0.6, 0.7]})

    rows = annotate_condition_rows(frame, condition)

    assert [row["control"] for row in rows] == ["default", "default"]
    assert [row["accuracy"] for row in rows] == [0.6, 0.7]


def test_run_robustness_conditions_aggregates_named_artifacts():
    conditions = (
        RobustnessCondition("default", "Default"),
        RobustnessCondition("reverse", "Reverse", {"transfer_direction": "cue-to-main"}),
    )

    artifacts = run_robustness_conditions(
        conditions,
        lambda condition: {
            "accuracy": [{"condition_name": condition.name, "accuracy": 0.5}],
            "predictions": [{"condition_name": condition.name, "trial": 1}],
        },
    )

    assert [row["control"] for row in artifacts["accuracy"]] == ["default", "reverse"]
    assert [row["condition_name"] for row in artifacts["predictions"]] == ["default", "reverse"]


def test_run_participant_robustness_conditions_preserves_control_participant_order_and_progress():
    conditions = (
        RobustnessCondition("default", "Default"),
        RobustnessCondition("weighted", "Weighted"),
    )
    messages = []

    artifacts = run_participant_robustness_conditions(
        conditions,
        [1, 2],
        lambda condition, participant: {
            "accuracy": [{"participant": participant, "condition_name": condition.name}],
            "empty": None,
        },
        progress=messages.append,
    )

    assert [(row["control"], row["participant"]) for row in artifacts["accuracy"]] == [
        ("default", 1),
        ("default", 2),
        ("weighted", 1),
        ("weighted", 2),
    ]
    assert artifacts["empty"] == []
    assert messages == [
        "START control=default",
        "START control=default participant=1",
        "DONE control=default participant=1",
        "START control=default participant=2",
        "DONE control=default participant=2",
        "DONE control=default",
        "START control=weighted",
        "START control=weighted participant=1",
        "DONE control=weighted participant=1",
        "START control=weighted participant=2",
        "DONE control=weighted participant=2",
        "DONE control=weighted",
    ]


def test_run_robustness_conditions_rejects_duplicate_condition_names():
    conditions = (
        RobustnessCondition("default", "First"),
        RobustnessCondition("default", "Second"),
    )

    with pytest.raises(ValueError, match="duplicates: default"):
        run_robustness_conditions(conditions, lambda _condition: {})
