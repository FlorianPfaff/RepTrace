from __future__ import annotations

import numpy as np
import pandas as pd

from reptrace.observations import ProbabilityObservationTable, probability_columns, stable_hash


def test_stable_hash_is_order_insensitive_for_mappings() -> None:
    assert stable_hash({"a": 1, "b": 2}) == stable_hash({"b": 2, "a": 1})


def test_probability_columns_are_sorted_by_numeric_suffix() -> None:
    frame = pd.DataFrame({"prob_class_10": [0.1], "prob_class_2": [0.2], "prob_class_1": [0.7]})

    assert probability_columns(frame) == ("prob_class_1", "prob_class_2", "prob_class_10")


def test_standardized_probability_table_adds_provenance_columns() -> None:
    table = ProbabilityObservationTable(
        pd.DataFrame(
            {
                "time": [0.1],
                "decoder": ["logistic"],
                "emission_mode": ["calibrated"],
                "prob_class_0": [0.4],
                "prob_class_1": [0.6],
            }
        )
    ).standardized(defaults={"backend": "sklearn", "split_id": "split-001", "seed": 13})

    assert table.frame.loc[0, "backend"] == "sklearn"
    assert table.frame.loc[0, "split_id"] == "split-001"
    assert table.frame.loc[0, "seed"] == 13
    assert table.frame.loc[0, "train_time"] == 0.1
    assert table.frame.loc[0, "test_time"] == 0.1
    assert table.probability_columns == ("prob_class_0", "prob_class_1")


def test_from_decoded_fold_builds_valid_canonical_rows() -> None:
    table = ProbabilityObservationTable.from_decoded_fold(
        probabilities=np.array([[0.2, 0.8], [0.9, 0.1]]),
        test_labels=np.array([1, 0]),
        predictions=np.array([1, 0]),
        class_names=["noise", "face"],
        test_indices=np.array([0, 1]),
        fold=0,
        decoder="logistic",
        backend="sklearn",
        emission_mode="calibrated",
        time=0.15,
    )

    assert table.validate().is_valid
    assert table.frame["probability_true_class"].tolist() == [0.8, 0.9]
