import numpy as np
import pytest

from reptrace.decoding.cross_subject import (
    CrossSubjectCandidateConfig,
    ParticipantFeatureSet,
    completed_outer_ids,
    leave_one_group_out_decode,
    make_cross_subject_candidate_configs,
    nested_leave_one_group_out_decode,
)


def _separable_feature_sets(n_groups=4):
    return tuple(
        ParticipantFeatureSet(
            id=f"S{group}",
            X=np.array([[-2.0, -1.0], [-1.0, -0.8], [1.0, 0.8], [2.0, 1.0]]),
            y=np.array([0, 0, 1, 1]),
            metadata={"site": "synthetic"},
        )
        for group in range(1, n_groups + 1)
    )


def test_leave_one_group_out_decode_scores_and_exports_predictions():
    artifacts = leave_one_group_out_decode(
        _separable_feature_sets(),
        config=CrossSubjectCandidateConfig(
            classifier="multiclass-svm",
            classifier_param=1.0,
            chance_accuracy=0.5,
            metadata={"window_center_s": 0.175},
        ),
    )

    assert len(artifacts["outer"]) == 4
    assert len(artifacts["predictions"]) == 16
    assert {row["balanced_accuracy"] for row in artifacts["outer"]} == {1.0}
    assert {row["top2_accuracy"] for row in artifacts["outer"]} == {1.0}
    assert {row["metadata_window_center_s"] for row in artifacts["outer"]} == {0.175}
    summary = artifacts["group_summary"][0]
    assert summary["participants_above_chance"] == 4
    assert summary["participants_total"] == 4
    assert summary["one_sided_exact_sign_p_value"] == 1 / 16


def test_nested_leave_one_group_out_selects_best_inner_candidate():
    candidate_configs = (
        CrossSubjectCandidateConfig(id="dummy", classifier="mostFrequentDummy", classifier_param=None, chance_accuracy=0.5),
        CrossSubjectCandidateConfig(id="svm", classifier="multiclass-svm", classifier_param=1.0, chance_accuracy=0.5, components_pca=1),
    )

    artifacts = nested_leave_one_group_out_decode(
        _separable_feature_sets(),
        candidate_configs=candidate_configs,
    )

    assert len(artifacts["outer"]) == 4
    assert len(artifacts["inner_validation"]) == 24
    assert len(artifacts["selected"]) == 4
    assert {row["selected_classifier"] for row in artifacts["selected"]} == {"multiclass-svm"}
    assert {row["balanced_accuracy"] for row in artifacts["outer"]} == {1.0}
    assert artifacts["group_summary"][0]["selected_classifier_counts"] == "multiclass-svm:4"
    assert artifacts["group_summary"][0]["inner_winner_margin_min"] > 0.0


def test_cross_subject_decode_supports_label_shuffle_control():
    artifacts = leave_one_group_out_decode(
        _separable_feature_sets(3),
        config=CrossSubjectCandidateConfig(classifier="multiclass-svm", classifier_param=1.0, chance_accuracy=0.5),
        label_control="label-shuffle",
        label_control_seed=7,
        include_predictions=False,
    )

    assert artifacts["predictions"] == []
    assert {row["label_control"] for row in artifacts["outer"]} == {"shuffle"}
    assert {row["label_control_seed"] for row in artifacts["outer"]} == {7}


def test_cross_subject_decode_resume_skips_existing_outer_rows():
    existing = {
        "outer": [
            {
                "test_group": "S1",
                "accuracy": 0.5,
                "balanced_accuracy": 0.5,
                "chance_accuracy": 0.5,
            }
        ],
        "predictions": [{"test_group": "S1", "trial": 0}],
    }

    artifacts = leave_one_group_out_decode(
        _separable_feature_sets(),
        config=CrossSubjectCandidateConfig(classifier="multiclass-svm", classifier_param=1.0, chance_accuracy=0.5),
        outer_ids=("S1", "S2"),
        existing_artifacts=existing,
    )

    assert completed_outer_ids(artifacts["outer"]) == {"S1", "S2"}
    assert len(artifacts["outer"]) == 2
    assert artifacts["outer"][0]["balanced_accuracy"] == 0.5
    assert sum(row["test_group"] == "S1" for row in artifacts["predictions"]) == 1
    assert sum(row["test_group"] == "S2" for row in artifacts["predictions"]) == 4


def test_make_cross_subject_candidate_configs_expands_metadata_grid():
    configs = make_cross_subject_candidate_configs(
        classifiers=("multiclass-svm", "shrinkage-lda"),
        classifier_params=(1.0,),
        components_pca_values=(1, 2),
        metadata_grid={"window_center_s": (0.150, 0.175)},
    )

    assert len(configs) == 8
    assert configs[0].id == 0
    assert {config.metadata["window_center_s"] for config in configs} == {0.150, 0.175}


def test_cross_subject_decode_rejects_unknown_label_control():
    with pytest.raises(ValueError, match="label_control"):
        leave_one_group_out_decode(
            _separable_feature_sets(),
            config=CrossSubjectCandidateConfig(classifier="multiclass-svm", classifier_param=1.0),
            label_control="scramble",
        )
