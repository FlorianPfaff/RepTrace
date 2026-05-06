import numpy as np

from reptrace.decoding.transfer import (
    append_null_class_features,
    cross_validate_feature_decoding,
    evaluate_feature_transfer,
    replace_null_class_predictions,
    sequential_fold_ids,
)


class _ConstantClassifier:
    def __init__(self, label):
        self.label = label

    def predict(self, features):
        return np.full(features.shape[0], self.label)


def test_sequential_fold_ids_matches_legacy_contiguous_assignment():
    assert sequential_fold_ids(4, 2).tolist() == [1, 1, 2, 2]
    assert sequential_fold_ids(5, 5).tolist() == [1, 2, 3, 4, 5]


def test_append_null_class_features_adds_null_rows():
    features, labels = append_null_class_features(
        np.array([[1.0], [2.0]]),
        np.array([1, 2]),
        np.array([[0.1], [0.2]]),
    )

    assert features.tolist() == [[1.0], [2.0], [0.1], [0.2]]
    assert labels.tolist() == [1, 2, 0, 0]


def test_replace_null_class_predictions_uses_least_frequent_non_null_label():
    predictions = replace_null_class_predictions(np.array([0, 1, 1, 2]))

    assert predictions.tolist() == [2, 1, 1, 2]


def test_cross_validate_feature_decoding_replaces_all_null_predictions():
    result = cross_validate_feature_decoding(
        np.array([[-2.0], [1.0], [-1.0], [2.0]]),
        np.array([1, 2, 1, 2]),
        n_folds=2,
        components_pca=float("inf"),
        fit_model=lambda _features, _labels: _ConstantClassifier(0),
    )

    assert result.predictions.tolist() == [1.0, 1.0, 1.0, 1.0]
    assert result.accuracy == 0.5


def test_cross_validate_feature_decoding_supports_binary_one_vs_rest():
    result = cross_validate_feature_decoding(
        np.array([[-2.0], [1.0], [-1.0], [2.0]]),
        np.array([1, 2, 1, 2]),
        n_folds=2,
        classifier="svm-binary",
        classifier_param=1.0,
        components_pca=float("inf"),
    )

    assert result.accuracy == 1.0
    assert result.predictions.tolist() == [1.0, 2.0, 1.0, 2.0]


def test_evaluate_feature_transfer_applies_pca_and_scores_accuracy():
    result = evaluate_feature_transfer(
        train_features=np.array([[-2.0, -2.0], [-1.0, -1.0], [1.0, 1.0], [2.0, 2.0]]),
        train_labels=np.array([0, 0, 1, 1]),
        validation_features=np.array([[-1.5, -1.5], [1.5, 1.5]]),
        validation_labels=np.array([0, 1]),
        classifier="multiclass-svm",
        classifier_param=1.0,
        components_pca=1,
        train_window=(0.1, 0.2),
        n_permutations=2,
        permutation_rng=np.random.default_rng(1),
    )

    assert result.accuracy == 1.0
    assert result.model_bundle.actual_components_pca == 1
    assert result.model_bundle.train_window == (0.1, 0.2)
    assert result.permutation_accuracy.shape == (2,)
