from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from itertools import product
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score

from reptrace.decoding.classifiers import get_default_classifier_param, should_use_default_classifier_param, train_classifier
from reptrace.decoding.windowed import fit_window_model, predict_window_model, transform_window_features
from reptrace.metrics.classification import ranked_accuracy_metrics, subject_level_signflip_summary


@dataclass(frozen=True)
class ParticipantFeatureSet:
    """Feature matrix and labels for one held-out unit, usually one participant."""

    id: Any
    X: Sequence[Sequence[float]] | np.ndarray
    y: Sequence[Any] | np.ndarray
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CrossSubjectCandidateConfig:
    """Dataset-independent candidate configuration for cross-subject decoding."""

    id: Any | None = None
    classifier: str = "multiclass-svm"
    classifier_param: Any = np.nan
    components_pca: int | float = float("inf")
    chance_accuracy: float | None = None
    random_state: int | None = 0
    top_ks: tuple[int, ...] = (2, 3)
    metadata: Mapping[str, Any] = field(default_factory=dict)


def make_cross_subject_candidate_configs(
    *,
    classifiers: Sequence[str] = ("multiclass-svm",),
    classifier_params: Sequence[Any] = (np.nan,),
    components_pca_values: Sequence[int | float] = (float("inf"),),
    chance_accuracy: float | None = None,
    random_state: int | None = 0,
    top_ks: Sequence[int] = (2, 3),
    metadata_grid: Mapping[str, Sequence[Any]] | None = None,
) -> tuple[CrossSubjectCandidateConfig, ...]:
    """Build a small Cartesian grid of generic cross-subject candidates."""
    metadata_grid = {} if metadata_grid is None else dict(metadata_grid)
    metadata_keys = tuple(metadata_grid)
    metadata_values = tuple(tuple(metadata_grid[key]) for key in metadata_keys)
    if metadata_keys and any(len(values) == 0 for values in metadata_values):
        raise ValueError("metadata_grid values must not be empty.")
    metadata_products = product(*metadata_values) if metadata_keys else [()]

    configs = []
    for classifier, classifier_param, components_pca, metadata_tuple in product(classifiers, classifier_params, components_pca_values, metadata_products):
        metadata = dict(zip(metadata_keys, metadata_tuple))
        configs.append(
            CrossSubjectCandidateConfig(
                id=len(configs),
                classifier=classifier,
                classifier_param=classifier_param,
                components_pca=components_pca,
                chance_accuracy=chance_accuracy,
                random_state=random_state,
                top_ks=tuple(int(k) for k in top_ks),
                metadata=metadata,
            )
        )
    return tuple(configs)


def leave_one_group_out_decode(
    feature_sets: Sequence[ParticipantFeatureSet],
    *,
    config: CrossSubjectCandidateConfig | None = None,
    outer_ids: Sequence[Any] | None = None,
    existing_artifacts: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
    label_control: str = "none",
    label_control_seed: int | None = 0,
    include_predictions: bool = True,
    after_outer_fold: Any | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Run fixed-candidate leave-one-group-out decoding over feature sets.

    ``existing_artifacts`` may contain prior ``outer`` and ``predictions`` rows.
    Any outer id already present in ``outer`` is skipped, which makes simple CSV
    based workflows resumable.
    """
    config = config or CrossSubjectCandidateConfig()
    normalized_sets = _normalize_feature_sets(feature_sets)
    outer_ids = _normalize_outer_ids(normalized_sets, outer_ids)
    label_control = _normalize_label_control(label_control)
    resumed = _existing_artifact_rows(existing_artifacts)
    outer_rows = resumed["outer"]
    prediction_rows = resumed["predictions"]
    completed_outer_ids = _completed_outer_ids(outer_rows)

    for test_id in outer_ids:
        if str(test_id) in completed_outer_ids:
            continue
        train_sets, test_set = _split_train_test(normalized_sets, test_id)
        outer_row, fold_predictions = _evaluate_fold(
            train_sets,
            test_set,
            config,
            candidate_index=0,
            label_control=label_control,
            label_control_seed=label_control_seed,
            label_control_context=("outer", test_id),
            include_predictions=include_predictions,
        )
        outer_rows.append(outer_row)
        prediction_rows.extend(fold_predictions)
        if after_outer_fold is not None:
            after_outer_fold(_assemble_artifacts(outer_rows, prediction_rows=prediction_rows))

    return _assemble_artifacts(outer_rows, prediction_rows=prediction_rows)


def nested_leave_one_group_out_decode(
    feature_sets: Sequence[ParticipantFeatureSet],
    *,
    candidate_configs: Sequence[CrossSubjectCandidateConfig],
    outer_ids: Sequence[Any] | None = None,
    selection_metric: str = "balanced_accuracy",
    existing_artifacts: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
    label_control: str = "none",
    label_control_seed: int | None = 0,
    include_predictions: bool = True,
    after_outer_fold: Any | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Run nested leave-one-group-out model selection without touching outer test groups."""
    normalized_sets = _normalize_feature_sets(feature_sets)
    candidate_configs = tuple(candidate_configs)
    if not candidate_configs:
        raise ValueError("At least one candidate configuration is required.")
    if len(normalized_sets) < 3:
        raise ValueError("Nested leave-one-group-out decoding requires at least three feature sets.")
    outer_ids = _normalize_outer_ids(normalized_sets, outer_ids)
    label_control = _normalize_label_control(label_control)
    resumed = _existing_artifact_rows(existing_artifacts)
    outer_rows = resumed["outer"]
    inner_rows = resumed["inner_validation"]
    selected_rows = resumed["selected"]
    prediction_rows = resumed["predictions"]
    completed_outer_ids = _completed_outer_ids(outer_rows)

    for test_id in outer_ids:
        if str(test_id) in completed_outer_ids:
            continue
        outer_train_sets, test_set = _split_train_test(normalized_sets, test_id)
        outer_inner_rows: list[dict[str, Any]] = []
        for candidate_index, candidate_config in enumerate(candidate_configs):
            for validation_set in outer_train_sets:
                inner_train_sets = [feature_set for feature_set in outer_train_sets if feature_set.id != validation_set.id]
                inner_row, _predictions = _evaluate_fold(
                    inner_train_sets,
                    validation_set,
                    candidate_config,
                    candidate_index=candidate_index,
                    label_control=label_control,
                    label_control_seed=label_control_seed,
                    label_control_context=("inner", test_id, validation_set.id, candidate_index),
                    include_predictions=False,
                )
                inner_row.update(
                    {
                        "selection_mode": "nested_loso",
                        "selection_metric": selection_metric,
                        "outer_test_group": test_id,
                        "inner_validation_group": validation_set.id,
                        "inner_train_groups": _join_group_ids(feature_set.id for feature_set in inner_train_sets),
                        "n_inner_train_groups": len(inner_train_sets),
                    }
                )
                outer_inner_rows.append(inner_row)

        selected_row = select_nested_candidate(outer_inner_rows, selection_metric=selection_metric)
        selected_config = candidate_configs[int(selected_row["selected_candidate_index"])]
        outer_row, fold_predictions = _evaluate_fold(
            outer_train_sets,
            test_set,
            selected_config,
            candidate_index=int(selected_row["selected_candidate_index"]),
            label_control=label_control,
            label_control_seed=label_control_seed,
            label_control_context=("outer", test_id, int(selected_row["selected_candidate_index"])),
            include_predictions=include_predictions,
        )
        _add_selected_candidate_fields(outer_row, selected_row)
        inner_rows.extend(outer_inner_rows)
        selected_rows.append(selected_row)
        outer_rows.append(outer_row)
        prediction_rows.extend(fold_predictions)
        if after_outer_fold is not None:
            after_outer_fold(_assemble_artifacts(outer_rows, inner_rows=inner_rows, selected_rows=selected_rows, prediction_rows=prediction_rows))

    return _assemble_artifacts(outer_rows, inner_rows=inner_rows, selected_rows=selected_rows, prediction_rows=prediction_rows)


def select_nested_candidate(inner_rows: Sequence[Mapping[str, Any]], *, selection_metric: str = "balanced_accuracy") -> dict[str, Any]:
    """Select the candidate with the best mean inner-validation metric."""
    if not inner_rows:
        raise ValueError("At least one inner-validation row is required.")
    summaries: list[dict[str, Any]] = []
    candidate_indices = sorted({int(row["candidate_index"]) for row in inner_rows})
    for candidate_index in candidate_indices:
        candidate_rows = [row for row in inner_rows if int(row["candidate_index"]) == candidate_index]
        values = _finite_metric_values(candidate_rows, selection_metric)
        example = candidate_rows[0]
        summary = {
            "selection_mode": "nested_loso",
            "selection_metric": selection_metric,
            "test_group": example.get("outer_test_group", ""),
            "selected_candidate_index": candidate_index,
            "selected_candidate_id": example.get("candidate_id", candidate_index),
            "n_candidates": len(candidate_indices),
            "n_inner_folds": len(candidate_rows),
            f"selected_inner_{selection_metric}_mean": _nanmean_or_nan(values),
            f"selected_inner_{selection_metric}_median": _nanmedian_or_nan(values),
            f"selected_inner_{selection_metric}_sem": _sem_or_nan(values),
        }
        for key, value in _candidate_identity_fields(example).items():
            summary[f"selected_{key}"] = value
        summaries.append(summary)

    ranked = sorted(
        summaries,
        key=lambda row: (
            _nan_to_negative_inf(row[f"selected_inner_{selection_metric}_mean"]),
            _nan_to_negative_inf(row[f"selected_inner_{selection_metric}_median"]),
            -int(row["selected_candidate_index"]),
        ),
        reverse=True,
    )
    selected = dict(ranked[0])
    selected_mean = float(selected[f"selected_inner_{selection_metric}_mean"])
    second_best = float(ranked[1][f"selected_inner_{selection_metric}_mean"]) if len(ranked) > 1 else np.nan
    selected[f"selected_inner_second_best_{selection_metric}_mean"] = second_best
    selected["selected_inner_winner_margin"] = selected_mean - second_best if np.isfinite(second_best) else np.nan
    return selected


def summarize_selected_candidate_stability(selected_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize how stable nested candidate selection was across outer folds."""
    if not selected_rows:
        return {}
    margins = _finite_metric_values(selected_rows, "selected_inner_winner_margin")
    summary = {
        "selected_candidate_counts": _format_counter(Counter(str(row.get("selected_candidate_index", "")) for row in selected_rows)),
        "selected_classifier_counts": _format_counter(Counter(str(row.get("selected_classifier", "")) for row in selected_rows)),
        "selected_components_pca_counts": _format_counter(Counter(str(row.get("selected_components_pca", "")) for row in selected_rows)),
        "inner_winner_margin_mean": _nanmean_or_nan(margins),
        "inner_winner_margin_median": _nanmedian_or_nan(margins),
        "inner_winner_margin_min": _nanmin_or_nan(margins),
    }
    metadata_keys = sorted({key.removeprefix("selected_metadata_") for row in selected_rows for key in row if key.startswith("selected_metadata_")})
    for key in metadata_keys:
        summary[f"selected_{key}_counts"] = _format_counter(Counter(str(row.get(f"selected_metadata_{key}", "")) for row in selected_rows))
    return summary


def completed_outer_ids(outer_rows: Sequence[Mapping[str, Any]], *, outer_column: str = "test_group") -> set[str]:
    """Return string-normalized completed outer ids from prior outer rows."""
    return _completed_outer_ids(outer_rows, outer_column=outer_column)


def _evaluate_fold(
    train_sets: Sequence[ParticipantFeatureSet],
    test_set: ParticipantFeatureSet,
    config: CrossSubjectCandidateConfig,
    *,
    candidate_index: int,
    label_control: str,
    label_control_seed: int | None,
    label_control_context: Sequence[Any],
    include_predictions: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    classifier_param = _resolved_classifier_param(config)
    train_features = np.vstack([feature_set.X for feature_set in train_sets])
    train_labels = np.concatenate(
        [
            _controlled_labels(
                feature_set.y,
                control=label_control,
                seed=label_control_seed,
                context=(*label_control_context, feature_set.id),
            )
            for feature_set in train_sets
        ]
    )
    bundle = fit_window_model(
        train_features,
        train_labels,
        fit_model=lambda features, labels: train_classifier(features, labels, config.classifier, classifier_param, random_state=config.random_state),
        components_pca=config.components_pca,
    )
    predictions, _scores = predict_window_model(bundle, test_set.X)
    class_scores, score_classes = _model_class_scores(bundle, test_set.X)
    rank_metrics = ranked_accuracy_metrics(test_set.y, class_scores, score_classes, top_ks=config.top_ks)
    accuracy = float(accuracy_score(test_set.y, predictions)) if len(test_set.y) else np.nan
    balanced_accuracy = float(balanced_accuracy_score(test_set.y, predictions)) if len(test_set.y) else np.nan
    chance_accuracy = _chance_accuracy(config, train_sets, test_set)
    train_class_counts = Counter(train_labels.tolist())
    test_class_counts = Counter(test_set.y.tolist())

    row: dict[str, Any] = {
        "outer_fold": test_set.id,
        "test_group": test_set.id,
        "train_groups": _join_group_ids(feature_set.id for feature_set in train_sets),
        "n_train_groups": len(train_sets),
        "n_test_groups": 1,
        "candidate_index": int(candidate_index),
        "candidate_id": config.id if config.id is not None else int(candidate_index),
        "classifier": config.classifier,
        "classifier_param": classifier_param,
        "components_pca": config.components_pca,
        "actual_components_pca": bundle.actual_components_pca,
        "pca_explained_variance_percent": bundle.explained_variance_percent,
        "accuracy": accuracy,
        "percent": 100.0 * accuracy,
        "balanced_accuracy": balanced_accuracy,
        "balanced_percent": 100.0 * balanced_accuracy,
        "chance_accuracy": chance_accuracy,
        "chance_percent": 100.0 * chance_accuracy,
        "mean_true_label_rank": rank_metrics["mean_true_label_rank"],
        "median_true_label_rank": rank_metrics["median_true_label_rank"],
        "chance_mean_rank": _chance_mean_rank(chance_accuracy),
        "above_chance": bool(balanced_accuracy > chance_accuracy) if np.isfinite(balanced_accuracy) else False,
        "n_train_trials": int(train_labels.shape[0]),
        "n_test_trials": int(test_set.y.shape[0]),
        "n_train_classes": int(len(train_class_counts)),
        "n_test_classes": int(len(test_class_counts)),
        "min_train_trials_per_class": int(min(train_class_counts.values())) if train_class_counts else 0,
        "min_test_trials_per_class": int(min(test_class_counts.values())) if test_class_counts else 0,
        "label_control": label_control,
        "label_control_seed": "" if label_control_seed is None else int(label_control_seed),
    }
    for top_k in config.top_ks:
        accuracy_key = f"top{int(top_k)}_accuracy"
        row[accuracy_key] = rank_metrics[accuracy_key]
        row[f"top{int(top_k)}_percent"] = 100.0 * float(rank_metrics[accuracy_key]) if np.isfinite(float(rank_metrics[accuracy_key])) else np.nan
        row[f"top{int(top_k)}_chance_accuracy"] = min(float(top_k) * chance_accuracy, 1.0)
        row[f"top{int(top_k)}_chance_percent"] = min(100.0 * float(top_k) * chance_accuracy, 100.0)
    _add_config_metadata(row, config)

    prediction_rows = (
        _prediction_rows(test_set, predictions, rank_metrics["true_label_ranks"], config=config, candidate_index=candidate_index, row_context=row)
        if include_predictions
        else []
    )
    return row, prediction_rows


def _prediction_rows(
    test_set: ParticipantFeatureSet,
    predictions: np.ndarray,
    true_label_ranks: np.ndarray,
    *,
    config: CrossSubjectCandidateConfig,
    candidate_index: int,
    row_context: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for trial_index, (true_label, predicted_label, true_label_rank) in enumerate(zip(test_set.y, predictions, true_label_ranks)):
        row = {
            "outer_fold": test_set.id,
            "test_group": test_set.id,
            "candidate_index": int(candidate_index),
            "candidate_id": config.id if config.id is not None else int(candidate_index),
            "classifier": config.classifier,
            "components_pca": config.components_pca,
            "actual_components_pca": row_context["actual_components_pca"],
            "trial": int(trial_index),
            "test_trial_index": int(trial_index),
            "true_label": true_label,
            "predicted_label": predicted_label,
            "correct": bool(predicted_label == true_label),
            "true_label_rank": float(true_label_rank) if np.isfinite(true_label_rank) else np.nan,
        }
        for top_k in config.top_ks:
            row[f"top{int(top_k)}_correct"] = bool(np.isfinite(true_label_rank) and true_label_rank <= int(top_k))
        _add_config_metadata(row, config)
        rows.append(row)
    return rows


def _model_class_scores(bundle: Any, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    transformed_features = transform_window_features(bundle, features)
    model = bundle.model
    classes = np.asarray(getattr(model, "classes_", np.unique(bundle.train_labels)))
    if hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(transformed_features), dtype=float)
    elif hasattr(model, "predict_proba"):
        scores = np.asarray(model.predict_proba(transformed_features), dtype=float)
    else:
        return np.full((transformed_features.shape[0], 0), np.nan, dtype=float), np.asarray([], dtype=object)
    if scores.ndim == 1:
        if classes.size != 2:
            return np.full((transformed_features.shape[0], 0), np.nan, dtype=float), np.asarray([], dtype=object)
        scores = np.column_stack((-scores, scores))
    if scores.ndim != 2 or scores.shape[1] != classes.size:
        return np.full((transformed_features.shape[0], 0), np.nan, dtype=float), np.asarray([], dtype=object)
    return scores, classes


def _assemble_artifacts(
    outer_rows: Sequence[Mapping[str, Any]],
    *,
    inner_rows: Sequence[Mapping[str, Any]] | None = None,
    selected_rows: Sequence[Mapping[str, Any]] | None = None,
    prediction_rows: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    outer = [dict(row) for row in outer_rows]
    selected = [] if selected_rows is None else [dict(row) for row in selected_rows]
    summary = summarize_cross_subject_scores(outer, selected_rows=selected)
    artifacts = {
        "outer": outer,
        "predictions": [] if prediction_rows is None else [dict(row) for row in prediction_rows],
        "group_summary": summary,
    }
    if inner_rows is not None:
        artifacts["inner_validation"] = [dict(row) for row in inner_rows]
    if selected_rows is not None:
        artifacts["selected"] = selected
    return artifacts


def summarize_cross_subject_scores(
    outer_rows: Sequence[Mapping[str, Any]],
    *,
    selected_rows: Sequence[Mapping[str, Any]] | None = None,
    signflip_permutations: int = 10_000,
    signflip_seed: int | None = 13,
) -> list[dict[str, Any]]:
    """Return one compact group row for outer held-out scores."""
    if not outer_rows:
        return []
    balanced_all = np.asarray([float(row.get("balanced_accuracy", np.nan)) for row in outer_rows], dtype=float)
    chance = np.asarray([float(row.get("chance_accuracy", np.nan)) for row in outer_rows], dtype=float)
    sign_mask = np.isfinite(balanced_all) & np.isfinite(chance)
    balanced = balanced_all[np.isfinite(balanced_all)]
    raw = _finite_metric_values(outer_rows, "accuracy")
    sign_summary = subject_level_signflip_summary(
        balanced_all[sign_mask],
        chance=chance[sign_mask],
        n_permutations=signflip_permutations,
        random_state=signflip_seed,
    )
    row: dict[str, Any] = {
        "n_outer_folds": len(outer_rows),
        "n_test_groups": len(outer_rows),
        "chance_accuracy": _nanmean_or_nan(chance),
        "chance_percent": 100.0 * _nanmean_or_nan(chance),
        "accuracy_mean": _nanmean_or_nan(raw),
        "accuracy_median": _nanmedian_or_nan(raw),
        "accuracy_sem": _sem_or_nan(raw),
        "percent_mean": 100.0 * _nanmean_or_nan(raw),
        "balanced_accuracy_mean": _nanmean_or_nan(balanced),
        "balanced_accuracy_median": _nanmedian_or_nan(balanced),
        "balanced_accuracy_sem": _sem_or_nan(balanced),
        "balanced_percent_mean": 100.0 * _nanmean_or_nan(balanced),
        "mean_above_chance": sign_summary["effect_mean"],
        "percent_above_chance": 100.0 * float(sign_summary["effect_mean"]) if np.isfinite(float(sign_summary["effect_mean"])) else np.nan,
        "participants_above_chance": sign_summary["n_above_chance"],
        "participants_total": sign_summary["n_subjects"],
        "one_sided_exact_sign_p_value": sign_summary["one_sided_exact_sign_p_value"],
        "one_sided_signflip_p_value": sign_summary["one_sided_signflip_p_value"],
    }
    top_keys = sorted({key for outer_row in outer_rows for key in outer_row if key.startswith("top") and key.endswith("_accuracy")})
    for key in top_keys:
        values = _finite_metric_values(outer_rows, key)
        row[f"{key}_mean"] = _nanmean_or_nan(values)
        row[f"{key}_sem"] = _sem_or_nan(values)
    rank_values = _finite_metric_values(outer_rows, "mean_true_label_rank")
    row["mean_true_label_rank_mean"] = _nanmean_or_nan(rank_values)
    row["mean_true_label_rank_sem"] = _sem_or_nan(rank_values)
    if selected_rows:
        row.update(summarize_selected_candidate_stability(selected_rows))
    return [row]


def _normalize_feature_sets(feature_sets: Sequence[ParticipantFeatureSet]) -> tuple[ParticipantFeatureSet, ...]:
    normalized = []
    for feature_set in feature_sets:
        features = np.asarray(feature_set.X, dtype=float)
        labels = np.asarray(feature_set.y).ravel()
        if features.ndim != 2:
            raise ValueError("ParticipantFeatureSet.X must be a two-dimensional feature matrix.")
        if features.shape[0] == 0:
            raise ValueError("ParticipantFeatureSet.X must contain at least one row.")
        if labels.ndim != 1 or labels.shape[0] != features.shape[0]:
            raise ValueError("ParticipantFeatureSet.y must be one-dimensional and match feature rows.")
        normalized.append(ParticipantFeatureSet(id=feature_set.id, X=features, y=labels, metadata=dict(feature_set.metadata)))
    if len(normalized) < 2:
        raise ValueError("At least two feature sets are required.")
    feature_counts = {feature_set.X.shape[1] for feature_set in normalized}
    if len(feature_counts) != 1:
        raise ValueError("All feature sets must have the same number of feature columns.")
    ids = [str(feature_set.id) for feature_set in normalized]
    if len(set(ids)) != len(ids):
        raise ValueError("ParticipantFeatureSet ids must be unique after string normalization.")
    return tuple(normalized)


def _normalize_outer_ids(feature_sets: Sequence[ParticipantFeatureSet], outer_ids: Sequence[Any] | None) -> tuple[Any, ...]:
    available = {str(feature_set.id): feature_set.id for feature_set in feature_sets}
    if outer_ids is None:
        return tuple(feature_set.id for feature_set in feature_sets)
    unknown = [outer_id for outer_id in outer_ids if str(outer_id) not in available]
    if unknown:
        raise ValueError(f"outer_ids must be present in feature_sets: {unknown}")
    return tuple(available[str(outer_id)] for outer_id in outer_ids)


def _split_train_test(feature_sets: Sequence[ParticipantFeatureSet], test_id: Any) -> tuple[list[ParticipantFeatureSet], ParticipantFeatureSet]:
    test_matches = [feature_set for feature_set in feature_sets if str(feature_set.id) == str(test_id)]
    if len(test_matches) != 1:
        raise ValueError(f"Expected exactly one feature set for test id {test_id!r}.")
    train_sets = [feature_set for feature_set in feature_sets if str(feature_set.id) != str(test_id)]
    return train_sets, test_matches[0]


def _resolved_classifier_param(config: CrossSubjectCandidateConfig) -> Any:
    classifier_param = config.classifier_param
    if should_use_default_classifier_param(classifier_param):
        return get_default_classifier_param(config.classifier)
    return classifier_param


def _chance_accuracy(config: CrossSubjectCandidateConfig, train_sets: Sequence[ParticipantFeatureSet], test_set: ParticipantFeatureSet) -> float:
    if config.chance_accuracy is not None:
        return float(config.chance_accuracy)
    labels = np.concatenate([feature_set.y for feature_set in [*train_sets, test_set]])
    n_classes = len(np.unique(labels))
    return 1.0 / n_classes if n_classes else np.nan


def _chance_mean_rank(chance_accuracy: float) -> float:
    if not np.isfinite(chance_accuracy) or chance_accuracy <= 0:
        return np.nan
    return 0.5 * ((1.0 / chance_accuracy) + 1.0)


def _controlled_labels(labels: np.ndarray, *, control: str, seed: int | None, context: Sequence[Any]) -> np.ndarray:
    labels = np.asarray(labels).copy()
    if control == "none":
        return labels
    rng = np.random.default_rng(_label_control_seed(seed, context))
    if control == "shuffle":
        rng.shuffle(labels)
        return labels
    if control == "circular_shift":
        if labels.size <= 1:
            return labels
        shift = int(rng.integers(1, labels.size))
        return np.roll(labels, shift)
    raise ValueError(f"Unsupported label_control: {control}")


def _normalize_label_control(label_control: str) -> str:
    normalized = str(label_control).strip().lower().replace("-", "_")
    aliases = {
        "": "none",
        "none": "none",
        "label_shuffle": "shuffle",
        "shuffle": "shuffle",
        "shuffled": "shuffle",
        "circular": "circular_shift",
        "circular_shift": "circular_shift",
        "circular_label_shift": "circular_shift",
    }
    if normalized not in aliases:
        raise ValueError("label_control must be one of: none, shuffle, circular_shift.")
    return aliases[normalized]


def _label_control_seed(seed: int | None, context: Sequence[Any]) -> np.random.SeedSequence:
    seed_values = [0 if seed is None else int(seed)]
    seed_values.extend(sum(ord(character) for character in str(value)) for value in context)
    return np.random.SeedSequence(seed_values)


def _existing_artifact_rows(existing_artifacts: Mapping[str, Sequence[Mapping[str, Any]]] | None) -> dict[str, list[dict[str, Any]]]:
    if existing_artifacts is None:
        existing_artifacts = {}
    return {
        "outer": [dict(row) for row in existing_artifacts.get("outer", [])],
        "inner_validation": [dict(row) for row in existing_artifacts.get("inner_validation", [])],
        "selected": [dict(row) for row in existing_artifacts.get("selected", [])],
        "predictions": [dict(row) for row in existing_artifacts.get("predictions", [])],
    }


def _completed_outer_ids(outer_rows: Sequence[Mapping[str, Any]], *, outer_column: str = "test_group") -> set[str]:
    return {str(row[outer_column]) for row in outer_rows if outer_column in row}


def _add_selected_candidate_fields(row: dict[str, Any], selected_row: Mapping[str, Any]) -> None:
    for key, value in selected_row.items():
        row[key] = value


def _candidate_identity_fields(row: Mapping[str, Any]) -> dict[str, Any]:
    keys = ("candidate_id", "classifier", "classifier_param", "components_pca", "label_control", "label_control_seed")
    result = {key: row.get(key, "") for key in keys}
    for key, value in row.items():
        if key.startswith("metadata_"):
            result[key] = value
    return result


def _add_config_metadata(row: dict[str, Any], config: CrossSubjectCandidateConfig) -> None:
    for key, value in config.metadata.items():
        column = f"metadata_{key}"
        row[column] = value


def _join_group_ids(group_ids: Sequence[Any]) -> str:
    return ",".join(str(group_id) for group_id in group_ids)


def _finite_metric_values(rows: Sequence[Mapping[str, Any]], key: str) -> np.ndarray:
    values = np.asarray([float(row.get(key, np.nan)) for row in rows], dtype=float)
    return values[np.isfinite(values)]


def _nan_to_negative_inf(value: Any) -> float:
    value = float(value)
    return value if np.isfinite(value) else -np.inf


def _nanmean_or_nan(values: Sequence[float] | np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan
    return float(np.mean(values))


def _nanmedian_or_nan(values: Sequence[float] | np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan
    return float(np.median(values))


def _nanmin_or_nan(values: Sequence[float] | np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan
    return float(np.min(values))


def _sem_or_nan(values: Sequence[float] | np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size <= 1:
        return 0.0 if values.size == 1 else np.nan
    return float(np.std(values, ddof=1) / np.sqrt(values.size))


def _format_counter(counter: Counter) -> str:
    return ";".join(f"{key}:{counter[key]}" for key in sorted(counter, key=str))
