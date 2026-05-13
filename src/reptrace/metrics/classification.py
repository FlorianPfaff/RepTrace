from __future__ import annotations

from collections.abc import Sequence
from math import comb
from typing import Any

import numpy as np


def true_label_ranks(
    true_labels: Sequence[Any] | np.ndarray,
    class_scores: Sequence[Sequence[float]] | np.ndarray,
    score_classes: Sequence[Any] | np.ndarray,
) -> np.ndarray:
    """Return one-based ranks of the true class in each row of class scores.

    Higher scores are better. Ties are resolved by the order of ``score_classes``
    through a stable sort, which makes rank-based summaries deterministic.
    Missing true labels receive ``NaN`` ranks.
    """
    labels = np.asarray(true_labels)
    scores = np.asarray(class_scores, dtype=float)
    classes = np.asarray(score_classes)

    if labels.ndim != 1:
        raise ValueError("true_labels must be one-dimensional.")
    if scores.ndim != 2:
        raise ValueError("class_scores must be a two-dimensional score matrix.")
    if scores.shape[0] != labels.shape[0]:
        raise ValueError("true_labels and class_scores must contain the same samples.")
    if scores.shape[1] == 0:
        return np.full(labels.shape[0], np.nan, dtype=float)
    if classes.ndim != 1 or classes.shape[0] != scores.shape[1]:
        raise ValueError("score_classes must contain one label per score column.")

    label_to_column = {label: column for column, label in enumerate(classes.tolist())}
    ranks: list[float] = []
    for true_label, row_scores in zip(labels.tolist(), scores):
        true_column = label_to_column.get(true_label)
        if true_column is None:
            ranks.append(np.nan)
            continue
        descending_columns = np.argsort(-row_scores, kind="mergesort")
        rank_locations = np.flatnonzero(descending_columns == true_column)
        ranks.append(float(rank_locations[0] + 1) if rank_locations.size else np.nan)
    return np.asarray(ranks, dtype=float)


def ranked_accuracy_metrics(
    true_labels: Sequence[Any] | np.ndarray,
    class_scores: Sequence[Sequence[float]] | np.ndarray,
    score_classes: Sequence[Any] | np.ndarray,
    *,
    top_ks: Sequence[int] = (2, 3),
) -> dict[str, float | np.ndarray]:
    """Summarize top-k accuracy and true-label ranks from class scores."""
    ranks = true_label_ranks(true_labels, class_scores, score_classes)
    finite_ranks = ranks[np.isfinite(ranks)]
    metrics: dict[str, float | np.ndarray] = {"true_label_ranks": ranks}

    normalized_top_ks = _normalize_top_ks(top_ks)
    if finite_ranks.size == 0:
        for k in normalized_top_ks:
            metrics[f"top{k}_accuracy"] = np.nan
        metrics["mean_true_label_rank"] = np.nan
        metrics["median_true_label_rank"] = np.nan
        return metrics

    for k in normalized_top_ks:
        metrics[f"top{k}_accuracy"] = float(np.mean(finite_ranks <= k))
    metrics["mean_true_label_rank"] = float(np.mean(finite_ranks))
    metrics["median_true_label_rank"] = float(np.median(finite_ranks))
    return metrics


def exact_one_sided_sign_p_value(differences: Sequence[float] | np.ndarray) -> float:
    """Return the exact one-sided binomial sign-test p-value for positive effects."""
    finite = _finite_differences(differences)
    if finite.size == 0:
        return np.nan
    n_positive = int(np.sum(finite > 0.0))
    n_total = int(finite.size)
    tail_count = sum(comb(n_total, k) for k in range(n_positive, n_total + 1))
    return float(tail_count / (2**n_total))


def one_sided_signflip_p_value(
    differences: Sequence[float] | np.ndarray,
    *,
    n_permutations: int = 10_000,
    random_state: int | None = 13,
    max_exact_subjects: int = 16,
) -> float:
    """Return a one-sided subject-level sign-flip p-value for mean effects."""
    finite = _finite_differences(differences)
    if finite.size == 0:
        return np.nan
    if n_permutations < 1:
        raise ValueError("n_permutations must be at least 1.")
    observed = float(np.mean(finite))
    if observed <= 0.0:
        return 1.0
    if finite.size <= max_exact_subjects:
        signs = np.array(np.meshgrid(*[[-1.0, 1.0]] * finite.size)).T.reshape(-1, finite.size)
        null_means = signs @ finite / finite.size
        return float(np.mean(null_means >= observed))

    rng = np.random.default_rng(random_state)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(int(n_permutations), finite.size))
    null_means = signs @ finite / finite.size
    return float((np.sum(null_means >= observed) + 1.0) / (int(n_permutations) + 1.0))


def subject_level_signflip_summary(
    values: Sequence[float] | np.ndarray,
    *,
    chance: float | Sequence[float] | np.ndarray = 0.0,
    n_permutations: int = 10_000,
    random_state: int | None = 13,
) -> dict[str, float | int]:
    """Summarize subject-level metric effects against chance with sign tests."""
    values_array = np.asarray(values, dtype=float)
    chance_array = _chance_array(chance, values_array.shape)
    differences = values_array - chance_array
    finite = _finite_differences(differences)
    finite_values = values_array[np.isfinite(values_array) & np.isfinite(chance_array)]

    if finite.size == 0:
        return {
            "n_subjects": 0,
            "n_above_chance": 0,
            "value_mean": np.nan,
            "value_median": np.nan,
            "effect_mean": np.nan,
            "effect_median": np.nan,
            "one_sided_exact_sign_p_value": np.nan,
            "one_sided_signflip_p_value": np.nan,
        }

    return {
        "n_subjects": int(finite.size),
        "n_above_chance": int(np.sum(finite > 0.0)),
        "value_mean": float(np.mean(finite_values)),
        "value_median": float(np.median(finite_values)),
        "effect_mean": float(np.mean(finite)),
        "effect_median": float(np.median(finite)),
        "one_sided_exact_sign_p_value": exact_one_sided_sign_p_value(finite),
        "one_sided_signflip_p_value": one_sided_signflip_p_value(
            finite,
            n_permutations=n_permutations,
            random_state=random_state,
        ),
    }


def _normalize_top_ks(top_ks: Sequence[int]) -> list[int]:
    normalized: list[int] = []
    for top_k in top_ks:
        top_k = int(top_k)
        if top_k < 1:
            raise ValueError("top-k values must be positive.")
        if top_k not in normalized:
            normalized.append(top_k)
    return normalized


def _finite_differences(differences: Sequence[float] | np.ndarray) -> np.ndarray:
    differences = np.asarray(differences, dtype=float).ravel()
    return differences[np.isfinite(differences)]


def _chance_array(chance: float | Sequence[float] | np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    chance_array = np.asarray(chance, dtype=float)
    if chance_array.ndim == 0:
        return np.full(shape, float(chance_array), dtype=float)
    if chance_array.shape != shape:
        raise ValueError("chance must be scalar or have the same shape as values.")
    return chance_array
