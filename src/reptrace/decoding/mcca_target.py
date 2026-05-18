"""Target-subject projection helpers for RepTrace M-CCA models.

These utilities cover the common held-out-subject calibration case: a multiset
CCA model is fitted on training subjects, a small row-aligned calibration matrix
is available for a target subject, and the target features should be projected
into the learned M-CCA component space without adding dataset-specific loading or
windowing assumptions to RepTrace.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from reptrace.decoding.mcca import (
    MCCAModel,
    _class_mean_matrix,
    _class_repetition_matrix,
    _feature_matrix,
    _label_vector,
    _normalize_sample_mode,
)
from reptrace.decoding.sampling import DEFAULT_CLASS_LIMIT_SEED, DEFAULT_CLASS_LIMIT_SELECTION


@dataclass(frozen=True)
class TargetMCCAProjection:
    """Linear map from a calibrated held-out subject into an M-CCA space.

    The projection is fitted from row-aligned target calibration features to an
    existing M-CCA component template.  Use :meth:`transform` when the scored
    features have the same feature layout as the calibration rows.  Projects
    with dataset-specific window or channel adapters can apply ``projection``
    and ``feature_mean`` themselves, then call :meth:`add_template_mean` on the
    projected rows.
    """

    feature_mean: np.ndarray
    template_mean: np.ndarray
    projection: np.ndarray
    regularization: float
    n_alignment_rows: int

    def transform(self, features: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
        """Transform target-subject rows into the fitted component space."""

        matrix = _feature_matrix(features, name="features")
        if matrix.shape[1] != self.projection.shape[0]:
            raise ValueError(
                "features column count does not match the target M-CCA projection: "
                f"{matrix.shape[1]} != {self.projection.shape[0]}."
            )
        return self.add_template_mean((matrix - self.feature_mean) @ self.projection, strict=True)

    def add_template_mean(self, transformed: Sequence[Sequence[float]] | np.ndarray, *, strict: bool = False) -> np.ndarray:
        """Add the fitted template mean to already projected target rows.

        When ``transformed`` contains repeated component blocks, the template
        mean is tiled across blocks.  With ``strict=False`` incompatible column
        counts are returned unchanged for compatibility with dataset-specific
        projection adapters.
        """

        return _add_template_mean(transformed, self.template_mean, strict=strict)


def class_alignment_matrix(
    features: Sequence[Sequence[float]] | np.ndarray,
    labels: Sequence | np.ndarray,
    *,
    classes: Sequence | np.ndarray | None = None,
    sample_mode: str = "class_mean",
    n_repetitions_per_class: int | None = None,
    repetition_selection: str = DEFAULT_CLASS_LIMIT_SELECTION,
    repetition_seed: int | str | None = DEFAULT_CLASS_LIMIT_SEED,
) -> np.ndarray:
    """Build one subject's class-aligned feature matrix.

    ``classes`` fixes the row order.  This is useful for held-out target
    calibration where the target rows must match an already fitted class
    alignment or M-CCA component template. For ``class_repetition``, use the
    same ``repetition_selection`` and ``repetition_seed`` as the fitted template.
    """

    sample_mode = _normalize_sample_mode(sample_mode)
    matrix = _feature_matrix(features, name="features")
    vector = _label_vector(labels, expected_length=matrix.shape[0], name="labels")
    if classes is None:
        class_order = np.unique(vector)
    else:
        class_order = np.asarray(classes).ravel()
    _check_requested_classes(vector, class_order)

    if sample_mode == "class_mean":
        return _class_mean_matrix(matrix, vector, class_order)

    if n_repetitions_per_class is None:
        n_repetitions_per_class = _minimum_class_count(vector, class_order)
    repetitions = int(n_repetitions_per_class)
    if repetitions < 1:
        raise ValueError("n_repetitions_per_class must be positive or None.")
    return _class_repetition_matrix(
        matrix,
        vector,
        class_order,
        repetitions,
        selection=repetition_selection,
        seed=repetition_seed,
    )


def fit_target_mcca_projection(
    features: Sequence[Sequence[float]] | np.ndarray,
    template: MCCAModel | Sequence[Sequence[float]] | np.ndarray,
    *,
    regularization: float | None = None,
) -> TargetMCCAProjection:
    """Fit a calibrated target-subject projection into an M-CCA space.

    Parameters
    ----------
    features:
        Row-aligned target calibration matrix.  The row order must match the
        fitted M-CCA template.
    template:
        Either a fitted :class:`~reptrace.decoding.mcca.MCCAModel` or an explicit
        component-template matrix with the same number of rows as ``features``.
        When a model is passed, ``model.component_scores`` is used.
    regularization:
        Non-negative ridge term for the target projection solve.  When omitted
        and ``template`` is an ``MCCAModel``, the model regularization is reused;
        otherwise a default of ``1e-6`` is used.
    """

    matrix = _feature_matrix(features, name="features")
    if isinstance(template, MCCAModel):
        template_matrix = template.component_scores
        if regularization is None:
            regularization = template.regularization
    else:
        template_matrix = _feature_matrix(template, name="template")
        if regularization is None:
            regularization = 1e-6
    regularization = float(regularization)
    if regularization < 0:
        raise ValueError("regularization must be non-negative.")
    if matrix.shape[0] != template_matrix.shape[0]:
        raise ValueError(
            "Target alignment rows must match the M-CCA template rows: "
            f"{matrix.shape[0]} != {template_matrix.shape[0]}."
        )

    feature_mean = np.mean(matrix, axis=0)
    template_mean = np.mean(template_matrix, axis=0)
    centered = matrix - feature_mean
    centered_template = template_matrix - template_mean
    gram = centered @ centered.T
    regularized = gram + regularization * np.eye(gram.shape[0], dtype=float)
    try:
        dual_weights = np.linalg.solve(regularized, centered_template)
    except np.linalg.LinAlgError:
        dual_weights = np.linalg.pinv(regularized) @ centered_template
    projection = centered.T @ dual_weights
    return TargetMCCAProjection(
        feature_mean=feature_mean,
        template_mean=template_mean,
        projection=projection,
        regularization=regularization,
        n_alignment_rows=int(matrix.shape[0]),
    )


def _check_requested_classes(labels: np.ndarray, classes: np.ndarray) -> None:
    if classes.size == 0:
        raise ValueError("classes must contain at least one label.")
    missing = [label for label in classes if not np.any(labels == label)]
    if missing:
        raise ValueError(f"classes include labels absent from labels: {missing!r}.")


def _minimum_class_count(labels: np.ndarray, classes: np.ndarray) -> int:
    return min(int(np.sum(labels == class_label)) for class_label in classes)


def _add_template_mean(transformed: Sequence[Sequence[float]] | np.ndarray, template_mean: Sequence[float] | np.ndarray, *, strict: bool) -> np.ndarray:
    matrix = _feature_matrix(transformed, name="transformed")
    mean = np.asarray(template_mean, dtype=float).ravel()
    if mean.size == 0:
        raise ValueError("template_mean must contain at least one value.")
    if not np.all(np.isfinite(mean)):
        raise ValueError("template_mean contains non-finite values.")
    if matrix.shape[1] == mean.shape[0]:
        return matrix + mean
    if matrix.shape[1] % mean.shape[0] == 0:
        repeats = matrix.shape[1] // mean.shape[0]
        return matrix + np.tile(mean, repeats)
    if strict:
        raise ValueError(
            "transformed column count must match or be a multiple of the template width: "
            f"{matrix.shape[1]} vs {mean.shape[0]}."
        )
    return matrix


__all__ = ["TargetMCCAProjection", "class_alignment_matrix", "fit_target_mcca_projection"]