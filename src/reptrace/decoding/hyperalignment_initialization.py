"""Template-initialized Procrustes hyperalignment utilities.

This module extends the base hyperalignment implementation with selectable
initialization strategies while preserving the existing PCA/SVD initialization
as the default behavior.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence

import numpy as np

from reptrace.decoding.hyperalignment import (
    CLASS_ALIGNMENT_SAMPLE_MODES,
    ClassAlignment,
    HyperalignmentModel,
    SubjectHyperalignmentProjection,
    _average_projection,
    _check_common_alignment_rows,
    _feature_matrix,
    _initial_projection,
    _normalize_template,
    _orthogonal_procrustes_projection,
    _requested_component_count,
    class_alignment_matrices,
    fit_hyperalignment as _fit_pca_hyperalignment,
    fit_projection_to_hyperalignment,
    transform_with_projection,
)
from reptrace.decoding.sampling import DEFAULT_CLASS_LIMIT_SEED, DEFAULT_CLASS_LIMIT_SELECTION

HYPERALIGNMENT_INITIALIZATION_MODES = ("pca", "mean")


def fit_hyperalignment(
    aligned_by_subject: Mapping[Hashable, Sequence[Sequence[float]] | np.ndarray],
    *,
    n_components: int | float = 64,
    n_iterations: int = 10,
    template_tolerance: float = 1e-8,
    initialization: str = "pca",
) -> HyperalignmentModel:
    """Fit Procrustes hyperalignment with a selectable template initialization.

    ``initialization="pca"`` delegates to RepTrace's original implementation.
    ``initialization="mean"`` initializes the common template from the grand
    mean of the centered alignment matrices before iterative Procrustes updates.
    Mean initialization requires equal feature dimensionality across subjects.
    """

    initialization = normalize_hyperalignment_initialization(initialization)
    if initialization == "pca":
        return _fit_pca_hyperalignment(
            aligned_by_subject,
            n_components=n_components,
            n_iterations=n_iterations,
            template_tolerance=template_tolerance,
        )
    return _fit_mean_initialized_hyperalignment(
        aligned_by_subject,
        n_components=n_components,
        n_iterations=n_iterations,
        template_tolerance=template_tolerance,
    )


def fit_class_hyperalignment(
    features_by_subject: Mapping[Hashable, Sequence[Sequence[float]] | np.ndarray],
    labels_by_subject: Mapping[Hashable, Sequence | np.ndarray],
    *,
    sample_mode: str = "class_mean",
    n_repetitions_per_class: int | None = None,
    repetition_selection: str = DEFAULT_CLASS_LIMIT_SELECTION,
    repetition_seed: int | str | None = DEFAULT_CLASS_LIMIT_SEED,
    n_components: int | float = 64,
    n_iterations: int = 10,
    template_tolerance: float = 1e-8,
    initialization: str = "pca",
) -> tuple[HyperalignmentModel, ClassAlignment]:
    """Build class anchors and fit initialized Procrustes hyperalignment."""

    alignment = class_alignment_matrices(
        features_by_subject,
        labels_by_subject,
        sample_mode=sample_mode,
        n_repetitions_per_class=n_repetitions_per_class,
        repetition_selection=repetition_selection,
        repetition_seed=repetition_seed,
    )
    model = fit_hyperalignment(
        alignment.aligned_by_subject,
        n_components=n_components,
        n_iterations=n_iterations,
        template_tolerance=template_tolerance,
        initialization=initialization,
    )
    return model, alignment


def normalize_hyperalignment_initialization(initialization: str) -> str:
    """Normalize and validate a hyperalignment initialization mode."""

    normalized = str(initialization).strip().lower().replace("-", "_")
    if normalized not in HYPERALIGNMENT_INITIALIZATION_MODES:
        raise ValueError(
            f"Unsupported hyperalignment initialization: {initialization}. "
            f"Supported modes: {', '.join(HYPERALIGNMENT_INITIALIZATION_MODES)}."
        )
    return normalized


def _fit_mean_initialized_hyperalignment(
    aligned_by_subject: Mapping[Hashable, Sequence[Sequence[float]] | np.ndarray],
    *,
    n_components: int | float,
    n_iterations: int,
    template_tolerance: float,
) -> HyperalignmentModel:
    if len(aligned_by_subject) < 2:
        raise ValueError("Hyperalignment requires at least two subjects.")
    if n_iterations < 1:
        raise ValueError("n_iterations must be positive.")

    subject_ids = tuple(aligned_by_subject.keys())
    matrices = {subject_id: _feature_matrix(matrix, name=f"aligned_by_subject[{subject_id!r}]") for subject_id, matrix in aligned_by_subject.items()}
    n_rows = _check_common_alignment_rows(matrices)
    if n_rows < 2:
        raise ValueError("Hyperalignment requires at least two aligned rows per subject.")

    feature_dims = {matrix.shape[1] for matrix in matrices.values()}
    if len(feature_dims) != 1:
        raise ValueError("Mean hyperalignment initialization requires all subjects to have the same feature dimension.")

    requested = _requested_component_count(n_components)
    actual = min(requested, n_rows - 1, next(iter(feature_dims)))
    if actual < 1:
        raise ValueError("No hyperalignment components are available.")

    means = {subject_id: np.mean(matrix, axis=0) for subject_id, matrix in matrices.items()}
    centered = {subject_id: matrices[subject_id] - means[subject_id] for subject_id in subject_ids}
    mean_centered = np.mean(np.stack([centered[subject_id] for subject_id in subject_ids], axis=0), axis=0)
    mean_projection = _initial_projection(mean_centered, actual)
    template = _normalize_template(mean_centered @ mean_projection)
    projections = {subject_id: _orthogonal_procrustes_projection(centered[subject_id], template) for subject_id in subject_ids}

    for _ in range(int(n_iterations)):
        new_projections = {subject_id: _orthogonal_procrustes_projection(centered[subject_id], template) for subject_id in subject_ids}
        new_template = _normalize_template(
            np.mean(np.stack([centered[subject_id] @ new_projections[subject_id] for subject_id in subject_ids], axis=0), axis=0)
        )
        delta = float(np.linalg.norm(new_template - template) / max(np.linalg.norm(template), 1e-12))
        projections = new_projections
        template = new_template
        if delta < template_tolerance:
            break

    projection_objects = {
        subject_id: SubjectHyperalignmentProjection(
            subject_id=subject_id,
            feature_mean=means[subject_id],
            projection=projections[subject_id],
            n_alignment_rows=n_rows,
        )
        for subject_id in subject_ids
    }
    group_feature_mean, group_projection = _average_projection(projection_objects)
    return HyperalignmentModel(
        subject_ids=subject_ids,
        n_components=actual,
        n_iterations=int(n_iterations),
        projections=projection_objects,
        template=template,
        group_feature_mean=group_feature_mean,
        group_projection=group_projection,
    )


__all__ = [
    "CLASS_ALIGNMENT_SAMPLE_MODES",
    "HYPERALIGNMENT_INITIALIZATION_MODES",
    "ClassAlignment",
    "HyperalignmentModel",
    "SubjectHyperalignmentProjection",
    "class_alignment_matrices",
    "fit_class_hyperalignment",
    "fit_hyperalignment",
    "fit_projection_to_hyperalignment",
    "normalize_hyperalignment_initialization",
    "transform_with_projection",
]
