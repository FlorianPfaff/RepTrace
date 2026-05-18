"""Procrustes hyperalignment utilities for cross-subject feature decoding.

The caller supplies row-aligned subject matrices, e.g. one row per class or per
class/repetition anchor. The fitted model stores subject-specific semi-orthogonal
maps into a common representational space and an average projection for
calibration-free held-out-subject baselines.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from reptrace.decoding.sampling import (
    DEFAULT_CLASS_LIMIT_SEED,
    DEFAULT_CLASS_LIMIT_SELECTION,
    normalize_class_limit_seed,
    normalize_class_limit_selection,
    select_class_limited_indices,
)

CLASS_ALIGNMENT_SAMPLE_MODES = ("class_mean", "class_repetition")


@dataclass(frozen=True)
class SubjectHyperalignmentProjection:
    subject_id: Hashable
    feature_mean: np.ndarray
    projection: np.ndarray
    n_alignment_rows: int


@dataclass(frozen=True)
class HyperalignmentModel:
    subject_ids: tuple[Hashable, ...]
    n_components: int
    n_iterations: int
    projections: Mapping[Hashable, SubjectHyperalignmentProjection]
    template: np.ndarray
    group_feature_mean: np.ndarray | None
    group_projection: np.ndarray | None

    def transform(self, subject_id: Hashable, features: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
        try:
            projection = self.projections[subject_id]
        except KeyError as exc:
            fitted = ", ".join(str(value) for value in self.subject_ids)
            raise KeyError(f"Unknown hyperalignment subject {subject_id!r}. Fitted subjects: {fitted}.") from exc
        return transform_with_projection(features, projection)

    def transform_group(self, features: Sequence[Sequence[float]] | np.ndarray, *, feature_mean: Sequence[float] | np.ndarray | None = None) -> np.ndarray:
        if self.group_projection is None or self.group_feature_mean is None:
            raise ValueError("A group projection is unavailable because fitted subjects have incompatible feature dimensions.")
        matrix = _feature_matrix(features, name="features")
        if matrix.shape[1] != self.group_projection.shape[0]:
            raise ValueError(f"features column count does not match the group projection: {matrix.shape[1]} != {self.group_projection.shape[0]}.")
        mean = self.group_feature_mean if feature_mean is None else np.asarray(feature_mean, dtype=float).ravel()
        if mean.shape[0] != matrix.shape[1]:
            raise ValueError(f"feature_mean length must match features columns: {mean.shape[0]} != {matrix.shape[1]}.")
        return (matrix - mean) @ self.group_projection


@dataclass(frozen=True)
class ClassAlignment:
    aligned_by_subject: Mapping[Hashable, np.ndarray]
    classes: np.ndarray
    sample_mode: str
    n_repetitions_per_class: int | None
    repetition_selection: str | None = None
    repetition_seed: int | None = None


def fit_hyperalignment(
    aligned_by_subject: Mapping[Hashable, Sequence[Sequence[float]] | np.ndarray],
    *,
    n_components: int | float = 64,
    n_iterations: int = 10,
    template_tolerance: float = 1e-8,
) -> HyperalignmentModel:
    """Fit an iterative Procrustes common-space model from row-aligned matrices."""

    if len(aligned_by_subject) < 2:
        raise ValueError("Hyperalignment requires at least two subjects.")
    if n_iterations < 1:
        raise ValueError("n_iterations must be positive.")

    subject_ids = tuple(aligned_by_subject.keys())
    matrices = {sid: _feature_matrix(mat, name=f"aligned_by_subject[{sid!r}]") for sid, mat in aligned_by_subject.items()}
    n_rows = _check_common_alignment_rows(matrices)
    if n_rows < 2:
        raise ValueError("Hyperalignment requires at least two aligned rows per subject.")
    requested = _requested_component_count(n_components)
    actual = min(requested, n_rows - 1, *(matrix.shape[1] for matrix in matrices.values()))
    if actual < 1:
        raise ValueError("No hyperalignment components are available.")

    means = {sid: np.mean(matrix, axis=0) for sid, matrix in matrices.items()}
    centered = {sid: matrices[sid] - means[sid] for sid in subject_ids}
    projections = {sid: _initial_projection(centered[sid], actual) for sid in subject_ids}
    template = _normalize_template(np.mean(np.stack([centered[sid] @ projections[sid] for sid in subject_ids], axis=0), axis=0))

    for _ in range(int(n_iterations)):
        new_projections = {sid: _orthogonal_procrustes_projection(centered[sid], template) for sid in subject_ids}
        new_template = _normalize_template(np.mean(np.stack([centered[sid] @ new_projections[sid] for sid in subject_ids], axis=0), axis=0))
        delta = float(np.linalg.norm(new_template - template) / max(np.linalg.norm(template), 1e-12))
        projections = new_projections
        template = new_template
        if delta < template_tolerance:
            break

    projection_objects = {
        sid: SubjectHyperalignmentProjection(
            subject_id=sid,
            feature_mean=means[sid],
            projection=projections[sid],
            n_alignment_rows=n_rows,
        )
        for sid in subject_ids
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


def fit_projection_to_hyperalignment(
    features: Sequence[Sequence[float]] | np.ndarray,
    *,
    template: Sequence[Sequence[float]] | np.ndarray,
) -> SubjectHyperalignmentProjection:
    """Fit one subject projection to an existing template from labeled anchors."""

    matrix = _feature_matrix(features, name="features")
    template_matrix = _feature_matrix(template, name="template")
    if matrix.shape[0] != template_matrix.shape[0]:
        raise ValueError(f"features and template need the same row count: {matrix.shape[0]} != {template_matrix.shape[0]}.")
    mean = np.mean(matrix, axis=0)
    projection = _orthogonal_procrustes_projection(matrix - mean, template_matrix)
    return SubjectHyperalignmentProjection(
        subject_id="target",
        feature_mean=mean,
        projection=projection,
        n_alignment_rows=matrix.shape[0],
    )


def transform_with_projection(
    features: Sequence[Sequence[float]] | np.ndarray,
    projection: SubjectHyperalignmentProjection,
) -> np.ndarray:
    matrix = _feature_matrix(features, name="features")
    if matrix.shape[1] != projection.projection.shape[0]:
        raise ValueError(f"features column count does not match projection: {matrix.shape[1]} != {projection.projection.shape[0]}.")
    return (matrix - projection.feature_mean) @ projection.projection


def class_alignment_matrices(
    features_by_subject: Mapping[Hashable, Sequence[Sequence[float]] | np.ndarray],
    labels_by_subject: Mapping[Hashable, Sequence | np.ndarray],
    *,
    sample_mode: str = "class_mean",
    n_repetitions_per_class: int | None = None,
    repetition_selection: str = DEFAULT_CLASS_LIMIT_SELECTION,
    repetition_seed: int | str | None = DEFAULT_CLASS_LIMIT_SEED,
) -> ClassAlignment:
    """Build row-aligned class anchors for hyperalignment.

    ``class_repetition`` caps are sampled reproducibly by default instead of
    taking the earliest rows in each class. This avoids run/order confounds in
    repeated-stimulus experiments while keeping row order aligned across subjects
    via common within-class repetition offsets.
    """

    _check_subject_keys(features_by_subject, labels_by_subject)
    sample_mode = _normalize_sample_mode(sample_mode)
    if n_repetitions_per_class is not None and n_repetitions_per_class < 1:
        raise ValueError("n_repetitions_per_class must be positive or None.")

    subject_ids = tuple(features_by_subject.keys())
    features = {sid: _feature_matrix(matrix, name=f"features_by_subject[{sid!r}]") for sid, matrix in features_by_subject.items()}
    labels = {sid: _label_vector(labels_by_subject[sid], expected_length=features[sid].shape[0], name=f"labels_by_subject[{sid!r}]") for sid in subject_ids}
    classes = _common_classes(labels)
    if sample_mode == "class_mean":
        aligned = {sid: _class_mean_matrix(features[sid], labels[sid], classes) for sid in subject_ids}
        repetitions = None
        normalized_selection = None
        normalized_seed = None
    else:
        repetitions = _common_repetition_count(labels, classes, requested=n_repetitions_per_class)
        normalized_selection = normalize_class_limit_selection(repetition_selection)
        normalized_seed = normalize_class_limit_seed(repetition_seed)
        aligned = {
            sid: _class_repetition_matrix(
                features[sid],
                labels[sid],
                classes,
                repetitions,
                selection=normalized_selection,
                seed=normalized_seed,
            )
            for sid in subject_ids
        }
    return ClassAlignment(
        aligned_by_subject=aligned,
        classes=classes,
        sample_mode=sample_mode,
        n_repetitions_per_class=repetitions,
        repetition_selection=normalized_selection,
        repetition_seed=normalized_seed,
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
) -> tuple[HyperalignmentModel, ClassAlignment]:
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
    )
    return model, alignment


def _initial_projection(centered: np.ndarray, n_components: int) -> np.ndarray:
    _u, _s, vt = np.linalg.svd(centered, full_matrices=False)
    projection = vt[:n_components].T
    if projection.shape[1] < n_components:
        projection = np.pad(projection, ((0, 0), (0, n_components - projection.shape[1])))
    return projection


def _orthogonal_procrustes_projection(centered: np.ndarray, template: np.ndarray) -> np.ndarray:
    cross_cov = centered.T @ template
    u, _s, vt = np.linalg.svd(cross_cov, full_matrices=False)
    return u @ vt


def _normalize_template(template: np.ndarray) -> np.ndarray:
    template = template - np.mean(template, axis=0, keepdims=True)
    scale = np.std(template, axis=0, ddof=1)
    scale = np.where(scale < 1e-12, 1.0, scale)
    return template / scale[None, :]


def _average_projection(projections: Mapping[Hashable, SubjectHyperalignmentProjection]) -> tuple[np.ndarray | None, np.ndarray | None]:
    feature_dims = {projection.projection.shape[0] for projection in projections.values()}
    if len(feature_dims) != 1:
        return None, None
    mean = np.mean(np.stack([projection.feature_mean for projection in projections.values()], axis=0), axis=0)
    matrix = np.mean(np.stack([projection.projection for projection in projections.values()], axis=0), axis=0)
    return mean, matrix


def _class_mean_matrix(features: np.ndarray, labels: np.ndarray, classes: np.ndarray) -> np.ndarray:
    return np.vstack([np.mean(features[labels == class_label], axis=0) for class_label in classes])


def _class_repetition_matrix(
    features: np.ndarray,
    labels: np.ndarray,
    classes: np.ndarray,
    repetitions: int,
    *,
    selection: str = DEFAULT_CLASS_LIMIT_SELECTION,
    seed: int | str | None = DEFAULT_CLASS_LIMIT_SEED,
) -> np.ndarray:
    rows = []
    for class_position, class_label in enumerate(classes):
        class_features = features[labels == class_label]
        if class_features.shape[0] < repetitions:
            raise ValueError(f"Class {class_label!r} has only {class_features.shape[0]} repetitions, need {repetitions}.")
        selected = select_class_limited_indices(
            np.zeros(class_features.shape[0], dtype=int),
            repetitions,
            selection=selection,
            seed=seed,
            seed_context=class_position,
        )
        rows.extend(class_features[selected])
    return np.vstack(rows)


def _common_classes(labels_by_subject: Mapping[Hashable, np.ndarray]) -> np.ndarray:
    subject_ids = tuple(labels_by_subject.keys())
    first = np.unique(labels_by_subject[subject_ids[0]])
    for subject_id in subject_ids[1:]:
        classes = np.unique(labels_by_subject[subject_id])
        if not np.array_equal(first, classes):
            raise ValueError(f"Subject {subject_id!r} has classes {classes.tolist()}, expected {first.tolist()}.")
    return first


def _common_repetition_count(labels_by_subject: Mapping[Hashable, np.ndarray], classes: np.ndarray, *, requested: int | None) -> int:
    counts = [int(np.sum(labels == class_label)) for labels in labels_by_subject.values() for class_label in classes]
    available = min(counts)
    if available < 1:
        raise ValueError("Every subject must have at least one sample for every class.")
    if requested is not None and requested > available:
        raise ValueError(f"Requested {requested} repetitions per class, but only {available} are common to all subjects.")
    return available if requested is None else int(requested)


def _check_common_alignment_rows(matrices: Mapping[Hashable, np.ndarray]) -> int:
    row_counts = {subject_id: matrix.shape[0] for subject_id, matrix in matrices.items()}
    unique = set(row_counts.values())
    if len(unique) != 1:
        raise ValueError(f"All subject alignment matrices must have the same row count, got {row_counts}.")
    return int(next(iter(unique)))


def _check_subject_keys(features_by_subject, labels_by_subject) -> None:
    if set(features_by_subject) != set(labels_by_subject):
        raise ValueError("features_by_subject and labels_by_subject must have identical subject keys.")
    if len(features_by_subject) < 2:
        raise ValueError("At least two subjects are required.")


def _requested_component_count(n_components: int | float) -> int:
    if n_components == float("inf"):
        return np.iinfo(np.int32).max
    requested = int(n_components)
    if requested < 1:
        raise ValueError("n_components must be positive or infinity.")
    return requested


def _normalize_sample_mode(sample_mode: str) -> str:
    normalized = str(sample_mode).strip().lower().replace("-", "_")
    if normalized not in CLASS_ALIGNMENT_SAMPLE_MODES:
        raise ValueError(f"Unknown class-alignment sample mode: {sample_mode}. Available modes: {', '.join(CLASS_ALIGNMENT_SAMPLE_MODES)}.")
    return normalized


def _feature_matrix(features: Sequence[Sequence[float]] | np.ndarray, *, name: str) -> np.ndarray:
    matrix = np.asarray(features, dtype=float)
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional feature matrix.")
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError(f"{name} must have at least one row and one column.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} contains non-finite values.")
    return matrix


def _label_vector(labels: Sequence | np.ndarray, *, expected_length: int, name: str) -> np.ndarray:
    vector = np.asarray(labels).ravel()
    if len(vector) != expected_length:
        raise ValueError(f"{name} length must match feature rows: {len(vector)} != {expected_length}.")
    return vector
