"""Procrustes hyperalignment utilities for cross-subject feature decoding.

This module implements a low-rank, Procrustes-style variant of functional
hyperalignment that is practical for high-dimensional M/EEG feature windows. The
caller provides row-aligned matrices, for example one row per stimulus class or
one row per stimulus class and repetition index. The algorithm learns a common
low-dimensional template and a subject-specific semi-orthogonal projection that
maps each subject's feature vectors into that common space.

The implementation deliberately does not assume a particular dataset. PyMEGDec
uses the class-based helpers below for 16-way image-identity MEG decoding, but
other datasets can call :func:`fit_hyperalignment` directly with any aligned rows.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

CLASS_ALIGNMENT_SAMPLE_MODES = ("class_mean", "class_repetition")
TEMPLATE_INITIALIZATION_MODES = ("pca", "mean")


@dataclass(frozen=True)
class SubjectHyperalignmentProjection:
    """Subject-specific Procrustes map into a common hyperalignment space."""

    subject_id: Hashable
    feature_mean: np.ndarray
    projection: np.ndarray
    n_alignment_rows: int


@dataclass(frozen=True)
class HyperalignmentModel:
    """Fitted low-rank Procrustes hyperalignment model.

    ``projection`` matrices have shape ``n_features x n_components``. For fitted
    training subjects, use :meth:`transform`. For a target subject with labeled
    calibration anchors, estimate a target projection with
    :func:`fit_projection_to_hyperalignment`. For calibration-free transfer, use
    :meth:`transform_group`; that group-average projection is a useful baseline
    but is not equivalent to true target-subject hyperalignment.
    """

    subject_ids: tuple[Hashable, ...]
    n_components: int
    n_iterations: int
    initialization: str
    projections: Mapping[Hashable, SubjectHyperalignmentProjection]
    template: np.ndarray
    template_history: tuple[float, ...]
    group_feature_mean: np.ndarray | None
    group_projection: np.ndarray | None
    center: bool
    normalize_template: bool

    def transform(self, subject_id: Hashable, features: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
        """Transform rows from a fitted subject into the common space."""

        try:
            subject_projection = self.projections[subject_id]
        except KeyError as exc:
            fitted = ", ".join(str(value) for value in self.subject_ids)
            raise KeyError(f"Unknown hyperalignment subject {subject_id!r}. Fitted subjects: {fitted}.") from exc
        return transform_with_projection(subject_projection, features)

    def transform_group(
        self,
        features: Sequence[Sequence[float]] | np.ndarray,
        *,
        feature_mean: Sequence[float] | np.ndarray | None = None,
    ) -> np.ndarray:
        """Transform rows with the across-training-subject average projection.

        This is intended for calibration-free transfer to a subject not included
        in :func:`fit_hyperalignment`. Passing ``feature_mean`` allows an
        unsupervised centering estimate from the target subject; when omitted the
        average training-subject alignment mean is used.
        """

        if self.group_projection is None or self.group_feature_mean is None:
            raise ValueError("A group projection is unavailable because fitted subjects have incompatible feature dimensions.")
        matrix = _feature_matrix(features, name="features")
        if matrix.shape[1] != self.group_projection.shape[0]:
            raise ValueError(
                "features column count does not match the group projection: "
                f"{matrix.shape[1]} != {self.group_projection.shape[0]}."
            )
        if feature_mean is None:
            mean = self.group_feature_mean
        else:
            mean = np.asarray(feature_mean, dtype=float).ravel()
            if mean.shape[0] != matrix.shape[1]:
                raise ValueError(f"feature_mean length must match features columns: {mean.shape[0]} != {matrix.shape[1]}.")
        return (matrix - mean) @ self.group_projection


@dataclass(frozen=True)
class ClassAlignment:
    """Aligned rows built from class labels for a collection of subjects."""

    aligned_by_subject: Mapping[Hashable, np.ndarray]
    classes: np.ndarray
    sample_mode: str
    n_repetitions_per_class: int | None


# pylint: disable-next=too-many-locals
def fit_hyperalignment(
    aligned_by_subject: Mapping[Hashable, Sequence[Sequence[float]] | np.ndarray],
    *,
    n_components: int | float = 64,
    n_iterations: int = 10,
    initialization: str = "pca",
    center: bool = True,
    normalize_template: bool = True,
    orthogonalize_group_projection: bool = True,
) -> HyperalignmentModel:
    """Fit a low-rank Procrustes hyperalignment model.

    Parameters
    ----------
    aligned_by_subject:
        Mapping from subject id to a matrix with shape
        ``n_aligned_samples x n_features``. All matrices must have the same
        number of rows and the same row order.
    n_components:
        Number of dimensions in the common model space. The actual count is
        capped by the number of aligned rows and the smallest subject feature
        dimension.
    n_iterations:
        Number of iterative template-refinement passes. ``0`` fits projections
        to the initialization template only.
    initialization:
        ``"pca"`` initializes the template from the row-space PCA of all
        subjects without materializing a huge horizontal concatenation. ``"mean"``
        initializes from the feature-space average and therefore requires all
        subjects to have the same number of features.
    center:
        Whether to subtract each subject's alignment-row mean before fitting and
        transforming.
    normalize_template:
        Whether to center and unit-RMS scale template columns after each update.
    orthogonalize_group_projection:
        Whether the average same-layout group projection should be projected back
        to the nearest semi-orthogonal matrix before calibration-free transfer.
    """

    if len(aligned_by_subject) < 2:
        raise ValueError("Hyperalignment requires at least two subjects.")
    if n_iterations < 0:
        raise ValueError("n_iterations must be non-negative.")
    initialization = str(initialization).strip().lower().replace("-", "_")
    if initialization not in TEMPLATE_INITIALIZATION_MODES:
        raise ValueError(f"Unsupported initialization: {initialization}. Supported modes: {', '.join(TEMPLATE_INITIALIZATION_MODES)}.")

    subject_ids = tuple(aligned_by_subject.keys())
    matrices = {subject_id: _feature_matrix(matrix, name=f"aligned_by_subject[{subject_id!r}]") for subject_id, matrix in aligned_by_subject.items()}
    n_rows = _check_common_alignment_rows(matrices)
    if n_rows < 2:
        raise ValueError("Hyperalignment requires at least two aligned rows per subject.")
    n_features_min = min(matrix.shape[1] for matrix in matrices.values())
    actual_components = min(_requested_component_count(n_components), n_rows, n_features_min)
    if actual_components < 1:
        raise ValueError("At least one hyperalignment component is required.")

    means = {subject_id: _feature_mean(matrix, center=center) for subject_id, matrix in matrices.items()}
    centered = {subject_id: matrix - means[subject_id] for subject_id, matrix in matrices.items()}
    template = _initial_template(centered, n_components=actual_components, initialization=initialization)
    template = _normalize_template(template, enabled=normalize_template)

    history: list[float] = []
    projections: dict[Hashable, SubjectHyperalignmentProjection] = {}
    for _ in range(int(n_iterations)):
        projections = {
            subject_id: _fit_subject_projection(centered[subject_id], template, subject_id=subject_id, feature_mean=means[subject_id])
            for subject_id in subject_ids
        }
        transformed = np.stack([centered[subject_id] @ projections[subject_id].projection for subject_id in subject_ids], axis=0)
        updated_template = np.mean(transformed, axis=0)
        updated_template = _normalize_template(updated_template, enabled=normalize_template)
        history.append(float(np.linalg.norm(updated_template - template) / max(np.linalg.norm(template), 1e-12)))
        template = updated_template

    projections = {
        subject_id: _fit_subject_projection(centered[subject_id], template, subject_id=subject_id, feature_mean=means[subject_id])
        for subject_id in subject_ids
    }
    transformed = np.stack([centered[subject_id] @ projections[subject_id].projection for subject_id in subject_ids], axis=0)
    component_scores = np.mean(transformed, axis=0)
    template = _normalize_template(component_scores, enabled=normalize_template)
    group_feature_mean, group_projection = _average_projection(projections, orthogonalize=orthogonalize_group_projection)
    return HyperalignmentModel(
        subject_ids=subject_ids,
        n_components=actual_components,
        n_iterations=int(n_iterations),
        initialization=initialization,
        projections=projections,
        template=template,
        template_history=tuple(history),
        group_feature_mean=group_feature_mean,
        group_projection=group_projection,
        center=bool(center),
        normalize_template=bool(normalize_template),
    )


def fit_class_hyperalignment(
    features_by_subject: Mapping[Hashable, Sequence[Sequence[float]] | np.ndarray],
    labels_by_subject: Mapping[Hashable, Sequence | np.ndarray],
    *,
    sample_mode: str = "class_mean",
    classes: Sequence | np.ndarray | None = None,
    max_repetitions_per_class: int | None = None,
    n_components: int | float = 64,
    n_iterations: int = 10,
    initialization: str = "pca",
    center: bool = True,
    normalize_template: bool = True,
) -> tuple[HyperalignmentModel, ClassAlignment]:
    """Fit hyperalignment from class-labeled feature matrices."""

    class_alignment = class_alignment_matrices(
        features_by_subject,
        labels_by_subject,
        sample_mode=sample_mode,
        classes=classes,
        max_repetitions_per_class=max_repetitions_per_class,
    )
    model = fit_hyperalignment(
        class_alignment.aligned_by_subject,
        n_components=n_components,
        n_iterations=n_iterations,
        initialization=initialization,
        center=center,
        normalize_template=normalize_template,
    )
    return model, class_alignment


def fit_projection_to_hyperalignment(
    model: HyperalignmentModel,
    aligned_features: Sequence[Sequence[float]] | np.ndarray,
    *,
    subject_id: Hashable = "target",
) -> SubjectHyperalignmentProjection:
    """Fit one subject's projection to an existing hyperalignment template.

    This is the target-subject calibration path: the common template stays fixed,
    and the new subject receives a Procrustes projection estimated from labeled or
    otherwise row-aligned calibration samples.
    """

    matrix = _feature_matrix(aligned_features, name="aligned_features")
    if matrix.shape[0] != model.template.shape[0]:
        raise ValueError(f"aligned_features rows must match model.template rows: {matrix.shape[0]} != {model.template.shape[0]}.")
    if matrix.shape[1] < model.n_components:
        raise ValueError(f"aligned_features must have at least {model.n_components} columns to fit a semi-orthogonal projection.")
    mean = _feature_mean(matrix, center=model.center)
    centered = matrix - mean
    return _fit_subject_projection(centered, model.template, subject_id=subject_id, feature_mean=mean)


def transform_with_projection(projection: SubjectHyperalignmentProjection, features: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
    """Apply a fitted subject hyperalignment projection."""

    matrix = _feature_matrix(features, name="features")
    if matrix.shape[1] != projection.projection.shape[0]:
        raise ValueError(
            "features column count does not match the projection: "
            f"{matrix.shape[1]} != {projection.projection.shape[0]}."
        )
    return (matrix - projection.feature_mean) @ projection.projection


def class_alignment_matrices(
    features_by_subject: Mapping[Hashable, Sequence[Sequence[float]] | np.ndarray],
    labels_by_subject: Mapping[Hashable, Sequence | np.ndarray],
    *,
    sample_mode: str = "class_mean",
    classes: Sequence | np.ndarray | None = None,
    max_repetitions_per_class: int | None = None,
) -> ClassAlignment:
    """Build row-aligned subject matrices from class labels.

    ``class_mean`` yields one aligned row per class and is robust with few target
    calibration examples. ``class_repetition`` yields one row per class and
    repetition index, preserving more trial-level structure when repetition order
    is comparable across subjects.
    """

    if not features_by_subject:
        raise ValueError("At least one subject is required.")
    if set(features_by_subject) != set(labels_by_subject):
        missing_labels = sorted(set(features_by_subject) - set(labels_by_subject), key=str)
        missing_features = sorted(set(labels_by_subject) - set(features_by_subject), key=str)
        raise ValueError(f"features_by_subject and labels_by_subject must have the same keys; missing_labels={missing_labels}, missing_features={missing_features}.")
    sample_mode = str(sample_mode).strip().lower().replace("-", "_")
    if sample_mode not in CLASS_ALIGNMENT_SAMPLE_MODES:
        raise ValueError(f"Unsupported class alignment sample mode: {sample_mode}. Supported modes: {', '.join(CLASS_ALIGNMENT_SAMPLE_MODES)}.")
    if max_repetitions_per_class is not None and max_repetitions_per_class < 1:
        raise ValueError("max_repetitions_per_class must be positive when provided.")

    matrices = {subject_id: _feature_matrix(features, name=f"features_by_subject[{subject_id!r}]") for subject_id, features in features_by_subject.items()}
    labels = {subject_id: _label_vector(labels_by_subject[subject_id], expected_length=matrices[subject_id].shape[0], name=f"labels_by_subject[{subject_id!r}]") for subject_id in matrices}
    class_values = _alignment_classes(labels, classes)

    if sample_mode == "class_mean":
        aligned = {subject_id: _class_mean_rows(matrices[subject_id], labels[subject_id], class_values) for subject_id in matrices}
        return ClassAlignment(aligned_by_subject=aligned, classes=class_values, sample_mode=sample_mode, n_repetitions_per_class=None)

    n_repetitions = _common_repetition_count(labels, class_values, max_repetitions_per_class=max_repetitions_per_class)
    aligned = {subject_id: _class_repetition_rows(matrices[subject_id], labels[subject_id], class_values, n_repetitions) for subject_id in matrices}
    return ClassAlignment(aligned_by_subject=aligned, classes=class_values, sample_mode=sample_mode, n_repetitions_per_class=n_repetitions)


def _fit_subject_projection(centered_features: np.ndarray, template: np.ndarray, *, subject_id: Hashable, feature_mean: np.ndarray) -> SubjectHyperalignmentProjection:
    cross_covariance = centered_features.T @ template
    left, _singular_values, right_t = np.linalg.svd(cross_covariance, full_matrices=False)
    projection = left @ right_t
    return SubjectHyperalignmentProjection(
        subject_id=subject_id,
        feature_mean=np.asarray(feature_mean, dtype=float),
        projection=projection,
        n_alignment_rows=int(centered_features.shape[0]),
    )


def _initial_template(centered: Mapping[Hashable, np.ndarray], *, n_components: int, initialization: str) -> np.ndarray:
    if initialization == "mean":
        feature_counts = {matrix.shape[1] for matrix in centered.values()}
        if len(feature_counts) != 1:
            raise ValueError("mean initialization requires all subjects to have the same feature count.")
        mean_matrix = np.mean(np.stack(list(centered.values()), axis=0), axis=0)
        return _row_pca_scores(mean_matrix, n_components=n_components)

    gram = None
    for matrix in centered.values():
        subject_gram = matrix @ matrix.T
        gram = subject_gram if gram is None else gram + subject_gram
    if gram is None:
        raise ValueError("At least one subject is required.")
    gram = gram / max(len(centered), 1)
    gram = 0.5 * (gram + gram.T)
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order[:n_components]], 0.0)
    eigenvectors = eigenvectors[:, order[:n_components]]
    return eigenvectors * np.sqrt(eigenvalues)[None, :]


def _row_pca_scores(matrix: np.ndarray, *, n_components: int) -> np.ndarray:
    left, singular_values, _right_t = np.linalg.svd(matrix, full_matrices=False)
    actual = min(n_components, left.shape[1])
    return left[:, :actual] * singular_values[:actual][None, :]


def _normalize_template(template: np.ndarray, *, enabled: bool) -> np.ndarray:
    template = np.asarray(template, dtype=float)
    template = template - np.mean(template, axis=0, keepdims=True)
    if not enabled:
        return template
    scale = np.sqrt(np.mean(template**2, axis=0, keepdims=True))
    scale = np.where(scale < 1e-12, 1.0, scale)
    return template / scale


def _average_projection(projections: Mapping[Hashable, SubjectHyperalignmentProjection], *, orthogonalize: bool) -> tuple[np.ndarray | None, np.ndarray | None]:
    if not projections:
        return None, None
    feature_counts = {projection.projection.shape[0] for projection in projections.values()}
    component_counts = {projection.projection.shape[1] for projection in projections.values()}
    if len(feature_counts) != 1 or len(component_counts) != 1:
        return None, None
    means = np.stack([projection.feature_mean for projection in projections.values()], axis=0)
    matrices = np.stack([projection.projection for projection in projections.values()], axis=0)
    group_projection = np.mean(matrices, axis=0)
    if orthogonalize:
        left, _singular_values, right_t = np.linalg.svd(group_projection, full_matrices=False)
        group_projection = left @ right_t
    return np.mean(means, axis=0), group_projection


def _alignment_classes(labels_by_subject: Mapping[Hashable, np.ndarray], classes: Sequence | np.ndarray | None) -> np.ndarray:
    if classes is None:
        first_subject = next(iter(labels_by_subject))
        class_values = np.asarray(sorted(np.unique(labels_by_subject[first_subject])), dtype=labels_by_subject[first_subject].dtype)
    else:
        class_values = np.asarray(classes).ravel()
    if class_values.size == 0:
        raise ValueError("At least one class is required for class alignment.")
    for subject_id, labels in labels_by_subject.items():
        missing = sorted(set(class_values.tolist()) - set(np.unique(labels).tolist()))
        if missing:
            raise ValueError(f"Subject {subject_id!r} is missing classes required for alignment: {missing}.")
    return class_values


def _class_mean_rows(features: np.ndarray, labels: np.ndarray, classes: np.ndarray) -> np.ndarray:
    return np.vstack([np.mean(features[labels == class_label], axis=0) for class_label in classes])


def _common_repetition_count(labels_by_subject: Mapping[Hashable, np.ndarray], classes: np.ndarray, *, max_repetitions_per_class: int | None) -> int:
    counts = []
    for labels in labels_by_subject.values():
        for class_label in classes:
            counts.append(int(np.sum(labels == class_label)))
    n_repetitions = min(counts) if counts else 0
    if max_repetitions_per_class is not None:
        n_repetitions = min(n_repetitions, int(max_repetitions_per_class))
    if n_repetitions < 1:
        raise ValueError("Every subject and class must have at least one repetition for class_repetition alignment.")
    return int(n_repetitions)


def _class_repetition_rows(features: np.ndarray, labels: np.ndarray, classes: np.ndarray, n_repetitions: int) -> np.ndarray:
    rows = []
    for class_label in classes:
        class_rows = features[labels == class_label]
        if class_rows.shape[0] < n_repetitions:
            raise ValueError(f"Class {class_label!r} has only {class_rows.shape[0]} rows, fewer than {n_repetitions} repetitions.")
        rows.extend(class_rows[:n_repetitions])
    return np.vstack(rows)


def _feature_mean(matrix: np.ndarray, *, center: bool) -> np.ndarray:
    if not center:
        return np.zeros(matrix.shape[1], dtype=float)
    return np.mean(matrix, axis=0)


def _requested_component_count(n_components: int | float) -> int:
    if n_components == float("inf"):
        return 10**12
    requested = int(n_components)
    if requested < 1:
        raise ValueError("n_components must be positive or inf.")
    return requested


def _check_common_alignment_rows(matrices: Mapping[Hashable, np.ndarray]) -> int:
    row_counts = {matrix.shape[0] for matrix in matrices.values()}
    if len(row_counts) != 1:
        raise ValueError(f"All subjects must have the same number of aligned rows; got {sorted(row_counts)}.")
    return int(next(iter(row_counts)))


def _feature_matrix(features: Sequence[Sequence[float]] | np.ndarray, *, name: str) -> np.ndarray:
    matrix = np.asarray(features, dtype=float)
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional feature matrix.")
    if matrix.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one row.")
    if matrix.shape[1] == 0:
        raise ValueError(f"{name} must contain at least one column.")
    return matrix


def _label_vector(labels: Sequence | np.ndarray, *, expected_length: int, name: str) -> np.ndarray:
    vector = np.asarray(labels).ravel()
    if len(vector) != expected_length:
        raise ValueError(f"{name} length must match feature rows: {len(vector)} != {expected_length}.")
    return vector
