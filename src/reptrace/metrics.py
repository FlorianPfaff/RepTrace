from __future__ import annotations

import numpy as np


def expected_calibration_error(
    probabilities: np.ndarray,
    labels: np.ndarray,
    *,
    n_bins: int = 10,
) -> float:
    """Compute top-label expected calibration error.

    Parameters
    ----------
    probabilities:
        Array of shape ``(n_samples, n_classes)`` with predicted class probabilities.
    labels:
        Integer class labels of shape ``(n_samples,)``.
    n_bins:
        Number of equally spaced confidence bins.
    """
    if probabilities.ndim != 2:
        raise ValueError("probabilities must have shape (n_samples, n_classes)")
    if labels.ndim != 1:
        raise ValueError("labels must have shape (n_samples,)")
    if probabilities.shape[0] != labels.shape[0]:
        raise ValueError("probabilities and labels must contain the same samples")
    if n_bins < 1:
        raise ValueError("n_bins must be positive")

    predictions = probabilities.argmax(axis=1)
    confidences = probabilities.max(axis=1)
    correct = predictions == labels

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for left, right in zip(edges[:-1], edges[1:]):
        if right == 1.0:
            in_bin = (confidences >= left) & (confidences <= right)
        else:
            in_bin = (confidences >= left) & (confidences < right)
        if not np.any(in_bin):
            continue
        bin_weight = np.mean(in_bin)
        bin_accuracy = np.mean(correct[in_bin])
        bin_confidence = np.mean(confidences[in_bin])
        ece += bin_weight * abs(bin_accuracy - bin_confidence)
    return float(ece)


def brier_score_multiclass(probabilities: np.ndarray, labels: np.ndarray) -> float:
    """Compute multiclass Brier score using one-hot targets."""
    if probabilities.ndim != 2:
        raise ValueError("probabilities must have shape (n_samples, n_classes)")
    if labels.ndim != 1:
        raise ValueError("labels must have shape (n_samples,)")
    if probabilities.shape[0] != labels.shape[0]:
        raise ValueError("probabilities and labels must contain the same samples")

    targets = np.zeros_like(probabilities, dtype=float)
    targets[np.arange(labels.shape[0]), labels] = 1.0
    return float(np.mean(np.sum((probabilities - targets) ** 2, axis=1)))
