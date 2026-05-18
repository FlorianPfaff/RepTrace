"""Sampling helpers for balanced decoding experiments."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable

import numpy as np

CLASS_LIMIT_SELECTION_MODES = ("first", "random")
DEFAULT_CLASS_LIMIT_SELECTION = "random"
DEFAULT_CLASS_LIMIT_SEED = 0


def select_class_limited_indices(
    labels,
    max_per_class,
    *,
    selection: str = DEFAULT_CLASS_LIMIT_SELECTION,
    seed: int | str | None = DEFAULT_CLASS_LIMIT_SEED,
    seed_context: int | Iterable[int] | None = None,
) -> np.ndarray:
    """Return row indices after applying an optional per-class cap.

    Parameters
    ----------
    labels:
        One-dimensional class labels, or any array-like object that can be flattened.
    max_per_class:
        Maximum number of rows to keep per class. ``None`` keeps every row.
    selection:
        ``"random"`` samples without replacement within each class, then returns
        the selected indices in ascending input order. This is the default to
        avoid order-dependent caps in result-producing benchmarks. ``"first"``
        is available for legacy/debug use and keeps the earliest rows in input
        order.
    seed:
        Base random seed for ``selection="random"``. ``None`` requests a fresh,
        non-deterministic generator.
    seed_context:
        Optional integer or integer iterable mixed into the deterministic seed.
        This is useful for independent participant-, fold-, or split-specific caps
        while keeping each split reproducible.
    """

    labels = np.asarray(labels).ravel()
    if max_per_class is None:
        return np.arange(labels.shape[0], dtype=int)

    max_per_class = int(max_per_class)
    if max_per_class <= 0:
        raise ValueError("max_per_class must be positive or None.")
    selection = normalize_class_limit_selection(selection)

    if selection == "first":
        selected = []
        counts = Counter()
        for index, label in enumerate(labels):
            if counts[label] < max_per_class:
                selected.append(index)
                counts[label] += 1
        return np.asarray(selected, dtype=int)

    rng = _class_limit_rng(seed, seed_context)
    selected = []
    for label in np.unique(labels):
        class_indices = np.flatnonzero(labels == label)
        if class_indices.size > max_per_class:
            class_indices = rng.choice(class_indices, size=max_per_class, replace=False)
        selected.extend(int(index) for index in class_indices)
    return np.asarray(sorted(selected), dtype=int)


def normalize_class_limit_selection(value: str) -> str:
    """Normalize and validate per-class cap selection mode names."""

    normalized = str(value).strip().lower().replace("-", "_")
    if normalized not in CLASS_LIMIT_SELECTION_MODES:
        raise ValueError(f"selection must be one of {CLASS_LIMIT_SELECTION_MODES}.")
    return normalized


def normalize_class_limit_seed(value: int | str | None) -> int | None:
    """Normalize a deterministic class-limit seed value."""

    if value is None or value == "":
        return None
    seed = int(value)
    if seed < 0:
        raise ValueError("seed must be non-negative or None.")
    return seed


def _class_limit_rng(seed: int | str | None, seed_context: int | Iterable[int] | None):
    seed = normalize_class_limit_seed(seed)
    if seed is None:
        return np.random.default_rng()
    entropy = [seed, *_seed_context_values(seed_context)]
    return np.random.default_rng(np.random.SeedSequence(entropy))


def _seed_context_values(seed_context: int | Iterable[int] | None) -> list[int]:
    if seed_context is None:
        return []
    if isinstance(seed_context, (str, bytes)) or np.isscalar(seed_context):
        values = [int(seed_context)]
    else:
        try:
            values = [int(value) for value in seed_context]
        except TypeError:
            values = [int(seed_context)]
    if any(value < 0 for value in values):
        raise ValueError("seed_context values must be non-negative.")
    return values
