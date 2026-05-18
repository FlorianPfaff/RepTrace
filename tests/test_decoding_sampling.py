import numpy as np
import pytest

from reptrace.decoding import (
    DEFAULT_CLASS_LIMIT_SELECTION,
    normalize_class_limit_seed,
    normalize_class_limit_selection,
    select_class_limited_indices,
)


def test_select_class_limited_indices_keeps_all_when_uncapped():
    labels = np.array([1, 2, 1, 2])

    selected = select_class_limited_indices(labels, None)

    assert selected.tolist() == [0, 1, 2, 3]


def test_select_class_limited_indices_defaults_to_seeded_random_selection():
    labels = np.array([1, 2, 1, 2, 1, 2])

    selected = select_class_limited_indices(labels, 2)
    repeated = select_class_limited_indices(labels, 2)

    assert DEFAULT_CLASS_LIMIT_SELECTION == "random"
    assert selected.tolist() == [1, 2, 4, 5]
    assert repeated.tolist() == selected.tolist()


def test_select_class_limited_indices_first_preserves_input_order():
    labels = np.array([1, 2, 1, 2, 1, 2])

    selected = select_class_limited_indices(labels, 2, selection="first")

    assert selected.tolist() == [0, 1, 2, 3]


def test_select_class_limited_indices_random_is_seeded_by_context():
    labels = np.array([1, 2, 1, 2, 1, 2])

    selected = select_class_limited_indices(labels, 2, selection="random", seed=0, seed_context=1)
    repeated = select_class_limited_indices(labels, 2, selection="random", seed=0, seed_context=1)
    other_context = select_class_limited_indices(labels, 2, selection="random", seed=0, seed_context=2)

    assert selected.tolist() == [1, 2, 3, 4]
    assert repeated.tolist() == selected.tolist()
    assert other_context.tolist() != selected.tolist()


def test_select_class_limited_indices_validates_inputs():
    with pytest.raises(ValueError, match="max_per_class"):
        select_class_limited_indices([1, 2], 0)
    with pytest.raises(ValueError, match="selection"):
        select_class_limited_indices([1, 2], 1, selection="middle")
    with pytest.raises(ValueError, match="seed"):
        normalize_class_limit_seed(-1)


def test_class_limit_normalizers_accept_aliases_and_empty_seed():
    assert normalize_class_limit_selection("random") == "random"
    assert normalize_class_limit_selection("first") == "first"
    assert normalize_class_limit_seed(7) == 7
    assert normalize_class_limit_seed("") is None
