import pandas as pd
import pytest

from reptrace.metadata import add_binary_label


def test_add_binary_label_marks_positive_and_default_negative():
    metadata = pd.DataFrame({"category": ["face/person", "chair", "car"]})

    labeled = add_binary_label(
        metadata,
        source_column="category",
        positive_pattern="face|person",
        label_column="is_face",
        positive_label="face",
        negative_label="object",
    )

    assert labeled["is_face"].tolist() == ["face", "object", "object"]


def test_add_binary_label_with_negative_pattern_leaves_unmatched_missing():
    metadata = pd.DataFrame({"category": ["face", "chair", "unknown"]})

    labeled = add_binary_label(
        metadata,
        source_column="category",
        positive_pattern="face",
        negative_pattern="chair",
        label_column="condition",
    )

    assert labeled["condition"].tolist()[:2] == ["positive", "negative"]
    assert pd.isna(labeled["condition"].tolist()[2])


def test_add_binary_label_rejects_existing_label_column():
    metadata = pd.DataFrame({"category": ["face"], "condition": ["old"]})

    with pytest.raises(ValueError, match="already exists"):
        add_binary_label(
            metadata,
            source_column="category",
            positive_pattern="face",
            label_column="condition",
        )
