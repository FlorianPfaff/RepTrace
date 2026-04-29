from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


def add_binary_label(
    metadata: pd.DataFrame,
    *,
    source_column: str,
    positive_pattern: str,
    label_column: str,
    negative_pattern: str | None = None,
    positive_label: str = "positive",
    negative_label: str = "negative",
    case_sensitive: bool = False,
) -> pd.DataFrame:
    """Add a binary label column by matching text patterns in an existing column.

    When ``negative_pattern`` is omitted, every non-null source value that does
    not match ``positive_pattern`` receives the negative label. When
    ``negative_pattern`` is provided, unmatched rows receive missing labels and
    can be excluded by the decoder.
    """
    if source_column not in metadata.columns:
        raise ValueError(f"Source column '{source_column}' not found in metadata.")
    if label_column in metadata.columns:
        raise ValueError(f"Label column '{label_column}' already exists.")

    flags = 0 if case_sensitive else re.IGNORECASE
    source = metadata[source_column].astype("string")
    positive = source.str.contains(positive_pattern, flags=flags, regex=True, na=False)
    if negative_pattern is None:
        negative = source.notna() & ~positive
    else:
        negative = source.str.contains(negative_pattern, flags=flags, regex=True, na=False)

    labeled = metadata.copy()
    labeled[label_column] = pd.NA
    labeled.loc[positive, label_column] = positive_label
    labeled.loc[negative, label_column] = negative_label
    return labeled


def prepare_binary_metadata(
    events_csv: Path,
    out_path: Path,
    *,
    source_column: str,
    positive_pattern: str,
    label_column: str,
    negative_pattern: str | None = None,
    positive_label: str = "positive",
    negative_label: str = "negative",
    case_sensitive: bool = False,
) -> pd.DataFrame:
    """Load metadata, add a binary label, and write the result as CSV."""
    metadata = pd.read_csv(events_csv)
    labeled = add_binary_label(
        metadata,
        source_column=source_column,
        positive_pattern=positive_pattern,
        negative_pattern=negative_pattern,
        label_column=label_column,
        positive_label=positive_label,
        negative_label=negative_label,
        case_sensitive=case_sensitive,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    labeled.to_csv(out_path, index=False)
    return labeled


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add a binary decoding label to an events or metadata CSV."
    )
    parser.add_argument("--events-csv", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source-column", required=True)
    parser.add_argument("--positive-pattern", required=True)
    parser.add_argument("--negative-pattern")
    parser.add_argument("--label-column", default="condition")
    parser.add_argument("--positive-label", default="positive")
    parser.add_argument("--negative-label", default="negative")
    parser.add_argument("--case-sensitive", action="store_true")
    args = parser.parse_args()

    labeled = prepare_binary_metadata(
        events_csv=args.events_csv,
        out_path=args.out,
        source_column=args.source_column,
        positive_pattern=args.positive_pattern,
        negative_pattern=args.negative_pattern,
        label_column=args.label_column,
        positive_label=args.positive_label,
        negative_label=args.negative_label,
        case_sensitive=args.case_sensitive,
    )
    counts = labeled[args.label_column].value_counts(dropna=False).to_dict()
    print(f"Wrote {args.out}")
    print(f"Label counts: {counts}")


if __name__ == "__main__":
    main()
