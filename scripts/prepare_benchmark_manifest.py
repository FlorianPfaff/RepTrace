"""Prepare a benchmark manifest for a runner-local data directory."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def prepare_manifest(input_csv: Path, output_csv: Path, data_root: Path) -> None:
    """Rewrite epoch and event paths in a manifest to point at ``data_root``."""
    data_root = data_root.expanduser().resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with input_csv.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames is None:
            raise ValueError(f"Manifest has no header: {input_csv}")
        rows = list(reader)

    for row in rows:
        row["epochs"] = str(data_root / Path(row["epochs"]).name)
        row["events_csv"] = str(data_root / Path(row["events_csv"]).name)

    with output_csv.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=reader.fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("output_csv", type=Path)
    parser.add_argument("--data-root", type=Path, required=True)
    args = parser.parse_args()

    prepare_manifest(args.input_csv, args.output_csv, args.data_root)
    print(f"Wrote runner manifest: {args.output_csv}")


if __name__ == "__main__":
    main()
