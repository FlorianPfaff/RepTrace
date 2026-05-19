from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

DEFAULT_DECODERS = ("logistic", "lda", "linear_svm")
DEFAULT_FEATURE_PREPROCESSORS = ("none", "pca", "pca_whiten")
DEFAULT_PCA_COMPONENTS = ("0.8", "0.9", "0.95", "16", "32", "64")
DEFAULT_TUNING_C_GRID = "0.01,0.1,1,10,100"
_GRID_COLUMNS = (
    "subject",
    "variant",
    "decoder",
    "feature_preprocessor",
    "pca_components",
    "tune_hyperparameters",
    "tuning_cv_splits",
    "tuning_scoring",
    "tuning_c_grid",
)


def _safe_name(value: object) -> str:
    return str(value).strip().lower().replace("-", "_").replace(" ", "_").replace(".", "p").replace("|", "_").replace(",", "_")


def _variant_name(decoder: str, feature_preprocessor: str, pca_components: str | None, *, tune_hyperparameters: bool) -> str:
    parts = [_safe_name(decoder), _safe_name(feature_preprocessor)]
    if feature_preprocessor != "none" and pca_components not in (None, ""):
        parts.append(f"pca{_safe_name(pca_components)}")
    if tune_hyperparameters:
        parts.append("tuned")
    return "_".join(parts)


def _ordered_columns(frame: pd.DataFrame) -> list[str]:
    leading = [column for column in _GRID_COLUMNS if column in frame.columns]
    return leading + [column for column in frame.columns if column not in leading]


def expand_pca_grid_manifest(
    base_manifest: Path,
    out_path: Path,
    *,
    decoders: Sequence[str] = DEFAULT_DECODERS,
    feature_preprocessors: Sequence[str] = DEFAULT_FEATURE_PREPROCESSORS,
    pca_components: Sequence[str] = DEFAULT_PCA_COMPONENTS,
    tune_hyperparameters: bool = False,
    tuning_cv_splits: int = 2,
    tuning_scoring: str = "balanced_accuracy",
    tuning_c_grid: str = DEFAULT_TUNING_C_GRID,
) -> pd.DataFrame:
    """Expand a subject manifest into a fold-local PCA/whitened-PCA evaluation grid.

    The generated manifest keeps the original subject, file, label, grouping, and
    time-window columns, then adds one row per requested decoder and feature
    preprocessing condition. For ``feature_preprocessor=none``, a single baseline
    row is emitted and ``pca_components`` is left empty. For PCA variants, each
    requested component count or explained-variance fraction becomes a separate
    benchmark condition.

    The resulting rows are intended for ``reptrace-benchmark``. PCA itself is
    still fitted by the existing sklearn pipeline inside each outer training
    fold; this helper only makes the comparison grid explicit and reproducible.
    """
    base = pd.read_csv(base_manifest)
    rows: list[dict[str, object]] = []

    for _, base_row in base.iterrows():
        original = base_row.to_dict()
        for decoder in decoders:
            for feature_preprocessor in feature_preprocessors:
                component_values = ("",) if feature_preprocessor == "none" else tuple(map(str, pca_components))
                for component in component_values:
                    row = dict(original)
                    row["decoder"] = decoder
                    row["feature_preprocessor"] = feature_preprocessor
                    row["pca_components"] = component
                    row["tune_hyperparameters"] = str(bool(tune_hyperparameters)).lower()
                    row["tuning_cv_splits"] = int(tuning_cv_splits)
                    row["tuning_scoring"] = tuning_scoring
                    row["tuning_c_grid"] = tuning_c_grid
                    row["variant"] = _variant_name(
                        decoder,
                        feature_preprocessor,
                        None if component == "" else str(component),
                        tune_hyperparameters=tune_hyperparameters,
                    )
                    rows.append(row)

    expanded = pd.DataFrame(rows)
    expanded = expanded[_ordered_columns(expanded)]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    expanded.to_csv(out_path, index=False)
    return expanded


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a RepTrace benchmark manifest for a PCA/whitened-PCA decoder grid.")
    parser.add_argument("base_manifest", type=Path, help="Input manifest with the subject/file/label columns to replicate.")
    parser.add_argument("--out", type=Path, required=True, help="Expanded manifest CSV to write.")
    parser.add_argument("--decoders", nargs="+", default=list(DEFAULT_DECODERS), help="Decoder names to evaluate.")
    parser.add_argument(
        "--feature-preprocessors",
        nargs="+",
        default=list(DEFAULT_FEATURE_PREPROCESSORS),
        help="Feature preprocessors to evaluate; use none, pca, and/or pca_whiten.",
    )
    parser.add_argument(
        "--pca-components",
        nargs="+",
        default=list(DEFAULT_PCA_COMPONENTS),
        help="PCA component counts or explained-variance fractions for PCA rows.",
    )
    parser.add_argument("--tune-hyperparameters", action="store_true", help="Set tune_hyperparameters=true on generated rows.")
    parser.add_argument("--tuning-cv-splits", type=int, default=2)
    parser.add_argument("--tuning-scoring", default="balanced_accuracy")
    parser.add_argument("--tuning-c-grid", default=DEFAULT_TUNING_C_GRID)
    args = parser.parse_args()

    expanded = expand_pca_grid_manifest(
        args.base_manifest,
        args.out,
        decoders=tuple(args.decoders),
        feature_preprocessors=tuple(args.feature_preprocessors),
        pca_components=tuple(args.pca_components),
        tune_hyperparameters=args.tune_hyperparameters,
        tuning_cv_splits=args.tuning_cv_splits,
        tuning_scoring=args.tuning_scoring,
        tuning_c_grid=args.tuning_c_grid,
    )
    print(f"Wrote {len(expanded)} PCA-grid benchmark row(s): {args.out}")


if __name__ == "__main__":
    main()
