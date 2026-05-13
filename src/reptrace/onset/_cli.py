from __future__ import annotations

import argparse
import glob
from pathlib import Path

from reptrace.onset._common import (
    DEFAULT_DETECTION_WINDOW,
    DEFAULT_THRESHOLD_QUANTILE,
    DEFAULT_THRESHOLD_WINDOW,
    THRESHOLD_METHODS,
)
from reptrace.onset._io import detect_onsets_from_csvs


def _expand_paths(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if matches:
            paths.extend(Path(match) for match in matches)
        else:
            paths.append(Path(pattern))
    return paths

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detect representation onsets from RepTrace probability observation CSVs."
    )
    parser.add_argument(
        "observation_csv",
        nargs="+",
        help="Observation CSVs or glob patterns emitted by RepTrace/PyMEGDec adapters.",
    )
    parser.add_argument("--out-events", type=Path, required=True)
    parser.add_argument("--out-summary", type=Path, required=True)
    parser.add_argument(
        "--threshold-window",
        nargs=2,
        type=float,
        default=DEFAULT_THRESHOLD_WINDOW,
        metavar=("START", "STOP"),
    )
    parser.add_argument("--threshold-quantile", type=float, default=DEFAULT_THRESHOLD_QUANTILE)
    parser.add_argument("--threshold-method", choices=THRESHOLD_METHODS, default="point")
    parser.add_argument("--score-column", default="confidence")
    parser.add_argument("--detection-start", type=float)
    parser.add_argument("--event-window", nargs=2, type=float, metavar=("START", "STOP"))
    parser.add_argument("--detection-window", nargs=2, type=float, default=DEFAULT_DETECTION_WINDOW, metavar=("START", "STOP"))
    parser.add_argument("--out-thresholded-observations", type=Path)
    parser.add_argument("--out-threshold-summary", type=Path)
    parser.add_argument(
        "--min-consecutive",
        type=int,
        default=1,
        help="Minimum number of adjacent above-threshold windows required for an onset.",
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        help="Minimum duration in seconds for an above-threshold onset run.",
    )
    parser.add_argument(
        "--require-stable-prediction",
        action="store_true",
        help="Require the predicted class to remain stable across the onset run.",
    )
    args = parser.parse_args()

    events, summary = detect_onsets_from_csvs(
        _expand_paths(args.observation_csv),
        threshold_window=tuple(args.threshold_window),
        threshold_quantile=args.threshold_quantile,
        score_column=args.score_column,
        threshold_method=args.threshold_method,
        detection_start=args.detection_start,
        event_window=tuple(args.event_window) if args.event_window is not None else None,
        detection_window=tuple(args.detection_window),
        min_consecutive=args.min_consecutive,
        min_duration=args.min_duration,
        require_stable_prediction=args.require_stable_prediction,
        out_events=args.out_events,
        out_summary=args.out_summary,
        out_thresholded_observations=args.out_thresholded_observations,
        out_threshold_summary=args.out_threshold_summary,
    )
    print(f"Wrote onset events: {args.out_events}")
    print(f"Wrote onset summary: {args.out_summary}")
    print(summary.to_string(index=False))
    if events.empty:
        print("No event rows were generated.")
