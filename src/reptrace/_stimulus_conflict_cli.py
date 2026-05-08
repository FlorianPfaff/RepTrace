"""CLI wrapper for stimulus conflict-resolution support."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from reptrace._event_detection_extensions import CONFLICT_RESOLUTION_MODES


def install() -> None:
    from reptrace import stimulus_detection as stimulus

    def detect_stimulus_events_from_csvs(
        observation_csvs: Sequence[str | Path],
        *,
        annotations_csv: str | Path | None = None,
        thresholds_csv: str | Path | None = None,
        threshold_window: tuple[float, float] = stimulus.DEFAULT_THRESHOLD_WINDOW,
        threshold_quantile: float = stimulus.DEFAULT_THRESHOLD_QUANTILE,
        threshold_method: str = "point",
        score_mode: str = "class_probability",
        target_classes: Sequence[str | int] | None = None,
        group_columns: Sequence[str] | None = None,
        stream_columns: Sequence[str] | None = None,
        detection_window: tuple[float, float] | None = None,
        min_consecutive: int = 1,
        min_duration: float | None = None,
        merge_gap: float | None = None,
        refractory: float | None = None,
        conflict_resolution: str = "none",
        match_tolerance: float = 0.1,
        out_events: Path | None = None,
        out_summary: Path | None = None,
        out_thresholds: Path | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        observations = stimulus.read_stimulus_probability_observations(observation_csvs)
        thresholds = pd.read_csv(thresholds_csv) if thresholds_csv is not None else None
        events = stimulus.detect_stimulus_events(
            observations,
            thresholds=thresholds,
            threshold_window=threshold_window,
            threshold_quantile=threshold_quantile,
            threshold_method=threshold_method,
            score_mode=score_mode,
            target_classes=target_classes,
            group_columns=group_columns,
            stream_columns=stream_columns,
            detection_window=detection_window,
            min_consecutive=min_consecutive,
            min_duration=min_duration,
            merge_gap=merge_gap,
            refractory=refractory,
            conflict_resolution=conflict_resolution,
        )
        if thresholds is None:
            thresholds = stimulus.fit_stimulus_detection_thresholds(
                observations,
                threshold_window=threshold_window,
                threshold_quantile=threshold_quantile,
                threshold_method=threshold_method,
                score_mode=score_mode,
                target_classes=target_classes,
                group_columns=group_columns,
                stream_columns=stream_columns,
                min_consecutive=min_consecutive,
                min_duration=min_duration,
            )
        annotations = pd.read_csv(annotations_csv) if annotations_csv is not None else None
        if annotations is not None:
            events = stimulus.match_stimulus_annotations(
                events,
                annotations,
                stream_columns=stream_columns,
                match_tolerance=match_tolerance,
            )
        summary = stimulus.summarize_stimulus_events(events, annotations=annotations, group_columns=group_columns)
        for path, frame in ((out_events, events), (out_summary, summary), (out_thresholds, thresholds)):
            if path is not None:
                path.parent.mkdir(parents=True, exist_ok=True)
                frame.to_csv(path, index=False)
        return events, summary, thresholds

    def main() -> None:
        parser = argparse.ArgumentParser(description="Detect one or more stimulus events in long probability streams.")
        parser.add_argument("observation_csv", nargs="+", help="Observation CSVs or glob patterns with time and prob_class_* columns.")
        parser.add_argument("--annotations-csv", type=Path)
        parser.add_argument("--thresholds-csv", type=Path)
        parser.add_argument("--threshold-window", nargs=2, type=float, default=stimulus.DEFAULT_THRESHOLD_WINDOW, metavar=("START", "STOP"))
        parser.add_argument("--threshold-quantile", type=float, default=stimulus.DEFAULT_THRESHOLD_QUANTILE)
        parser.add_argument("--threshold-method", choices=stimulus.THRESHOLD_METHODS, default="point")
        parser.add_argument("--score-mode", choices=stimulus.SCORE_MODES, default="class_probability")
        parser.add_argument("--target-class", action="append", dest="target_classes")
        parser.add_argument("--group-column", action="append", dest="group_columns")
        parser.add_argument("--stream-column", action="append", dest="stream_columns")
        parser.add_argument("--detection-window", nargs=2, type=float, metavar=("START", "STOP"))
        parser.add_argument("--min-consecutive", type=int, default=1)
        parser.add_argument("--min-duration", type=float)
        parser.add_argument("--merge-gap", type=float)
        parser.add_argument("--refractory", type=float)
        parser.add_argument("--conflict-resolution", choices=CONFLICT_RESOLUTION_MODES, default="none")
        parser.add_argument("--match-tolerance", type=float, default=0.1)
        parser.add_argument("--out-events", type=Path, required=True)
        parser.add_argument("--out-summary", type=Path, required=True)
        parser.add_argument("--out-thresholds", type=Path)
        args = parser.parse_args()
        events, summary, _thresholds = detect_stimulus_events_from_csvs(
            args.observation_csv,
            annotations_csv=args.annotations_csv,
            thresholds_csv=args.thresholds_csv,
            threshold_window=tuple(args.threshold_window),
            threshold_quantile=args.threshold_quantile,
            threshold_method=args.threshold_method,
            score_mode=args.score_mode,
            target_classes=args.target_classes,
            group_columns=args.group_columns,
            stream_columns=args.stream_columns,
            detection_window=tuple(args.detection_window) if args.detection_window is not None else None,
            min_consecutive=args.min_consecutive,
            min_duration=args.min_duration,
            merge_gap=args.merge_gap,
            refractory=args.refractory,
            conflict_resolution=args.conflict_resolution,
            match_tolerance=args.match_tolerance,
            out_events=args.out_events,
            out_summary=args.out_summary,
            out_thresholds=args.out_thresholds,
        )
        print(f"Wrote stimulus events: {args.out_events}")
        print(f"Wrote stimulus event summary: {args.out_summary}")
        if args.out_thresholds is not None:
            print(f"Wrote stimulus thresholds: {args.out_thresholds}")
        print(summary.to_string(index=False))
        if not events.empty:
            print(events.head().to_string(index=False))

    stimulus.detect_stimulus_events_from_csvs = detect_stimulus_events_from_csvs
    stimulus.main = main
