from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from reptrace.onset_detection import (
    DEFAULT_THRESHOLD_QUANTILE,
    DEFAULT_THRESHOLD_WINDOW,
    THRESHOLD_METHODS,
    _expand_paths,
    annotate_threshold_crossings,
    detect_onsets,
    summarize_onset_events,
)
from reptrace.temporal_model import read_probability_observations


@dataclass(frozen=True)
class OnsetChunk:
    """One time chunk used to validate onset behavior."""

    name: str
    start: float
    stop: float
    expected_response: str = "unknown"


DEFAULT_CHUNKS = (
    OnsetChunk("pre", -0.30, -0.05, "null"),
    OnsetChunk("early", 0.05, 0.20, "early"),
    OnsetChunk("late", 0.20, 0.60, "positive"),
)


def parse_chunk_spec(spec: str) -> OnsetChunk:
    """Parse a chunk specification of the form ``name:start:stop[:expected]``."""

    parts = spec.split(":")
    if len(parts) not in (3, 4):
        raise ValueError("Chunk specs must have the form name:start:stop[:expected].")
    name, raw_start, raw_stop = parts[:3]
    if not name:
        raise ValueError("Chunk name must not be empty.")
    start = float(raw_start)
    stop = float(raw_stop)
    if start > stop:
        raise ValueError("Chunk start must be less than or equal to chunk stop.")
    expected = parts[3] if len(parts) == 4 else "unknown"
    return OnsetChunk(name=name, start=start, stop=stop, expected_response=expected)


def _tag_chunk(frame: pd.DataFrame, chunk: OnsetChunk) -> pd.DataFrame:
    tagged = frame.copy()
    tagged.insert(0, "chunk_expected_response", chunk.expected_response)
    tagged.insert(0, "chunk_stop", chunk.stop)
    tagged.insert(0, "chunk_start", chunk.start)
    tagged.insert(0, "chunk", chunk.name)
    return tagged


# pylint: disable-next=too-many-arguments,too-many-locals
def summarize_onset_chunks(
    observations: pd.DataFrame,
    chunks: list[OnsetChunk] | tuple[OnsetChunk, ...] = DEFAULT_CHUNKS,
    *,
    threshold_window: tuple[float, float] = DEFAULT_THRESHOLD_WINDOW,
    threshold_quantile: float = DEFAULT_THRESHOLD_QUANTILE,
    threshold_method: str = "max_run",
    score_column: str = "confidence",
    min_consecutive: int = 2,
    min_duration: float | None = None,
    require_stable_prediction: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run onset detection separately inside named time chunks.

    The threshold is still estimated from ``threshold_window`` over the full
    observation table. Each chunk only limits the candidate event window. This
    makes pre-stimulus chunks useful as negative controls and post-stimulus
    chunks useful as coarse positive checks.
    """

    if not chunks:
        raise ValueError("At least one onset-validation chunk is required.")
    thresholded = annotate_threshold_crossings(
        observations,
        threshold_window=threshold_window,
        threshold_quantile=threshold_quantile,
        threshold_method=threshold_method,
        score_column=score_column,
        min_consecutive=min_consecutive,
        min_duration=min_duration,
        require_stable_prediction=require_stable_prediction,
    )
    event_frames = []
    summary_frames = []
    for chunk in chunks:
        scan_mask = (
            thresholded["time"].between(threshold_window[0], threshold_window[1])
            | thresholded["time"].between(chunk.start, chunk.stop)
        )
        events = detect_onsets(
            thresholded.loc[scan_mask],
            threshold_window=threshold_window,
            threshold_quantile=threshold_quantile,
            threshold_method=threshold_method,
            score_column=score_column,
            detection_window=(chunk.start, chunk.stop),
            min_consecutive=min_consecutive,
            min_duration=min_duration,
            require_stable_prediction=require_stable_prediction,
        )
        summary = summarize_onset_events(events)
        event_frames.append(_tag_chunk(events, chunk))
        summary_frames.append(_tag_chunk(summary, chunk))
    return pd.concat(event_frames, ignore_index=True), pd.concat(summary_frames, ignore_index=True)


def run_onset_chunk_validation(
    observation_csvs: list[Path],
    *,
    chunks: list[OnsetChunk] | tuple[OnsetChunk, ...] = DEFAULT_CHUNKS,
    threshold_window: tuple[float, float] = DEFAULT_THRESHOLD_WINDOW,
    threshold_quantile: float = DEFAULT_THRESHOLD_QUANTILE,
    threshold_method: str = "max_run",
    score_column: str = "confidence",
    min_consecutive: int = 2,
    min_duration: float | None = None,
    require_stable_prediction: bool = True,
    out_events: Path | None = None,
    out_summary: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read observation CSVs and write optional chunk-validation summaries."""

    observations = read_probability_observations(observation_csvs)
    events, summary = summarize_onset_chunks(
        observations,
        chunks,
        threshold_window=threshold_window,
        threshold_quantile=threshold_quantile,
        threshold_method=threshold_method,
        score_column=score_column,
        min_consecutive=min_consecutive,
        min_duration=min_duration,
        require_stable_prediction=require_stable_prediction,
    )
    if out_events is not None:
        out_events.parent.mkdir(parents=True, exist_ok=True)
        events.to_csv(out_events, index=False)
    if out_summary is not None:
        out_summary.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(out_summary, index=False)
    return events, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate onset detection across named time chunks.")
    parser.add_argument("observation_csv", nargs="+", help="Observation CSVs or glob patterns.")
    parser.add_argument(
        "--chunk",
        action="append",
        help="Chunk specification name:start:stop[:expected]. Repeat for multiple chunks.",
    )
    parser.add_argument(
        "--threshold-window",
        nargs=2,
        type=float,
        default=DEFAULT_THRESHOLD_WINDOW,
        metavar=("START", "STOP"),
    )
    parser.add_argument("--threshold-quantile", type=float, default=DEFAULT_THRESHOLD_QUANTILE)
    parser.add_argument("--threshold-method", choices=THRESHOLD_METHODS, default="max_run")
    parser.add_argument("--score-column", default="confidence")
    parser.add_argument("--min-consecutive", type=int, default=2)
    parser.add_argument("--min-duration", type=float)
    parser.add_argument("--allow-unstable-prediction", action="store_true")
    parser.add_argument("--out-events", type=Path)
    parser.add_argument("--out-summary", type=Path, required=True)
    args = parser.parse_args()

    chunks = tuple(parse_chunk_spec(spec) for spec in args.chunk) if args.chunk else DEFAULT_CHUNKS
    events, summary = run_onset_chunk_validation(
        _expand_paths(args.observation_csv),
        chunks=chunks,
        threshold_window=tuple(args.threshold_window),
        threshold_quantile=args.threshold_quantile,
        threshold_method=args.threshold_method,
        score_column=args.score_column,
        min_consecutive=args.min_consecutive,
        min_duration=args.min_duration,
        require_stable_prediction=not args.allow_unstable_prediction,
        out_events=args.out_events,
        out_summary=args.out_summary,
    )
    if args.out_events is not None:
        print(f"Wrote onset chunk events: {args.out_events}")
    print(f"Wrote onset chunk summary: {args.out_summary}")
    print(summary.to_string(index=False))
    if events.empty:
        print("No chunk-validation event rows were generated.")


if __name__ == "__main__":
    main()
