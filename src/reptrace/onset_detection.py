from __future__ import annotations

from reptrace.onset import (
    DEFAULT_DETECTION_WINDOW,
    DEFAULT_THRESHOLD_QUANTILE,
    DEFAULT_THRESHOLD_WINDOW,
    THRESHOLD_METHODS,
    annotate_threshold_crossings,
    detect_onsets,
    detect_onsets_from_csvs,
    summarize_onset_events,
    summarize_threshold_crossings,
)
from reptrace.onset._common import _run_duration
from reptrace.onset._cli import _expand_paths, main
from reptrace.onset._events import _candidate_segments, _detection_runs, _first_detection_run

__all__ = [
    "DEFAULT_DETECTION_WINDOW",
    "DEFAULT_THRESHOLD_QUANTILE",
    "DEFAULT_THRESHOLD_WINDOW",
    "THRESHOLD_METHODS",
    "_candidate_segments",
    "_detection_runs",
    "_expand_paths",
    "_first_detection_run",
    "_run_duration",
    "annotate_threshold_crossings",
    "detect_onsets",
    "detect_onsets_from_csvs",
    "main",
    "summarize_onset_events",
    "summarize_threshold_crossings",
]


if __name__ == "__main__":
    main()
