from __future__ import annotations

from reptrace.onset._common import (
    DEFAULT_DETECTION_WINDOW,
    DEFAULT_THRESHOLD_QUANTILE,
    DEFAULT_THRESHOLD_WINDOW,
    THRESHOLD_METHODS,
)
from reptrace.onset._events import detect_onsets
from reptrace.onset._io import detect_onsets_from_csvs
from reptrace.onset._summaries import summarize_onset_events, summarize_threshold_crossings
from reptrace.onset._thresholds import annotate_threshold_crossings

__all__ = [
    "DEFAULT_DETECTION_WINDOW",
    "DEFAULT_THRESHOLD_QUANTILE",
    "DEFAULT_THRESHOLD_WINDOW",
    "THRESHOLD_METHODS",
    "annotate_threshold_crossings",
    "detect_onsets",
    "detect_onsets_from_csvs",
    "summarize_onset_events",
    "summarize_threshold_crossings",
]
