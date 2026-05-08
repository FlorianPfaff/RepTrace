"""Public event-detection API for stream-level stimulus events.

This module provides the stable public surface for detecting zero, one, or many
stimulus events in long probability streams.  It intentionally sits next to the
legacy :mod:`reptrace.onset_detection` API, which keeps its narrower
first-onset-per-sequence semantics.
"""

from __future__ import annotations

from reptrace._stimulus_detection_public import (
    CONFLICT_RESOLUTION_MODES,
    DEFAULT_GROUP_COLUMNS,
    DEFAULT_STREAM_FALLBACKS,
    DEFAULT_THRESHOLD_QUANTILE,
    DEFAULT_THRESHOLD_WINDOW,
    EVENT_COLUMNS,
    SCORE_MODES,
    THRESHOLD_METHODS,
    detect_stimulus_events,
    detect_stimulus_events_from_csvs,
    fit_stimulus_detection_thresholds,
    main,
    match_stimulus_annotations,
    read_stimulus_probability_observations,
    summarize_stimulus_events,
)
from reptrace.streaming_stimulus_detection import (
    StimulusDetectionConfig,
    StreamingStimulusDetector,
)

__all__ = [
    "CONFLICT_RESOLUTION_MODES",
    "DEFAULT_GROUP_COLUMNS",
    "DEFAULT_STREAM_FALLBACKS",
    "DEFAULT_THRESHOLD_QUANTILE",
    "DEFAULT_THRESHOLD_WINDOW",
    "EVENT_COLUMNS",
    "SCORE_MODES",
    "THRESHOLD_METHODS",
    "StimulusDetectionConfig",
    "StreamingStimulusDetector",
    "detect_stimulus_events",
    "detect_stimulus_events_from_csvs",
    "fit_stimulus_detection_thresholds",
    "main",
    "match_stimulus_annotations",
    "read_stimulus_probability_observations",
    "summarize_stimulus_events",
]


if __name__ == "__main__":  # pragma: no cover
    main()
