from __future__ import annotations

import inspect

import reptrace._stimulus_detection_public as public_stimulus_detection
import reptrace.continuous_stimulus_scan as continuous_stimulus_scan


def test_continuous_scan_uses_public_stimulus_detection_api() -> None:
    """Guard the continuous scanner against binding to the legacy API."""

    assert continuous_stimulus_scan.CONFLICT_RESOLUTION_MODES == public_stimulus_detection.CONFLICT_RESOLUTION_MODES
    assert continuous_stimulus_scan.detect_stimulus_events is public_stimulus_detection.detect_stimulus_events
    assert continuous_stimulus_scan.summarize_stimulus_events is public_stimulus_detection.summarize_stimulus_events

    detect_signature = inspect.signature(continuous_stimulus_scan.detect_stimulus_events)
    summary_signature = inspect.signature(continuous_stimulus_scan.summarize_stimulus_events)
    assert "conflict_resolution" in detect_signature.parameters
    assert "observations" in summary_signature.parameters
    assert "stream_columns" in summary_signature.parameters
