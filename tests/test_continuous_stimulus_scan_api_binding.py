from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path

import reptrace._stimulus_detection_public as public_stimulus_detection
import reptrace.continuous_stimulus_scan as continuous_stimulus_scan
import reptrace.event_detection as event_detection
import reptrace.stimulus_detection as stimulus_detection


def test_continuous_scan_uses_public_stimulus_detection_api() -> None:
    """Guard the continuous scanner against binding to the legacy API."""

    assert continuous_stimulus_scan.CONFLICT_RESOLUTION_MODES == event_detection.CONFLICT_RESOLUTION_MODES
    assert continuous_stimulus_scan.CONFLICT_RESOLUTION_MODES == public_stimulus_detection.CONFLICT_RESOLUTION_MODES
    assert continuous_stimulus_scan.detect_stimulus_events is event_detection.detect_stimulus_events
    assert continuous_stimulus_scan.summarize_stimulus_events is event_detection.summarize_stimulus_events
    assert continuous_stimulus_scan.detect_stimulus_events is public_stimulus_detection.detect_stimulus_events
    assert continuous_stimulus_scan.summarize_stimulus_events is public_stimulus_detection.summarize_stimulus_events

    detect_signature = inspect.signature(continuous_stimulus_scan.detect_stimulus_events)
    summary_signature = inspect.signature(continuous_stimulus_scan.summarize_stimulus_events)
    assert "conflict_resolution" in detect_signature.parameters
    assert "observations" in summary_signature.parameters
    assert "stream_columns" in summary_signature.parameters


def _load_module_from_path(path: Path):
    spec = importlib.util.spec_from_file_location("direct_stimulus_detection", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_stimulus_detection_module_is_a_physical_public_wrapper() -> None:
    """Guard against hiding the legacy file behind a package-level alias."""

    assert Path(stimulus_detection.__file__).name == "stimulus_detection.py"
    assert stimulus_detection.CONFLICT_RESOLUTION_MODES == public_stimulus_detection.CONFLICT_RESOLUTION_MODES
    assert stimulus_detection.detect_stimulus_events is public_stimulus_detection.detect_stimulus_events
    assert stimulus_detection.summarize_stimulus_events is public_stimulus_detection.summarize_stimulus_events

    detect_signature = inspect.signature(stimulus_detection.detect_stimulus_events)
    summary_signature = inspect.signature(stimulus_detection.summarize_stimulus_events)
    assert "conflict_resolution" in detect_signature.parameters
    assert "observations" in summary_signature.parameters
    assert "stream_columns" in summary_signature.parameters


def test_direct_stimulus_detection_file_load_uses_public_api() -> None:
    """A direct path load should not expose the legacy implementation."""

    direct_module = _load_module_from_path(Path(stimulus_detection.__file__))
    assert direct_module.CONFLICT_RESOLUTION_MODES == public_stimulus_detection.CONFLICT_RESOLUTION_MODES
    assert direct_module.detect_stimulus_events is public_stimulus_detection.detect_stimulus_events
    assert "conflict_resolution" in inspect.signature(direct_module.detect_stimulus_events).parameters
