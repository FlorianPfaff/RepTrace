"""Probabilistic tracing of neural representations over time."""

from __future__ import annotations

import importlib.util as _importlib_util
from pathlib import Path as _Path
import sys as _sys

__all__ = ["__version__"]
__version__ = "0.1.0"

from reptrace import _continuous_stimulus_scan_extensions, _event_detection_extensions
from reptrace import _stimulus_detection_public as stimulus_detection

_event_detection_extensions.install()

_stimulus_public_path = _Path(__file__).with_name("_stimulus_detection_public.py")
_stimulus_spec = _importlib_util.spec_from_file_location("reptrace.stimulus_detection", _stimulus_public_path)
if _stimulus_spec is not None:
    stimulus_detection.__spec__ = _stimulus_spec
_sys.modules.setdefault("reptrace.stimulus_detection", stimulus_detection)

_continuous_stimulus_scan_extensions.install()
