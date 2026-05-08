"""Probabilistic tracing of neural representations over time."""

from __future__ import annotations

import sys as _sys

__all__ = ["__version__"]
__version__ = "0.1.0"

from reptrace import _event_detection_extensions
from reptrace import _stimulus_detection_public as stimulus_detection

_event_detection_extensions.install()
_sys.modules.setdefault("reptrace.stimulus_detection", stimulus_detection)
