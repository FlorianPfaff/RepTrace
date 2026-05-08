"""Probabilistic tracing of neural representations over time."""

__all__ = ["__version__"]
__version__ = "0.1.0"

from reptrace import _event_detection_extensions
from reptrace import _stimulus_conflict_cli

_event_detection_extensions.install()
_stimulus_conflict_cli.install()
