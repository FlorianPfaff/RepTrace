"""Probabilistic tracing of neural representations over time."""

__all__ = ["__version__"]
__version__ = "0.1.0"

from reptrace._event_detection_extensions import install as _install_event_detection_extensions

_install_event_detection_extensions()
del _install_event_detection_extensions
