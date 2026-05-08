from __future__ import annotations

DEFAULT_THRESHOLD_WINDOW = (-0.35, -0.05)
DEFAULT_DETECTION_WINDOW = (0.0, float("inf"))
DEFAULT_THRESHOLD_QUANTILE = 0.95
THRESHOLD_METHODS = ("point", "max_run")
GROUP_COLUMNS = ("subject", "decoder", "emission_mode")
