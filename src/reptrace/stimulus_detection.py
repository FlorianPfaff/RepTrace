"""Public stimulus-event detection API.

The stream-level detector used to live directly in this module.  The legacy
implementation is now private in :mod:`reptrace._stimulus_detection_legacy_impl`,
while this module exposes the extended public API used by the CLI and by
continuous stimulus scanning.  Keeping the wrapper here makes direct imports,
console entry points, and ``python -m reptrace.stimulus_detection`` all bind to
the same API without relying on package-level ``sys.modules`` aliasing.
"""

from __future__ import annotations

if __name__ == "reptrace._stimulus_detection_legacy":
    # Compatibility path for reptrace._stimulus_detection_public while it loads
    # the original implementation under a private module name.
    from reptrace import _stimulus_detection_legacy_impl as _legacy_impl

    globals().update(
        {
            name: getattr(_legacy_impl, name)
            for name in dir(_legacy_impl)
            if not (name.startswith("__") and name.endswith("__"))
        }
    )
else:
    from reptrace._stimulus_detection_public import *  # noqa: F401,F403
    from reptrace._stimulus_detection_public import __all__ as _public_all
    from reptrace._stimulus_detection_public import main
    from reptrace.matched_filter_detection import (  # noqa: F401
        MATCHED_FILTER_SCORE_COLUMN,
        MATCHED_FILTER_SCORE_MODE,
        MATCHED_FILTER_THRESHOLD_METHOD,
        detect_matched_filter_stimulus_events,
        fit_matched_filter_thresholds,
        fit_stimulus_event_templates,
        score_stimulus_event_templates,
    )

    # Backwards-compatible private helpers used by the streaming detector.
    from reptrace._stimulus_detection_public import _event_row, _run_duration  # noqa: F401

    __all__ = [
        *_public_all,
        "MATCHED_FILTER_SCORE_COLUMN",
        "MATCHED_FILTER_SCORE_MODE",
        "MATCHED_FILTER_THRESHOLD_METHOD",
        "detect_matched_filter_stimulus_events",
        "fit_matched_filter_thresholds",
        "fit_stimulus_event_templates",
        "score_stimulus_event_templates",
    ]

    if __name__ == "__main__":  # pragma: no cover
        raise SystemExit(main())