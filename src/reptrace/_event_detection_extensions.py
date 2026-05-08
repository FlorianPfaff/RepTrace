"""Compatibility installer for onset run detection helpers."""

from __future__ import annotations

from reptrace._onset_detection_runs import detection_runs as _enumerate_detection_runs
from reptrace._onset_detection_runs import first_detection_run as _select_first_detection_run


def _install_onset_detection_extensions() -> None:
    """Expose all-run onset helpers on :mod:`reptrace.onset_detection`.

    ``reptrace.onset_detection`` remains the public detector module. This
    installer is intentionally idempotent so importing :mod:`reptrace` repeatedly
    never wraps or replaces an already-native implementation unnecessarily.
    """
    from reptrace import onset_detection as onset

    existing_all_runs = getattr(onset, "_detection_runs", None)
    if existing_all_runs is not None and getattr(existing_all_runs, "__module__", "") == onset.__name__:
        return

    def _detection_runs(
        candidates,
        *,
        threshold,
        min_consecutive,
        min_duration,
        require_stable_prediction,
        merge_gap=None,
        refractory=None,
    ):
        """Return all valid above-threshold onset runs."""
        return _enumerate_detection_runs(
            candidates,
            threshold=threshold,
            min_consecutive=min_consecutive,
            min_duration=min_duration,
            require_stable_prediction=require_stable_prediction,
            merge_gap=merge_gap,
            refractory=refractory,
            segmenter=onset._candidate_segments,  # noqa: SLF001
            duration=onset._run_duration,  # noqa: SLF001
        )

    def _first_detection_run(candidates, *, threshold, min_consecutive, min_duration, require_stable_prediction):
        return _select_first_detection_run(
            candidates,
            threshold=threshold,
            min_consecutive=min_consecutive,
            min_duration=min_duration,
            require_stable_prediction=require_stable_prediction,
            segmenter=onset._candidate_segments,  # noqa: SLF001
            duration=onset._run_duration,  # noqa: SLF001
        )

    onset._detection_runs = _detection_runs  # noqa: SLF001
    onset._first_detection_run = _first_detection_run  # noqa: SLF001


def install() -> None:
    _install_onset_detection_extensions()
