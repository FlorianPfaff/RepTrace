"""Runtime extensions for onset run detection."""

from __future__ import annotations


def _install_onset_detection_extensions() -> None:
    from reptrace import onset_detection as onset

    def _detection_runs(candidates, *, threshold, min_consecutive, min_duration, require_stable_prediction):
        """Return all valid above-threshold onset runs."""
        if min_consecutive < 1:
            raise ValueError("min_consecutive must be at least 1.")
        if min_duration is not None and min_duration < 0:
            raise ValueError("min_duration must be non-negative when provided.")
        runs = []
        for segment in onset._candidate_segments(  # noqa: SLF001
            candidates,
            threshold=threshold,
            require_stable_prediction=require_stable_prediction,
        ):
            if len(segment) < min_consecutive:
                continue
            if min_duration is not None and onset._run_duration(segment) < min_duration:  # noqa: SLF001
                continue
            runs.append(segment)
        return runs

    def _first_detection_run(candidates, *, threshold, min_consecutive, min_duration, require_stable_prediction):
        runs = _detection_runs(
            candidates,
            threshold=threshold,
            min_consecutive=min_consecutive,
            min_duration=min_duration,
            require_stable_prediction=require_stable_prediction,
        )
        return runs[0] if runs else None

    onset._detection_runs = _detection_runs  # noqa: SLF001
    onset._first_detection_run = _first_detection_run  # noqa: SLF001


def install() -> None:
    _install_onset_detection_extensions()
