"""Runtime extensions for onset run detection."""

from __future__ import annotations

from reptrace._onset_detection_runs import detection_runs as _enumerate_detection_runs
from reptrace._onset_detection_runs import first_detection_run as _select_first_detection_run


def _install_onset_detection_extensions() -> None:
    from reptrace import onset_detection as onset

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
