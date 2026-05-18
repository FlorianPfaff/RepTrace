"""Compatibility installer for onset run and threshold helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

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


def _metadata_text_matches(observations: pd.DataFrame, column: str, expected: object) -> bool:
    if column not in observations.columns:
        return False
    values = observations[column].dropna()
    return not values.empty and values.astype(str).eq(str(expected)).all()


def _metadata_number_matches(observations: pd.DataFrame, column: str, expected: float) -> bool:
    if column not in observations.columns:
        return False
    values = pd.to_numeric(observations[column], errors="coerce")
    return not values.empty and not values.isna().any() and np.allclose(values.to_numpy(dtype=float), float(expected))


def _metadata_optional_number_matches(observations: pd.DataFrame, column: str, expected: float | None) -> bool:
    if column not in observations.columns:
        return False
    values = pd.to_numeric(observations[column], errors="coerce")
    if values.empty:
        return False
    if expected is None:
        return values.isna().all()
    return not values.isna().any() and np.allclose(values.to_numpy(dtype=float), float(expected))


def _metadata_bool_matches(observations: pd.DataFrame, column: str, expected: bool) -> bool:
    if column not in observations.columns:
        return False
    values = observations[column].dropna().astype(str).str.lower()
    mapped = values.map({"true": True, "1": True, "false": False, "0": False})
    return not values.empty and not mapped.isna().any() and mapped.eq(bool(expected)).all()


def _install_threshold_annotation_extensions() -> None:
    """Reject cached onset-threshold annotations when result-relevant parameters differ."""
    from reptrace import onset_detection as onset

    if getattr(onset, "_threshold_annotation_compat_patched", False):
        return

    original_annotate_group_threshold = onset._annotate_group_threshold  # noqa: SLF001

    def _annotate_group_threshold(
        frame: pd.DataFrame,
        *,
        threshold_window: tuple[float, float],
        threshold_quantile: float,
        score_column: str,
        threshold_method: str,
        min_consecutive: int,
        min_duration: float | None,
        require_stable_prediction: bool,
    ) -> pd.DataFrame:
        thresholded = original_annotate_group_threshold(
            frame,
            threshold_window=threshold_window,
            threshold_quantile=threshold_quantile,
            score_column=score_column,
            threshold_method=threshold_method,
            min_consecutive=min_consecutive,
            min_duration=min_duration,
            require_stable_prediction=require_stable_prediction,
        )
        thresholded["min_consecutive"] = min_consecutive
        thresholded["min_duration"] = min_duration if min_duration is not None else np.nan
        thresholded["require_stable_prediction"] = require_stable_prediction
        return thresholded

    def _has_matching_threshold_annotation(
        observations: pd.DataFrame,
        *,
        threshold_method: str,
        threshold_quantile: float,
        score_column: str,
        threshold_window: tuple[float, float],
        min_consecutive: int,
        min_duration: float | None,
        require_stable_prediction: bool,
    ) -> bool:
        if "score_threshold" not in observations.columns:
            return False
        if "_onset_score" not in observations.columns and "onset_score" not in observations.columns:
            return False
        return (
            _metadata_text_matches(observations, "threshold_method", threshold_method)
            and _metadata_text_matches(observations, "score_column", score_column)
            and _metadata_number_matches(observations, "threshold_quantile", threshold_quantile)
            and _metadata_number_matches(observations, "threshold_window_start", threshold_window[0])
            and _metadata_number_matches(observations, "threshold_window_stop", threshold_window[1])
            and _metadata_number_matches(observations, "min_consecutive", min_consecutive)
            and _metadata_optional_number_matches(observations, "min_duration", min_duration)
            and _metadata_bool_matches(observations, "require_stable_prediction", require_stable_prediction)
        )

    def _prepare_thresholded_observations(
        observations: pd.DataFrame,
        *,
        threshold_window: tuple[float, float],
        threshold_quantile: float,
        score_column: str,
        threshold_method: str,
        min_consecutive: int,
        min_duration: float | None,
        require_stable_prediction: bool,
    ) -> pd.DataFrame:
        if _has_matching_threshold_annotation(
            observations,
            threshold_method=threshold_method,
            threshold_quantile=threshold_quantile,
            score_column=score_column,
            threshold_window=threshold_window,
            min_consecutive=min_consecutive,
            min_duration=min_duration,
            require_stable_prediction=require_stable_prediction,
        ):
            thresholded = onset._ensure_prediction_columns(observations).copy()  # noqa: SLF001
            if "_onset_score" not in thresholded.columns:
                thresholded["_onset_score"] = pd.to_numeric(thresholded["onset_score"], errors="coerce")
            thresholded["above_threshold"] = thresholded["_onset_score"] >= pd.to_numeric(
                thresholded["score_threshold"],
                errors="coerce",
            )
            return thresholded
        return onset.annotate_threshold_crossings(
            observations,
            threshold_window=threshold_window,
            threshold_quantile=threshold_quantile,
            score_column=score_column,
            threshold_method=threshold_method,
            min_consecutive=min_consecutive,
            min_duration=min_duration,
            require_stable_prediction=require_stable_prediction,
        )

    onset._annotate_group_threshold = _annotate_group_threshold  # noqa: SLF001
    onset._has_matching_threshold_annotation = _has_matching_threshold_annotation  # noqa: SLF001
    onset._prepare_thresholded_observations = _prepare_thresholded_observations  # noqa: SLF001
    onset._threshold_annotation_compat_patched = True  # noqa: SLF001


def install() -> None:
    _install_onset_detection_extensions()
    _install_threshold_annotation_extensions()
