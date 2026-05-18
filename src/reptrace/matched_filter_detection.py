from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

DEFAULT_THRESHOLD_WINDOW = (-0.35, -0.05)
DEFAULT_THRESHOLD_QUANTILE = 0.95
DEFAULT_TEMPLATE_WINDOW = (0.0, 0.35)
DEFAULT_GROUP_COLUMNS = ("subject", "decoder", "emission_mode")
DEFAULT_STREAM_FALLBACKS = (("subject", "stream_id"), ("subject", "sequence_id"), ("stream_id",), ("sequence_id",), ("sample_index",))
MATCHED_FILTER_SCORE_COLUMN = "matched_filter_score"
MATCHED_FILTER_SCORE_MODE = "matched_filter"
MATCHED_FILTER_THRESHOLD_METHOD = "matched_filter"


def _present_columns(frame: pd.DataFrame, columns: Sequence[str]) -> list[str]:
    return [column for column in columns if column in frame.columns]


def _group_columns(frame: pd.DataFrame, group_columns: Sequence[str] | None = None) -> list[str]:
    return _present_columns(frame, DEFAULT_GROUP_COLUMNS if group_columns is None else group_columns)


def _stream_columns(frame: pd.DataFrame, stream_columns: Sequence[str] | None = None) -> list[str]:
    if stream_columns is not None:
        missing = [column for column in stream_columns if column not in frame.columns]
        if missing:
            raise ValueError(f"Observation rows are missing stream columns: {missing}")
        return list(stream_columns)
    for candidates in DEFAULT_STREAM_FALLBACKS:
        if all(column in frame.columns for column in candidates):
            return list(candidates)
    raise ValueError("Observation rows must contain stream_id, sequence_id, or sample_index.")


def _grouped(frame: pd.DataFrame, columns: Sequence[str], *, sort: bool = True):
    if not columns:
        return [((), frame)]
    by: str | list[str] = columns[0] if len(columns) == 1 else list(columns)
    return frame.groupby(by, sort=sort)


def _key_values(key: object, columns: Sequence[str]) -> dict[str, object]:
    values = key if isinstance(key, tuple) else (key,)
    return dict(zip(columns, values, strict=True))


def _filter_by_values(frame: pd.DataFrame, values: dict[str, object]) -> pd.DataFrame:
    filtered = frame
    for column, value in values.items():
        if column in filtered.columns:
            filtered = filtered.loc[filtered[column].astype(str) == str(value)]
    return filtered


def _probability_columns(frame: pd.DataFrame) -> list[str]:
    columns = [str(column) for column in frame.columns if str(column).startswith("prob_class_")]
    if not columns:
        raise ValueError("Observation rows must contain probability columns named 'prob_class_*'.")

    def sort_key(column: str) -> tuple[int, str]:
        suffix = column.removeprefix("prob_class_")
        return (int(suffix), suffix) if suffix.isdigit() else (10_000, suffix)

    return sorted(columns, key=sort_key)


def _class_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in _probability_columns(frame):
        suffix = column.removeprefix("prob_class_")
        try:
            label: int | str = int(suffix)
        except ValueError:
            label = suffix
        class_name = str(label)
        class_column = f"class_{suffix}"
        if class_column in frame.columns and frame[class_column].notna().any():
            class_name = str(frame[class_column].dropna().iloc[0])
        rows.append({"stimulus_label": label, "stimulus_class": class_name, "score_column": column})
    return pd.DataFrame(rows)


def _target_class_table(frame: pd.DataFrame, target_classes: Sequence[str | int] | None) -> pd.DataFrame:
    classes = _class_table(frame)
    if target_classes is None:
        return classes
    requested = {str(value) for value in target_classes}
    selected = classes.loc[classes["stimulus_label"].astype(str).isin(requested) | classes["stimulus_class"].astype(str).isin(requested)].reset_index(drop=True)
    if selected.empty:
        raise ValueError(f"No target classes matched {list(target_classes)!r}.")
    return selected


def _score_values(frame: pd.DataFrame, *, stimulus_label: int | str, stimulus_class: str, score_column: str, score_mode: str) -> pd.Series:
    if score_mode == "class_probability":
        if score_column not in frame.columns:
            raise ValueError(f"Score column '{score_column}' is missing.")
        return pd.to_numeric(frame[score_column], errors="coerce")
    if score_mode != "predicted_class_confidence":
        raise ValueError("score_mode must be 'class_probability' or 'predicted_class_confidence'.")
    if "confidence" not in frame.columns:
        raise ValueError("predicted_class_confidence scoring requires a confidence column.")
    confidence = pd.to_numeric(frame["confidence"], errors="coerce").fillna(0.0)
    if "predicted_label" in frame.columns:
        matches = frame["predicted_label"].astype(str).eq(str(stimulus_label))
    elif "predicted_class" in frame.columns:
        matches = frame["predicted_class"].astype(str).eq(str(stimulus_class))
    else:
        probabilities = frame[_probability_columns(frame)].to_numpy(dtype=float)
        matches = pd.Series(probabilities.argmax(axis=1).astype(str) == str(stimulus_label), index=frame.index)
    return confidence.where(matches, 0.0)


def _window_mask(frame: pd.DataFrame, window: tuple[float, float]) -> pd.Series:
    times = pd.to_numeric(frame["time"], errors="coerce")
    return (times >= window[0]) & (times <= window[1])


def _annotation_class_mask(annotations: pd.DataFrame, *, stimulus_label: int | str, stimulus_class: str) -> pd.Series:
    if "stimulus_class" in annotations.columns:
        return annotations["stimulus_class"].astype(str).eq(str(stimulus_class))
    if "stimulus_label" in annotations.columns:
        return annotations["stimulus_label"].astype(str).eq(str(stimulus_label))
    raise ValueError("Template annotations must contain stimulus_class or stimulus_label.")


def _infer_template_step(observations: pd.DataFrame, streams: Sequence[str]) -> float:
    deltas: list[float] = []
    for _, stream_frame in _grouped(observations, streams, sort=False):
        times = np.unique(np.sort(pd.to_numeric(stream_frame["time"], errors="coerce").dropna().to_numpy(dtype=float)))
        if len(times) > 1:
            deltas.extend(float(delta) for delta in np.diff(times) if delta > 0)
    if not deltas:
        raise ValueError("Cannot infer template_step from observation times; pass template_step explicitly.")
    return float(np.median(deltas))


def _template_offsets(template_window: tuple[float, float], template_step: float) -> np.ndarray:
    start, stop = map(float, template_window)
    if stop < start:
        raise ValueError("template_window stop must be greater than or equal to start.")
    if template_step <= 0:
        raise ValueError("template_step must be positive.")
    return start + np.arange(int(np.floor((stop - start) / template_step + 0.5)) + 1, dtype=float) * template_step


def _time_score_table(frame: pd.DataFrame, scores: pd.Series) -> pd.DataFrame:
    table = pd.DataFrame({"time": pd.to_numeric(frame["time"], errors="coerce"), "score": pd.to_numeric(scores, errors="coerce")}).dropna()
    return table.groupby("time", as_index=False, sort=True)["score"].mean() if not table.empty else table


def _interpolate(table: pd.DataFrame, query_times: np.ndarray) -> np.ndarray:
    if table.empty:
        return np.full(len(query_times), np.nan, dtype=float)
    return np.interp(query_times, table["time"].to_numpy(dtype=float), table["score"].to_numpy(dtype=float), left=np.nan, right=np.nan)


def fit_stimulus_event_templates(
    observations: pd.DataFrame,
    annotations: pd.DataFrame,
    *,
    template_window: tuple[float, float] = DEFAULT_TEMPLATE_WINDOW,
    template_step: float | None = None,
    score_mode: str = "class_probability",
    target_classes: Sequence[str | int] | None = None,
    group_columns: Sequence[str] | None = None,
    stream_columns: Sequence[str] | None = None,
    min_template_events: int = 1,
    min_template_coverage: float = 0.8,
) -> pd.DataFrame:
    """Fit class-specific matched-filter templates from annotated probability traces."""
    if "time" not in observations.columns:
        raise ValueError("Observation rows must contain a time column.")
    if "onset_time" not in annotations.columns:
        raise ValueError("Template annotations must contain onset_time.")
    if min_template_events < 1:
        raise ValueError("min_template_events must be at least 1.")
    if not 0 < min_template_coverage <= 1:
        raise ValueError("min_template_coverage must be in (0, 1].")

    groups = _group_columns(observations, group_columns)
    streams = _stream_columns(observations, stream_columns)
    offsets = _template_offsets(template_window, _infer_template_step(observations, streams) if template_step is None else float(template_step))
    classes = _target_class_table(observations, target_classes)
    rows: list[dict[str, object]] = []
    for group_key, group_frame in _grouped(observations, groups, sort=True):
        group_values = _key_values(group_key, groups)
        group_annotations = _filter_by_values(annotations, group_values)
        if group_annotations.empty:
            continue
        stream_groups = {stream_key: stream_frame for stream_key, stream_frame in _grouped(group_frame, streams, sort=False)}
        for _, class_row in classes.iterrows():
            class_annotations = group_annotations.loc[
                _annotation_class_mask(group_annotations, stimulus_label=class_row["stimulus_label"], stimulus_class=str(class_row["stimulus_class"]))
            ]
            if class_annotations.empty:
                continue
            scores = _score_values(
                group_frame,
                stimulus_label=class_row["stimulus_label"],
                stimulus_class=str(class_row["stimulus_class"]),
                score_column=str(class_row["score_column"]),
                score_mode=score_mode,
            )
            baseline = float(pd.to_numeric(scores, errors="coerce").dropna().median())
            event_vectors = []
            for _, annotation in class_annotations.iterrows():
                annotation_stream_values = {column: annotation[column] for column in streams if column in annotation and pd.notna(annotation[column])}
                matching_streams = stream_groups.items()
                if annotation_stream_values:
                    matching_streams = [
                        (stream_key, stream_frame)
                        for stream_key, stream_frame in stream_groups.items()
                        if all(str(_key_values(stream_key, streams)[column]) == str(value) for column, value in annotation_stream_values.items())
                    ]
                for _, stream_frame in matching_streams:
                    table = _time_score_table(stream_frame, scores.loc[stream_frame.index])
                    sampled = _interpolate(table, float(annotation["onset_time"]) + offsets)
                    if np.isfinite(sampled).mean() >= min_template_coverage:
                        event_vectors.append(sampled)
            if len(event_vectors) < min_template_events:
                continue
            template_values = np.nanmean(np.vstack(event_vectors), axis=0)
            excess = np.where(np.isfinite(template_values), template_values - baseline, 0.0)
            norm = float(np.sqrt(np.sum(excess**2)))
            if not np.isfinite(norm) or norm <= 0:
                continue
            for time, value, weight in zip(offsets, template_values, excess / norm, strict=True):
                rows.append(
                    {
                        **group_values,
                        "stimulus_label": class_row["stimulus_label"],
                        "stimulus_class": class_row["stimulus_class"],
                        "score_column": class_row["score_column"],
                        "score_mode": score_mode,
                        "template_time": float(time),
                        "template_value": float(value) if np.isfinite(value) else np.nan,
                        "template_weight": float(weight),
                        "baseline_score": baseline,
                        "n_template_events": len(event_vectors),
                    }
                )
    return pd.DataFrame(rows)


def _template_key_columns(templates: pd.DataFrame, groups: Sequence[str]) -> list[str]:
    return [column for column in [*groups, "stimulus_label", "stimulus_class", "score_column", "score_mode"] if column in templates.columns]


def score_stimulus_event_templates(
    observations: pd.DataFrame,
    templates: pd.DataFrame,
    *,
    group_columns: Sequence[str] | None = None,
    stream_columns: Sequence[str] | None = None,
    detection_window: tuple[float, float] | None = None,
    min_template_coverage: float = 0.8,
) -> pd.DataFrame:
    """Return matched-filter scores for candidate event-onset times."""
    if templates.empty:
        return pd.DataFrame()
    groups = _group_columns(observations, group_columns)
    streams = _stream_columns(observations, stream_columns)
    rows: list[dict[str, object]] = []
    for template_key, template in templates.groupby(_template_key_columns(templates, groups), sort=True, dropna=False):
        metadata = _key_values(template_key, _template_key_columns(templates, groups))
        group_values = {column: metadata[column] for column in groups if column in metadata}
        group_frame = _filter_by_values(observations, group_values) if group_values else observations
        if group_frame.empty:
            continue
        scores = _score_values(
            group_frame,
            stimulus_label=metadata["stimulus_label"],
            stimulus_class=str(metadata["stimulus_class"]),
            score_column=str(metadata["score_column"]),
            score_mode=str(metadata["score_mode"]),
        )
        template = template.sort_values("template_time")
        offsets = pd.to_numeric(template["template_time"], errors="coerce").to_numpy(dtype=float)
        weights = pd.to_numeric(template["template_weight"], errors="coerce").to_numpy(dtype=float)
        baseline = float(pd.to_numeric(template["baseline_score"], errors="coerce").dropna().iloc[0])
        n_template_events = int(pd.to_numeric(template["n_template_events"], errors="coerce").dropna().iloc[0])
        for stream_key, stream_frame in _grouped(group_frame, streams, sort=True):
            stream_values = _key_values(stream_key, streams)
            table = _time_score_table(stream_frame, scores.loc[stream_frame.index])
            for _, candidate in stream_frame.sort_values("time").iterrows():
                time = float(candidate["time"])
                if detection_window is not None and not (detection_window[0] <= time <= detection_window[1]):
                    continue
                sampled = _interpolate(table, time + offsets)
                finite = np.isfinite(sampled) & np.isfinite(weights)
                if not finite.size or float(finite.mean()) < min_template_coverage:
                    continue
                local_weights = weights[finite]
                norm = float(np.sqrt(np.sum(local_weights**2)))
                if not np.isfinite(norm) or norm <= 0:
                    continue
                score = float(np.dot(sampled[finite] - baseline, local_weights / norm))
                rows.append(
                    {
                        **group_values,
                        **stream_values,
                        "stimulus_label": metadata["stimulus_label"],
                        "stimulus_class": metadata["stimulus_class"],
                        "score_column": MATCHED_FILTER_SCORE_COLUMN,
                        "score_mode": MATCHED_FILTER_SCORE_MODE,
                        **candidate.to_dict(),
                        "_stimulus_score": score,
                        MATCHED_FILTER_SCORE_COLUMN: score,
                        "template_coverage": float(finite.mean()),
                        "n_template_events": n_template_events,
                    }
                )
    return pd.DataFrame(rows)


def fit_matched_filter_thresholds(
    observations: pd.DataFrame,
    templates: pd.DataFrame,
    *,
    threshold_window: tuple[float, float] = DEFAULT_THRESHOLD_WINDOW,
    threshold_quantile: float = DEFAULT_THRESHOLD_QUANTILE,
    group_columns: Sequence[str] | None = None,
    stream_columns: Sequence[str] | None = None,
    min_template_coverage: float = 0.8,
) -> pd.DataFrame:
    """Fit baseline-window thresholds for matched-filter scores."""
    if not 0 <= threshold_quantile <= 1:
        raise ValueError("threshold_quantile must be between 0 and 1.")
    groups = _group_columns(observations, group_columns)
    scores = score_stimulus_event_templates(observations, templates, group_columns=group_columns, stream_columns=stream_columns, min_template_coverage=min_template_coverage)
    if scores.empty:
        return pd.DataFrame()
    rows = []
    group_keys = [*groups, "stimulus_label", "stimulus_class"] if groups else ["stimulus_label", "stimulus_class"]
    for key, score_frame in scores.groupby(group_keys, sort=True, dropna=False):
        values = _key_values(key, group_keys)
        baseline_scores = pd.to_numeric(score_frame.loc[_window_mask(score_frame, threshold_window), MATCHED_FILTER_SCORE_COLUMN], errors="coerce").dropna()
        rows.append(
            {
                **{column: values[column] for column in groups},
                "stimulus_label": values["stimulus_label"],
                "stimulus_class": values["stimulus_class"],
                "score_column": MATCHED_FILTER_SCORE_COLUMN,
                "score_mode": MATCHED_FILTER_SCORE_MODE,
                "score_threshold": float(baseline_scores.quantile(threshold_quantile)) if not baseline_scores.empty else np.nan,
                "threshold_method": MATCHED_FILTER_THRESHOLD_METHOD,
                "threshold_quantile": threshold_quantile,
                "threshold_window_start": threshold_window[0],
                "threshold_window_stop": threshold_window[1],
            }
        )
    return pd.DataFrame(rows)


def _local_peak_mask(values: np.ndarray) -> np.ndarray:
    mask = np.zeros(len(values), dtype=bool)
    for index, value in enumerate(values):
        if not np.isfinite(value):
            continue
        left = values[index - 1] if index else -np.inf
        right = values[index + 1] if index + 1 < len(values) else -np.inf
        mask[index] = value >= left and value >= right
    return mask


def _select_refractory_peaks(peaks: pd.DataFrame, *, refractory: float | None) -> pd.DataFrame:
    if refractory is None or refractory <= 0 or peaks.empty:
        return peaks.sort_values("time", kind="mergesort")
    ranked = peaks.sort_values([MATCHED_FILTER_SCORE_COLUMN, "time"], ascending=[False, True], kind="mergesort")
    kept: list[object] = []
    kept_times: list[float] = []
    for index, row in ranked.iterrows():
        time = float(row["time"])
        if any(abs(time - kept_time) < refractory for kept_time in kept_times):
            continue
        kept.append(index)
        kept_times.append(time)
    return peaks.loc[kept].sort_values("time", kind="mergesort")


def _run_duration(row: pd.Series) -> float:
    if "window_start" in row and "window_stop" in row and pd.notna(row["window_start"]) and pd.notna(row["window_stop"]):
        return float(row["window_stop"] - row["window_start"])
    return 0.0


def detect_matched_filter_stimulus_events(
    observations: pd.DataFrame,
    *,
    templates: pd.DataFrame | None = None,
    template_annotations: pd.DataFrame | None = None,
    thresholds: pd.DataFrame | None = None,
    template_window: tuple[float, float] = DEFAULT_TEMPLATE_WINDOW,
    template_step: float | None = None,
    threshold_window: tuple[float, float] = DEFAULT_THRESHOLD_WINDOW,
    threshold_quantile: float = DEFAULT_THRESHOLD_QUANTILE,
    score_mode: str = "class_probability",
    target_classes: Sequence[str | int] | None = None,
    group_columns: Sequence[str] | None = None,
    stream_columns: Sequence[str] | None = None,
    detection_window: tuple[float, float] | None = None,
    refractory: float | None = None,
    min_template_events: int = 1,
    min_template_coverage: float = 0.8,
) -> pd.DataFrame:
    """Detect stimulus events by matched filtering class-probability templates."""
    if templates is None:
        if template_annotations is None:
            raise ValueError("templates or template_annotations must be provided for matched-filter detection.")
        templates = fit_stimulus_event_templates(
            observations,
            template_annotations,
            template_window=template_window,
            template_step=template_step,
            score_mode=score_mode,
            target_classes=target_classes,
            group_columns=group_columns,
            stream_columns=stream_columns,
            min_template_events=min_template_events,
            min_template_coverage=min_template_coverage,
        )
    groups = _group_columns(observations, group_columns)
    streams = _stream_columns(observations, stream_columns)
    scores = score_stimulus_event_templates(
        observations,
        templates,
        group_columns=group_columns,
        stream_columns=stream_columns,
        detection_window=detection_window,
        min_template_coverage=min_template_coverage,
    )
    if thresholds is None:
        thresholds = fit_matched_filter_thresholds(
            observations,
            templates,
            threshold_window=threshold_window,
            threshold_quantile=threshold_quantile,
            group_columns=group_columns,
            stream_columns=stream_columns,
            min_template_coverage=min_template_coverage,
        )
    if scores.empty or thresholds.empty:
        return pd.DataFrame(columns=[*groups, *streams, "event_index", "stimulus_class", "onset_time", "peak_score", "score_threshold", "detector_method"])

    rows = []
    threshold_group_columns = _present_columns(thresholds, groups)
    event_counters: dict[tuple[object, ...], int] = {}
    for _, threshold_row in thresholds.iterrows():
        threshold = float(threshold_row["score_threshold"])
        if not np.isfinite(threshold):
            continue
        group_values = {column: threshold_row[column] for column in threshold_group_columns}
        class_scores = _filter_by_values(scores, group_values) if group_values else scores
        class_scores = class_scores.loc[
            class_scores["stimulus_label"].astype(str).eq(str(threshold_row["stimulus_label"]))
            & class_scores["stimulus_class"].astype(str).eq(str(threshold_row["stimulus_class"]))
        ]
        for stream_key, stream_scores in _grouped(class_scores, streams, sort=True):
            stream_values = _key_values(stream_key, streams)
            ordered = stream_scores.sort_values("time", kind="mergesort").reset_index(drop=True)
            values = pd.to_numeric(ordered[MATCHED_FILTER_SCORE_COLUMN], errors="coerce").to_numpy(dtype=float)
            candidates = ordered.loc[_local_peak_mask(values) & (values > threshold)].copy()
            candidates = _select_refractory_peaks(candidates, refractory=refractory)
            stream_counter_key = tuple(stream_values[column] for column in streams)
            for _, peak_row in candidates.iterrows():
                event_index = event_counters.get(stream_counter_key, 0)
                event_counters[stream_counter_key] = event_index + 1
                score = float(peak_row[MATCHED_FILTER_SCORE_COLUMN])
                time = float(peak_row["time"])
                rows.append(
                    {
                        **group_values,
                        **stream_values,
                        "event_index": event_index,
                        "detected": True,
                        "stimulus_label": threshold_row["stimulus_label"],
                        "stimulus_class": threshold_row["stimulus_class"],
                        "onset_time": time,
                        "offset_time": time,
                        "peak_time": time,
                        "detection_confirmed_time": time,
                        "run_length": 1,
                        "run_duration": _run_duration(peak_row),
                        "score_at_onset": score,
                        "peak_score": score,
                        "score_threshold": threshold,
                        "score_column": MATCHED_FILTER_SCORE_COLUMN,
                        "score_mode": MATCHED_FILTER_SCORE_MODE,
                        "threshold_method": MATCHED_FILTER_THRESHOLD_METHOD,
                        "threshold_quantile": float(threshold_row["threshold_quantile"]),
                        "threshold_window_start": float(threshold_row["threshold_window_start"]),
                        "threshold_window_stop": float(threshold_row["threshold_window_stop"]),
                        "refractory": np.nan if refractory is None else refractory,
                        "detector_method": MATCHED_FILTER_THRESHOLD_METHOD,
                        "template_coverage": float(peak_row.get("template_coverage", np.nan)),
                        "n_template_events": int(peak_row.get("n_template_events", 0)),
                        "predicted_label_at_peak": peak_row.get("predicted_label", np.nan),
                        "predicted_class_at_peak": peak_row.get("predicted_class", ""),
                    }
                )
    if not rows:
        return pd.DataFrame(columns=[*groups, *streams, "event_index", "stimulus_class", "onset_time", "peak_score", "score_threshold", "detector_method"])
    events = pd.DataFrame(rows).sort_values([*streams, "onset_time", "stimulus_class"], kind="mergesort").reset_index(drop=True)
    for _, partition in _grouped(events, [*groups, *streams], sort=False):
        events.loc[partition.index, "event_index"] = range(len(partition))
    events["event_index"] = events["event_index"].astype(int)
    return events


__all__ = [
    "DEFAULT_TEMPLATE_WINDOW",
    "MATCHED_FILTER_SCORE_COLUMN",
    "MATCHED_FILTER_SCORE_MODE",
    "MATCHED_FILTER_THRESHOLD_METHOD",
    "detect_matched_filter_stimulus_events",
    "fit_matched_filter_thresholds",
    "fit_stimulus_event_templates",
    "score_stimulus_event_templates",
]
