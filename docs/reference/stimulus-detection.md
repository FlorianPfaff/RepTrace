# Stimulus Detection

`reptrace.stimulus_detection` detects zero, one, or many stimulus events in a
long probability stream. It is the stream-oriented counterpart to
`reptrace.onset_detection`, which reports the first threshold crossing per
trial-like probability-observation sequence.

Use this module when a decoder has produced a time series of class probabilities
and the question is:

> Did one of the possible stimuli occur in this stream, and if so, when?

The detector works with the usual RepTrace probability-observation columns:

| Column | Meaning |
| --- | --- |
| `time` | Center time of the decoding window. |
| `stream_id` | Identifier for the long stream, run, session, or block. |
| `sequence_id` | Accepted fallback when `stream_id` is absent. |
| `prob_class_0`, `prob_class_1`, ... | Decoder probability for each stimulus class. |
| `class_0`, `class_1`, ... | Optional human-readable class names. |
| `window_start`, `window_stop` | Optional window boundaries used for event duration. |
| `subject`, `decoder`, `emission_mode` | Optional grouping columns. |

The event output has one row per detected stimulus event. Important columns are:

| Column | Meaning |
| --- | --- |
| `event_index` | Event number within the stream. |
| `stimulus_class` | Detected stimulus class name. |
| `stimulus_label` | Detected stimulus class index or label. |
| `onset_time` | First above-threshold time bin in the event run. |
| `offset_time` | Last above-threshold time bin in the event run. |
| `peak_time` | Time bin with the largest event score. |
| `detection_confirmed_time` | First time at which persistence requirements are satisfied. |
| `run_length` | Number of above-threshold bins in the event. |
| `run_duration` | Duration of the event run. |
| `peak_score` | Largest event score in the run. |
| `score_threshold` | Threshold used for this class and group. |
| `matched_annotation_id` | Optional matched ground-truth event. |
| `latency` | Detected onset minus annotated onset. |
| `is_true_positive` | Whether the event matched an annotation. |

## Minimal input example

An observation CSV may look like this:

| stream_id | time | class_0 | class_1 | prob_class_0 | prob_class_1 |
| --- | ---: | --- | --- | ---: | ---: |
| run-1 | -0.30 | face | object | 0.52 | 0.48 |
| run-1 | -0.20 | face | object | 0.55 | 0.45 |
| run-1 | -0.10 | face | object | 0.53 | 0.47 |
| run-1 | 0.10 | face | object | 0.89 | 0.11 |
| run-1 | 0.20 | face | object | 0.91 | 0.09 |
| run-1 | 0.70 | face | object | 0.18 | 0.82 |
| run-1 | 0.80 | face | object | 0.12 | 0.88 |

An optional annotation CSV may look like this:

| stream_id | annotation_id | stimulus_class | onset_time |
| --- | ---: | --- | ---: |
| run-1 | 1 | face | 0.10 |
| run-1 | 2 | object | 0.70 |

## CLI example

Without annotations:

```bash
python -m reptrace.stimulus_detection \
  results/sub-01_stream_observations.csv \
  --stream-column sequence_id \
  --score-mode class_probability \
  --threshold-window -0.35 -0.05 \
  --threshold-method max_run \
  --threshold-quantile 0.95 \
  --min-consecutive 2 \
  --merge-gap 0.05 \
  --refractory 0.20 \
  --out-events results/stimulus_events.csv \
  --out-summary results/stimulus_event_summary.csv
```

With annotations:

```bash
python -m reptrace.stimulus_detection \
  results/sub-01_stream_observations.csv \
  --annotations results/sub-01_stimulus_annotations.csv \
  --stream-column stream_id \
  --score-mode class_probability \
  --threshold-window -0.35 -0.05 \
  --threshold-method max_run \
  --threshold-quantile 0.95 \
  --detection-window 0.0 inf \
  --min-consecutive 2 \
  --merge-gap 0.05 \
  --refractory 0.20 \
  --match-tolerance 0.10 \
  --out-events results/sub-01_stimulus_events.csv \
  --out-summary results/sub-01_stimulus_event_summary.csv \
  --out-thresholds results/sub-01_stimulus_thresholds.csv
```

`--annotations-csv` remains accepted as a backwards-compatible alias for `--annotations`.

This command:

1. derives class-specific thresholds from the baseline window;
2. scans the post-baseline stream for above-threshold stimulus runs;
3. merges brief interruptions shorter than `--merge-gap`;
4. suppresses close duplicate detections with `--refractory`;
5. optionally matches detections to annotated stimulus onsets; and
6. writes event, summary, and threshold tables.

## Python API example

```python
import pandas as pd

from reptrace.stimulus_detection import (
    detect_stimulus_events,
    fit_stimulus_detection_thresholds,
    match_stimulus_annotations,
    summarize_stimulus_events,
)

observations = pd.read_csv("results/sub-01_stream_observations.csv")
annotations = pd.read_csv("results/sub-01_stimulus_annotations.csv")

thresholds = fit_stimulus_detection_thresholds(
    observations,
    stream_columns=("stream_id",),
    threshold_window=(-0.35, -0.05),
    threshold_method="max_run",
    threshold_quantile=0.95,
    score_mode="class_probability",
    min_consecutive=2,
)

events = detect_stimulus_events(
    observations,
    thresholds=thresholds,
    stream_columns=("stream_id",),
    detection_window=(0.0, float("inf")),
    min_consecutive=2,
    merge_gap=0.05,
    refractory=0.20,
)

events = match_stimulus_annotations(
    events,
    annotations,
    stream_columns=("stream_id",),
    match_tolerance=0.10,
)

summary = summarize_stimulus_events(events, annotations=annotations)
```

## Choosing a score mode

`score_mode="class_probability"` scans each `prob_class_*` column as a separate
stimulus evidence trace. This is the recommended mode when the task is to detect
which stimulus occurred in a long stream.

`score_mode="predicted_class_confidence"` uses the decoder confidence only when
the predicted class matches the candidate stimulus. This is useful when event
detection should follow the decoder's winning class rather than independent
class-probability traces.

## Onset time versus confirmed detection time

`onset_time` is the first above-threshold bin of the event. It is useful for
offline latency analyses.

`detection_confirmed_time` is the first time at which the event satisfied the
persistence settings, such as `min_consecutive` or `min_duration`. This is the
more realistic time for online or causal detection, because the detector must
observe enough evidence before confirming the event.

## API reference

::: reptrace.stimulus_detection
