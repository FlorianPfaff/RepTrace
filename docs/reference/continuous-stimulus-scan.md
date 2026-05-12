# Continuous Stimulus Scan

`reptrace.continuous_stimulus_scan` turns the long-stream event-detection idea
into a single reproducible workflow:

1. train an event-locked decoder on labeled events from one raw run;
2. scan a held-out raw run with the same window and preprocessing;
3. export `P(class | time)` as RepTrace stream observations;
4. run `reptrace.stimulus_detection`; and
5. write event-level precision, recall, F1, latency, and false-alarm summaries.

Use this when the question is:

> I have an event-locked decoder for a stimulus class. Does a held-out
> continuous recording contain intervals that look like that class?

## CLI Example

```bash
python -m reptrace.continuous_stimulus_scan \
  --train-raw data/ds000117/sub-01/ses-meg/meg/sub-01_ses-meg_task-facerecognition_run-01_meg.fif \
  --train-events data/ds000117/sub-01/ses-meg/meg/sub-01_ses-meg_task-facerecognition_run-01_events.tsv \
  --scan-raw data/ds000117/sub-01/ses-meg/meg/sub-01_ses-meg_task-facerecognition_run-02_meg.fif \
  --scan-events data/ds000117/sub-01/ses-meg/meg/sub-01_ses-meg_task-facerecognition_run-02_events.tsv \
  --source-column stim_type \
  --positive-pattern "Famous|Unfamiliar" \
  --negative-pattern "Scrambled" \
  --positive-label face \
  --negative-label scrambled \
  --target-class face \
  --train-window 0.15 0.25 \
  --picks meg \
  --demean-window \
  --slice-duration 6.0 \
  --slice-count 10 \
  --require-target-event \
  --exclude-events-from-threshold-window \
  --threshold-window 0.0 0.8 \
  --detection-window 0.8 6.0 \
  --threshold-method max_run \
  --threshold-quantile 0.975 \
  --min-consecutive 2 \
  --min-duration 0.05 \
  --merge-gap 0.05 \
  --refractory 0.30 \
  --match-tolerance 0.35 \
  --out-dir results/ds000117_continuous_scan
```

The installed command `reptrace-continuous-stimulus-scan` exposes the same
arguments.

## Outputs

The output directory contains:

| File | Meaning |
| --- | --- |
| `stream_observations.csv` | Long-stream probability observations with `prob_class_*` columns. |
| `stimulus_annotations.csv` | Held-out event annotations converted to stream-relative times. |
| `stimulus_thresholds.csv` | Class-specific detector thresholds. |
| `stimulus_events.csv` | One row per detected event. |
| `stimulus_summary.csv` | Precision, recall, F1, false alarms, and latency summaries. |
| `heldout_event_metrics.csv` | Event-locked held-out accuracy/log-loss before continuous scanning. |
| `training_class_counts.csv` | Training event counts per class. |

## API Reference

::: reptrace.continuous_stimulus_scan
