# Examples

Run the synthetic time-resolved decoding example with:

```bash
python examples/basic/time_resolved_decoding.py
```

The example creates a small MNE `EpochsArray`, attaches metadata, and runs the
same decoder used for public benchmark data.

Plot the output with:

```bash
python -m reptrace.plot_time_decode \
  results/synthetic_decoding.csv \
  --chance 0.5 \
  --out results/synthetic_decoding.png
```

## Public datasets

- [NOD](nod/README.md) covers the first natural-image semantic decoding
  benchmarks.
- [OpenNeuro ds000117](ds000117/README.md) stages a small face-vs-scrambled MEG
  benchmark for cleaner onset-detection sanity checks.

## Stream stimulus detection

After exporting probability observations with `prob_class_*` columns, detect
zero, one, or many stimulus events in a long stream with:

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

When annotated stimulus onsets are available, pass them for event-level matching
and latency summaries:

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
  --out-events results/sub-01_stimulus_events.csv \
  --out-summary results/sub-01_stimulus_event_summary.csv \
  --out-thresholds results/sub-01_stimulus_thresholds.csv
```
