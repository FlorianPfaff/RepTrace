# Onset Detection

`reptrace.onset_detection` detects the first threshold-crossing time in held-out probability-observation traces.

The basic detector estimates a score threshold from a baseline window, then reports the first time each trial/sequence crosses that threshold. The module also supports sustained-onset criteria to reduce one-bin false detections.

## Sustained-onset controls

Use the following options to require a more persistent representation onset:

```bash
python -m reptrace.onset_detection \
  results/nod_sub-01_animate_observations.csv \
  --threshold-window -0.10 0.00 \
  --threshold-quantile 0.95 \
  --detection-start 0.00 \
  --min-consecutive 3 \
  --require-stable-prediction \
  --out-events results/nod_sub-01_animate_onset_events.csv \
  --out-summary results/nod_sub-01_animate_onset_summary.csv
```

- `--min-consecutive` requires at least this many adjacent above-threshold windows.
- `--min-duration` requires the above-threshold run to last at least the given duration in seconds.
- `--require-stable-prediction` breaks an onset run when the predicted class changes across adjacent above-threshold bins.

The event CSV includes the run length, run duration, run stop time, and peak score within the detection run. The summary CSV reports detection rates, false-alarm rates, post-zero detection rates, correct-at-detection rates, and median post-zero detection latencies.

::: reptrace.onset_detection
