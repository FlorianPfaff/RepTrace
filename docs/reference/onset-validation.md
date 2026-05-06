# Onset Validation

`reptrace.onset_validation` runs the onset detector inside named time chunks. Use it as a compact negative/positive control before interpreting onset latencies.

Example:

```bash
python -m reptrace.onset_validation \
  results/nod_sub-01_animate_observations.csv \
  --threshold-window -0.10 0.00 \
  --threshold-quantile 0.95 \
  --threshold-method max_run \
  --min-consecutive 2 \
  --chunk pre:-0.30:-0.05:null \
  --chunk early:0.05:0.20:early \
  --chunk late:0.20:0.60:positive \
  --out-events results/nod_sub-01_animate_onset_chunk_events.csv \
  --out-summary results/nod_sub-01_animate_onset_chunk_summary.csv
```

Pre-stimulus chunks should show low detection rates. Post-stimulus chunks should show higher detection rates only when the decoded probability traces contain stable task-related evidence.

::: reptrace.onset_validation
