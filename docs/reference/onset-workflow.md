# Onset Workflow

`reptrace.onset_workflow` runs onset detection across multiple task result directories and writes task-level plus combined summaries.

Example:

```bash
python -m reptrace.onset_workflow \
  --task-dir results/nod_animate_all \
  --task-dir results/nod_superclass_canine_device_all \
  --task-dir results/nod_superclass_container_covering_all \
  --threshold-window -0.100 0.000 \
  --threshold-quantile 0.95 \
  --threshold-method max_run \
  --detection-start 0.000 \
  --detection-window 0.000 0.800 \
  --min-consecutive 3 \
  --require-stable-prediction \
  --out-dir results/onset_detection_all \
  --plot-out results/onset_detection_all/onset_summary.png
```

The workflow looks for `observations/*_observations.csv` in each task directory by default. It writes per-task `onset_events.csv` and `onset_summary.csv` files, a combined `onset_summary_all.csv`, and optionally a compact plot and combined event table.

Use `--detection-window 0.000 0.800` for a post-stimulus latency benchmark. Use
an earlier start, for example `--detection-window -0.200 0.800`, when the goal is
to allow and count pre-stimulus false alarms.

::: reptrace.onset_workflow
