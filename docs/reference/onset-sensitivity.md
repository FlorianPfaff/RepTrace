# Onset Sensitivity

`reptrace.onset_sensitivity` runs onset detection over a grid of threshold and persistence settings.

Example:

```bash
python -m reptrace.onset_sensitivity \
  --task-dir results/nod_animate_all \
  --task-dir results/nod_superclass_canine_device_all \
  --task-dir results/nod_superclass_container_covering_all \
  --threshold-window -0.100 0.000 \
  --threshold-methods point max_run \
  --threshold-quantiles 0.90 0.95 0.975 \
  --detection-start 0.000 \
  --detection-window 0.000 0.800 \
  --min-consecutive-values 1 2 3 \
  --include-stable-prediction \
  --out-dir results/onset_sensitivity_all \
  --plot-out results/onset_sensitivity_all/onset_sensitivity.png
```

The workflow writes `onset_sensitivity_summary.csv` with one row per threshold method, threshold quantile, persistence setting, and task summary, plus `onset_sensitivity_robustness.csv` with latency spread and detection-quality aggregates across settings.

Use a post-stimulus `--detection-window` for latency sweeps, and use an earlier
window start when evaluating false-alarm behavior.

::: reptrace.onset_sensitivity
