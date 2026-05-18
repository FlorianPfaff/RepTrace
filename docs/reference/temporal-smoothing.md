# Temporal Smoothing

Temporal posterior smoothing can be run directly on probability observations
exported by time decoding:

```bash
python -m reptrace.temporal_smoothing \
  results/nod_animate_logistic_temporal_smoothing_all/observations/*_observations.csv \
  --out-observations results/nod_animate_logistic_temporal_smoothing_all/temporal_smoothing/smoothed_observations.csv \
  --out-metrics results/nod_animate_logistic_temporal_smoothing_all/temporal_smoothing/smoothed_metrics.csv \
  --fit-window 0.1 0.8
```

For benchmark comparisons, prefer `reptrace.benchmark
--temporal-smoothing-dir`. That runs smoothing on the exact held-out
observations from the decoder run and aggregates raw plus smoothed metrics in
one summary table.

::: reptrace.temporal_smoothing
