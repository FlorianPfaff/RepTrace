# Examples

The `examples/` directory contains executable scripts and notes.

Start with `examples/basic/time_resolved_decoding.py` to see how to create a
small synthetic epochs object and run the time-resolved decoding pipeline.

The script writes `results/synthetic_decoding.csv`. Plot it with:

```bash
python -m reptrace.plot_time_decode \
  results/synthetic_decoding.csv \
  --chance 0.5 \
  --out results/synthetic_decoding.png
```
