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
