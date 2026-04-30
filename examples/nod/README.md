# NOD Benchmark Notes

RepTrace expects NOD data to be staged locally. The first pilot should use one
subject's epoched `.fif` file and matching events CSV.

Prepare a binary animate/inanimate metadata file:

```bash
python -m reptrace.metadata \
  --events-csv data/nod/sub-01_events.csv \
  --source-column stim_is_animate \
  --positive-pattern "True" \
  --label-column condition \
  --positive-label animate \
  --negative-label inanimate \
  --out data/nod/sub-01_metadata_animate.csv
```

Run time-resolved decoding:

```bash
python -m reptrace.mne_time_decode \
  --epochs data/nod/sub-01_epo.fif \
  --metadata-csv data/nod/sub-01_metadata_animate.csv \
  --label-column condition \
  --group-column session \
  --out results/nod_sub-01_animate.csv
```

Plot a subject:

```bash
python -m reptrace.plot_time_decode \
  results/nod_sub-01_animate.csv \
  --chance 0.5 \
  --out results/nod_sub-01_animate.png
```

Aggregate multiple subjects:

```bash
python -m reptrace.results \
  results/nod_sub-01_animate.csv \
  results/nod_sub-02_animate.csv \
  --out results/nod_animate_summary.csv
```

Or run the manifest:

```bash
python -m reptrace.validate_manifest \
  benchmarks/nod_animate_sub01.csv \
  --report-out results/nod_animate_sub01_validation.csv

python -m reptrace.benchmark \
  benchmarks/nod_animate_sub01.csv \
  --out-dir results/nod_animate_sub01 \
  --aggregate-out results/nod_animate_sub01_summary.csv \
  --plot-out results/nod_animate_sub01_summary.png \
  --chance 0.5
```

The sub-01 pilot uses `stim_is_animate` because the detailed event file has a
usable animate/inanimate split.
