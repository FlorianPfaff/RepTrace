# RepTrace

Probabilistic tracing of neural representations over time.

RepTrace is an early-stage Python toolkit for benchmarking calibrated,
time-resolved decoders on M/EEG data. The initial goal is to turn classifier
outputs from non-invasive neural recordings into probability traces that are
useful for studying representational dynamics, planning, and replay-like
sequences.

## Features

RepTrace currently provides tools for:

- time-resolved decoding from MNE `Epochs` files;
- calibrated classification metrics, including Brier score and expected
  calibration error;
- calibration-first reports and reliability-bin diagnostics;
- standard decoder baselines, including logistic regression, LDA, and
  calibrated linear SVM;
- grouped cross-validation for session- or run-aware benchmarks; and
- CSV aggregation, plotting, reporting, and subject-level inference for
  downstream interpretation.

## Installation

RepTrace requires Python 3.11 or newer and earlier than Python 3.14.

For development from a source checkout, use Poetry:

```bash
poetry install --with dev
```

Alternatively, install the package in editable mode with pip:

```bash
python -m pip install -e .
```

## Quickstart

Run the pilot NOD-EEG benchmark from a manifest:

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

Run time-resolved decoding directly on an MNE epochs file with metadata:

```bash
python -m reptrace.mne_time_decode \
  --epochs path/to/sub-01_epo.fif \
  --metadata-csv path/to/sub-01_events.csv \
  --label-column stim_is_animate \
  --group-column session \
  --out results/nod_sub-01_animate.csv
```

Plot the resulting time course:

```bash
python -m reptrace.plot_time_decode \
  results/nod_sub-01_animate.csv \
  --chance 0.5 \
  --out results/nod_sub-01_animate.png
```

If the events CSV has the NOD `stim_is_animate` column but no named decoding
condition yet, create one:

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

After running several subjects, aggregate them:

```bash
python -m reptrace.results \
  results/nod_sub-01_animate.csv \
  results/nod_sub-02_animate.csv \
  --out results/nod_animate_summary.csv
```

## Benchmark Plan

The first public benchmark target is NOD-MEG/NOD-EEG because the dataset
provides preprocessed MNE epochs and metadata for natural-image decoding. The
recommended first task is animate-versus-inanimate decoding from the NOD-EEG
`stim_is_animate` metadata. Face/object labels are too sparse in the initial
sub-01 detailed events slice to make a useful first pilot, but richer semantic
tasks can be added once the public baseline is reproducible.

THINGS-EEG and THINGS-MEG are natural follow-up benchmarks for larger visual
object representation experiments. Lab data with task localizers and planning
periods should come after these public baselines are reproducible.

## Documentation

The `docs/` directory contains the project documentation:

- [Getting started](docs/getting-started.md) covers installation and the first
  decoding run.
- [Data staging](docs/data-staging.md) describes the public NOD files to place
  under `data/`.
- [Benchmarking](docs/benchmarking.md) describes the initial NOD pilot.
- [API overview](docs/api-overview.md) maps the main public modules.
- [Examples](examples/README.md) lists executable examples.

Build the documentation site locally with:

```bash
poetry install --with docs --without dev
poetry run mkdocs build --strict
```

## Tests

Run the test suite from a development environment:

```bash
python -m pytest
```

## Citation

If you use **RepTrace** in your research, please cite the repository for now:

```bibtex
@software{pfaff_reptrace_2026,
  author = {Florian Pfaff},
  title = {RepTrace: Probabilistic Tracing of Neural Representations over Time},
  year = {2026},
  url = {https://github.com/FlorianPfaff/RepTrace},
  license = {MIT}
}
```

## License

`RepTrace` is licensed under the MIT License.
