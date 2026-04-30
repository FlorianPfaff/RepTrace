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
- grouped cross-validation for session- or run-aware benchmarks; and
- CSV aggregation and plotting for downstream interpretation.

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

Run time-resolved decoding on an MNE epochs file with metadata:

```bash
python -m reptrace.mne_time_decode \
  --epochs path/to/sub-01_epo.fif \
  --label-column is_face \
  --out results/nod_sub-01_face_object.csv
```

Plot the resulting time course:

```bash
python -m reptrace.plot_time_decode \
  results/nod_sub-01_face_object.csv \
  --chance 0.5 \
  --out results/nod_sub-01_face_object.png
```

If labels are stored separately from the epochs metadata, pass a CSV with one
row per epoch:

```bash
python -m reptrace.mne_time_decode \
  --epochs path/to/sub-01_epo.fif \
  --metadata-csv path/to/sub-01_metadata.csv \
  --label-column is_face \
  --group-column run \
  --out results/nod_sub-01_face_object_grouped.csv
```

If the events CSV has a category column but no binary label yet, create one:

```bash
python -m reptrace.metadata \
  --events-csv data/nod/sub-01_events.csv \
  --source-column category \
  --positive-pattern "face|person" \
  --label-column condition \
  --positive-label face \
  --negative-label object \
  --out data/nod/sub-01_metadata_face_object.csv
```

After running several subjects, aggregate them:

```bash
python -m reptrace.results \
  results/nod_sub-01_face_object.csv \
  results/nod_sub-02_face_object.csv \
  --out results/nod_face_object_summary.csv
```

For reproducible multi-subject runs, use a benchmark manifest:

```bash
python -m reptrace.benchmark \
  benchmarks/nod_face_object.csv \
  --out-dir results/nod \
  --aggregate-out results/nod_face_object_summary.csv \
  --plot-out results/nod_face_object_summary.png \
  --chance 0.5
```

## Benchmark Plan

The first public benchmark target is NOD-MEG/NOD-EEG because the dataset
provides preprocessed MNE epochs and metadata for natural-object decoding. The
recommended first task is face-versus-object decoding, followed by multi-class
ImageNet or semantic-superclass decoding where metadata supports it.

THINGS-EEG and THINGS-MEG are natural follow-up benchmarks for larger visual
object representation experiments. Lab data with task localizers and planning
periods should come after these public baselines are reproducible.

## Documentation

The `docs/` directory contains the project documentation:

- [Getting started](docs/getting-started.md) covers installation and the first
  decoding run.
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
