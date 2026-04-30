# API Overview

RepTrace currently exposes these small modules:

- `reptrace.benchmark` runs manifest-defined multi-subject benchmarks.
- `reptrace.metrics` provides calibration and probabilistic scoring metrics.
- `reptrace.decoding` provides cross-validation and baseline decoder helpers.
- `reptrace.metadata` provides simple metadata labeling helpers.
- `reptrace.results` aggregates fold- and subject-level result CSV files.
- `reptrace.plot_time_decode` plots raw or aggregated time-resolved metrics.
- `reptrace.validate_manifest` checks staged files and metadata before decoding.
- `reptrace.mne_time_decode` provides the MNE epochs decoding command-line
  workflow.
