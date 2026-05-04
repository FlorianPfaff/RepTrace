# API Overview

RepTrace currently exposes these small modules:

- `reptrace.benchmark` runs manifest-defined multi-subject benchmarks.
- `reptrace.calibration` creates calibration-first summaries and reliability
  bin aggregates.
- `reptrace.inference` runs subject-level sign-flip and cluster inference.
- `reptrace.metrics` provides calibration and probabilistic scoring metrics.
- `reptrace.decoding` provides cross-validation and baseline decoder helpers.
- `reptrace.emission_compare` compares calibrated and uncalibrated temporal
  state-inference evidence.
- `reptrace.metadata` provides simple metadata labeling helpers.
- `reptrace.results` aggregates fold- and subject-level result CSV files.
- `reptrace.semantic_stages` summarizes category-conditioned state traces into
  stable temporal stage candidates.
- `reptrace.temporal_model` fits conservative sticky switching models to
  held-out probability observations.
- `reptrace.plot_time_decode` plots raw or aggregated time-resolved metrics.
- `reptrace.plot_calibration` plots calibration reliability diagrams.
- `reptrace.paired_stats` runs paired subject-level decoder comparisons.
- `reptrace.report` creates compact Markdown reports from benchmark outputs.
- `reptrace.validate_manifest` checks staged files and metadata before decoding.
- `reptrace.mne_time_decode` provides the MNE epochs decoding command-line
  workflow, including optional held-out probability observation exports.
