# API Overview

RepTrace exposes modules for metadata preparation, manifest validation, MNE time decoding, result aggregation, calibration reporting, plotting, inference, paired decoder statistics,
probability-trace onset detection, continuous raw-stream stimulus scanning, stream-level stimulus event detection, onset chunk validation, multi-task onset workflows, onset sensitivity
analysis, sticky temporal modeling, emission comparison, semantic-stage analysis, and the calibration-aware temporal-state workflow.

Key command-line modules include:

- reptrace.metadata
- reptrace.validate_manifest
- reptrace.mne_time_decode
- reptrace.benchmark
- reptrace.continuous_stimulus_scan
- reptrace.results
- reptrace.report
- reptrace.calibration
- reptrace.plot_time_decode
- reptrace.plot_calibration
- reptrace.inference
- reptrace.paired_stats
- reptrace.onset_detection
- reptrace.stimulus_detection
- reptrace.onset_validation
- reptrace.onset_workflow
- reptrace.onset_sensitivity
- reptrace.temporal_model
- reptrace.emission_compare
- reptrace.semantic_stages
- reptrace.temporal_state_workflow

Reusable table-oriented APIs include:

- `reptrace.metrics` for calibration/probabilistic scoring metrics, pre/post window comparisons, and confusion-table summaries.
- `reptrace.decoding.alignment_window` for applying projections fitted on one M/EEG feature window to matching-channel features from another window.
- `reptrace.continuous_stimulus_scan` for training an event-locked decoder on one raw run, scanning a held-out raw run, exporting long-stream class probabilities, and scoring detected events.
- `reptrace.stimulus_detection` for detecting zero, one, or many stimulus events in long probability streams and evaluating them against annotation tables.
- `reptrace.results` for time-decoding aggregation, participant/window result tables, and peak-window diagnostics.
