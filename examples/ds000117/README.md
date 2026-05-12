# OpenNeuro ds000117 Face Onset Benchmark

`ds000117` is the Wakeman/Henson multimodal face-recognition dataset on
OpenNeuro. It is a good next onset benchmark because the task has a clean
time-locked visual contrast and the MNE-BIDS-Pipeline already uses it for
time-by-time and temporal-generalization decoding examples.

The first RepTrace target is a conservative single-subject smoke test:

- `face`: `Famous` and `Unfamiliar`
- `scrambled`: `Scrambled`
- modality: MEG
- subject: `sub-01`
- runs: `01`, `02`

## Download A Small Subset

The full dataset is large. Start with the filtered single-subject download used
by the MNE-BIDS-Pipeline example:

```bash
python -m pip install openneuro-py

openneuro-py download \
  --dataset ds000117 \
  --tag 1.1.0 \
  --target-dir data/ds000117 \
  --include sub-01/ses-meg/meg/sub-01_ses-meg_task-facerecognition_run-01_meg.fif

openneuro-py download \
  --dataset ds000117 \
  --tag 1.1.0 \
  --target-dir data/ds000117 \
  --include sub-01/ses-meg/meg/sub-01_ses-meg_task-facerecognition_run-01_events.tsv

openneuro-py download \
  --dataset ds000117 \
  --tag 1.1.0 \
  --target-dir data/ds000117 \
  --include sub-01/ses-meg/meg/sub-01_ses-meg_task-facerecognition_run-02_meg.fif

openneuro-py download \
  --dataset ds000117 \
  --tag 1.1.0 \
  --target-dir data/ds000117 \
  --include sub-01/ses-meg/meg/sub-01_ses-meg_task-facerecognition_run-02_events.tsv
```

Source for the filtered download recipe:
<https://mne.tools/mne-bids-pipeline/stable/examples/ds000117.html>

## Stage RepTrace Epochs

Convert the downloaded BIDS-style FIF and events files into an MNE epochs file
and a RepTrace benchmark manifest:

```bash
python scripts/stage_ds000117_faces.py \
  --bids-root data/ds000117 \
  --staged-dir data/ds000117_reptrace \
  --manifest-out benchmarks/ds000117_faces_sub01.csv \
  --subjects 01 \
  --runs 01 02 \
  --max-events-per-label 40 \
  --decoders logistic \
  --n-splits 2 \
  --on-mismatch warn
```

Omit `--max-events-per-label` for the full sub-01 run after the smoke test is
working. The raw runs have different device-to-head transforms; the
`--on-mismatch warn` option preserves the smoke-test path, while
Maxwell-filtered derivatives are the better choice for a full multi-run
analysis.

Add `--decoders linear_svm` only for a separate calibration-aware comparison;
on raw sub-01 smoke data it is substantially slower than logistic regression.

## Run Time-Resolved Decoding

```bash
reptrace-validate-manifest \
  benchmarks/ds000117_faces_sub01.csv \
  --report-out results/ds000117_faces_sub01_validation.csv

reptrace-benchmark \
  benchmarks/ds000117_faces_sub01.csv \
  --out-dir results/ds000117_faces_sub01 \
  --aggregate-out results/ds000117_faces_sub01_summary.csv \
  --plot-out results/ds000117_faces_sub01_summary.png \
  --chance 0.5
```

For onset detection, run the benchmark with probability observations enabled
through the manifest workflow, then apply the same false-alarm-controlled
settings that worked best in the PyMEGDec stress test:

```bash
reptrace-onset-detect \
  results/ds000117_faces_sub01/observations/*_observations.csv \
  --threshold-window -0.2 0.0 \
  --threshold-method max_run \
  --threshold-quantile 0.96 \
  --min-consecutive 2 \
  --min-duration 0.05 \
  --out-events results/ds000117_faces_sub01_onset_events.csv \
  --out-summary results/ds000117_faces_sub01_onset_summary.csv \
  --out-threshold-summary results/ds000117_faces_sub01_onset_thresholds.csv
```

## Interpretation

This dataset should be treated as a high-SNR sanity check, not as a replacement
for NOD. The useful comparison is:

- PyMEGDec: hard stress test where naive point detection false-alarms badly.
- `ds000117`: cleaner visual face-vs-scrambled test where onset should be easier.
- NOD: semantic natural-image benchmark that connects back to the main RepTrace
  calibration story.
