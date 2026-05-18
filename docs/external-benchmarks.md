# External Benchmarks

RepTrace keeps dataset-specific file conversion outside the core decoder while
standardizing the downstream benchmark interface. For external datasets, stage
local files into MNE `Epochs` plus a metadata CSV, write a RepTrace manifest,
validate the manifest, then run the same `reptrace benchmark` and downstream
probability-trace workflows used for NOD.

## THINGS-EEG2 author-preprocessed arrays

The installed `reptrace-stage-things-eeg2` command converts the THINGS-EEG2
author-preprocessed NumPy arrays into the RepTrace manifest format. It expects a
per-subject array with shape `condition x repetition x channel x time`, plus
`ch_names` and `times` entries. The command writes one `.fif` epochs file, one
metadata CSV, and one benchmark manifest row per subject and decoder.

A compact smoke-test benchmark can be staged with two labels, a small number of
conditions per label, and a small number of repetitions:

```bash
reptrace-stage-things-eeg2 \
  --things-root data/things_eeg2 \
  --staged-dir data/things_eeg2_reptrace \
  --manifest-out benchmarks/things_eeg2_animacy_first5.csv \
  --subjects 1 2 3 4 5 \
  --partition test \
  --label-map-csv data/things_eeg2/condition_labels.csv \
  --label-map-key-column image_condition \
  --label-map-label-column animacy \
  --label-column condition \
  --group-column image_condition \
  --target-labels animate inanimate \
  --max-conditions-per-label 80 \
  --max-repetitions 4 \
  --decoders logistic linear_svm \
  --n-splits 2
```

The grouped CLI exposes the same workflow:

```bash
reptrace stage-things-eeg2 \
  --things-root data/things_eeg2 \
  --staged-dir data/things_eeg2_reptrace \
  --manifest-out benchmarks/things_eeg2_animacy_first5.csv \
  --subjects 1 2 3 4 5 \
  --label-map-csv data/things_eeg2/condition_labels.csv \
  --label-map-label-column animacy \
  --target-labels animate inanimate \
  --decoders logistic linear_svm \
  --n-splits 2
```

The label-map CSV must contain one row per image condition. The default key is
`image_condition`; use `--label-map-key-column` when the condition identifier has
a different name. The label column is arbitrary, which makes the same staging
command usable for animacy, category, superclass, or custom semantic contrasts.
If the label map contains separate rows for training and test partitions, pass
`--label-map-partition-column` so partition-specific labels are used.

## Run the frozen RepTrace benchmark

After staging, validate before decoding:

```bash
reptrace validate-manifest \
  benchmarks/things_eeg2_animacy_first5.csv \
  --report-out results/things_eeg2_animacy_first5_validation.csv
```

Then run the same benchmark machinery used by the NOD experiments:

```bash
reptrace benchmark \
  benchmarks/things_eeg2_animacy_first5.csv \
  --out-dir results/things_eeg2_animacy_first5 \
  --aggregate-out results/things_eeg2_animacy_first5/summary.csv \
  --plot-out results/things_eeg2_animacy_first5/summary.png \
  --calibration-dir results/things_eeg2_animacy_first5/calibration \
  --observation-dir results/things_eeg2_animacy_first5/observations \
  --emission-mode both \
  --chance 0.5 \
  --resume
```

For result comparability, keep the decoder set, time window, fold grouping,
calibration output, and observation export fixed across NOD and THINGS runs.
Tune only the dataset adapter settings that define the semantic contrast and the
number of available trials.

## Downstream probability-trace analyses

The staged THINGS run can feed the same temporal-state analyses as NOD:

```bash
reptrace temporal-model \
  "results/things_eeg2_animacy_first5/observations/*_observations.csv" \
  --out-summary results/things_eeg2_animacy_first5/temporal_model.csv \
  --out-states results/things_eeg2_animacy_first5/state_trace.csv \
  --n-permutations 100

python -m reptrace.emission_compare \
  results/things_eeg2_animacy_first5/temporal_model.csv \
  --out-csv results/things_eeg2_animacy_first5/emission_compare.csv \
  --out-report results/things_eeg2_animacy_first5/emission_compare.md

python -m reptrace.semantic_stages \
  results/things_eeg2_animacy_first5/state_trace.csv \
  --out-time results/things_eeg2_animacy_first5/semantic_stage_time.csv \
  --out-stages results/things_eeg2_animacy_first5/semantic_stages.csv \
  --out-report results/things_eeg2_animacy_first5/semantic_stages.md
```

Use the smoke-test settings above to check the data path. For a paper-facing
external generalization result, remove the smoke-test caps, document the label
map, and report the same aggregate accuracy, Brier score, expected calibration
error, log loss, and temporal-state controls as for the NOD benchmark.
