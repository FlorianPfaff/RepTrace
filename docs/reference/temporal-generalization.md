# Temporal Generalization

RepTrace supports temporal generalization in two complementary ways:

1. a dataset-independent matrix API for training at one time and testing at all
   other times; and
2. an MNE time-decoding ensemble mode that can directly improve reported
   time-resolved results.

For the MNE workflow, pass `--temporal-train-window START STOP` to train one
model for each decoding-window center inside the selected interval. Each model
is evaluated at every test time, and RepTrace averages the resulting class
probabilities before computing accuracy, log loss, Brier score, ECE,
calibration bins, and probability-observation exports.

```bash
reptrace-mne-time-decode \
  --epochs path/to/sub-01_epo.fif \
  --metadata-csv path/to/sub-01_events.csv \
  --label-column condition \
  --group-column session \
  --temporal-train-window 0.12 0.25 \
  --out results/sub-01_temporal_ensemble.csv \
  --observations-out results/sub-01_temporal_ensemble_observations.csv
```

The ensemble can be combined with nested decoder tuning:

```bash
reptrace-mne-time-decode \
  --epochs path/to/sub-01_epo.fif \
  --metadata-csv path/to/sub-01_events.csv \
  --label-column condition \
  --group-column session \
  --temporal-train-window 0.12 0.25 \
  --tune-hyperparameters \
  --tuning-cv-splits 2 \
  --tuning-scoring balanced_accuracy \
  --out results/sub-01_temporal_ensemble_tuned.csv
```

The emitted result and observation tables include provenance columns such as
`temporal_mode`, `train_time`, `test_time`, `train_window_start`,
`train_window_stop`, `temporal_train_window_start`,
`temporal_train_window_stop`, and `n_train_windows`, so the ensemble output
remains separable from same-time decoding runs and from other train-window
choices.

::: reptrace.decoding.temporal_generalization
