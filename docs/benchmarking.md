# Benchmarking

The first intended public benchmark is NOD-MEG/NOD-EEG. RepTrace does not
download large public datasets automatically; stage the relevant subject epochs
and metadata locally first.

## NOD Animate/Inanimate Pilot

Use the NOD preprocessed epochs file and the matching detailed events CSV. If
the metadata already contains `stim_is_animate`, pass it directly to
`reptrace.mne_time_decode` or derive a named condition column first:

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

Then run the decoder:

```bash
python -m reptrace.mne_time_decode \
  --epochs data/nod/sub-01_epo.fif \
  --metadata-csv data/nod/sub-01_metadata_animate.csv \
  --label-column condition \
  --group-column session \
  --tmin -0.1 \
  --tmax 0.8 \
  --window-ms 20 \
  --step-ms 10 \
  --out results/nod_sub-01_animate.csv
```

The output CSV contains fold-wise accuracy, log loss, Brier score, and expected
calibration error for each time window.

Plot the single-subject result:

```bash
python -m reptrace.plot_time_decode \
  results/nod_sub-01_animate.csv \
  --chance 0.5 \
  --title "NOD sub-01 animate/inanimate" \
  --out results/nod_sub-01_animate.png
```

After running several subjects, aggregate across subjects:

```bash
python -m reptrace.results \
  results/nod_sub-01_animate.csv \
  results/nod_sub-02_animate.csv \
  results/nod_sub-03_animate.csv \
  --out results/nod_animate_summary.csv
```

Then plot the aggregate:

```bash
python -m reptrace.plot_time_decode \
  results/nod_animate_summary.csv \
  --chance 0.5 \
  --title "NOD animate/inanimate summary" \
  --out results/nod_animate_summary.png
```

## Manifest Runner

The same workflow can be run from a manifest:

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

Manifest paths are resolved relative to the manifest file. The example manifest
expects staged files under `data/nod/`.

## Five-Subject Pilot

For a paper-ready first pass, use the same animate/inanimate task and run five
subjects at once from a single manifest:

```bash
python -m reptrace.validate_manifest \
  benchmarks/nod_animate_first5.csv \
  --report-out results/nod_animate_first5_validation.csv

python -m reptrace.benchmark \
  benchmarks/nod_animate_first5.csv \
  --out-dir results/nod_animate_first5 \
  --aggregate-out results/nod_animate_first5_summary.csv \
  --plot-out results/nod_animate_first5_summary.png \
  --chance 0.5
```

This keeps the experiment scope fixed (same preprocessing, same target labels,
same window/grid parameters) and changes only the subject set.

Generate a compact Markdown report from the aggregate and subject-level result
CSVs:

```bash
python -m reptrace.report \
  results/nod_animate_first5/summary.csv \
  "results/nod_animate_first5/sub-*_time_decode.csv" \
  --chance 0.5 \
  --out results/nod_animate_first5/report.md
```

The report records the aggregate peak, baseline-window accuracy,
effect-window accuracy, calibration metrics at the peak, and per-subject peaks.

## Full NOD-EEG Pilot

After staging all available NOD-EEG preprocessed epoch files and detailed event
files, validate the 19-subject manifest. This manifest uses 2 grouped folds
because several subjects have only 2 unique session groups:

```bash
python -m reptrace.validate_manifest \
  benchmarks/nod_animate_all.csv \
  --report-out results/nod_animate_all_validation.csv
```

Then run the same animate/inanimate benchmark over every staged NOD-EEG subject:

```bash
python -m reptrace.benchmark \
  benchmarks/nod_animate_all.csv \
  --out-dir results/nod_animate_all \
  --aggregate-out results/nod_animate_all/summary.csv \
  --plot-out results/nod_animate_all/nod_animate_all_summary.png \
  --chance 0.5
```

Make calibration explicit in the benchmark report:

```bash
python -m reptrace.calibration \
  results/nod_animate_all/summary.csv \
  --out-report results/nod_animate_all/calibration_report.md
```

The calibration report orders models by effect-window ECE, then Brier score and
log loss. Accuracy is included as context, but the report is designed to keep
probability quality visible rather than treating it as a secondary metric.

Run subject-level inference on the resulting subject CSVs:

```bash
python -m reptrace.inference \
  "results/nod_animate_all/sub-*_time_decode.csv" \
  --chance 0.5 \
  --n-permutations 10000 \
  --cluster-alpha 0.05 \
  --out-time results/nod_animate_all/inference_time.csv \
  --out-clusters results/nod_animate_all/inference_clusters.csv
```

The inference command first averages folds within each subject, then runs a
one-sided subject-level sign-flip test against chance at each time point. It
also reports max-cluster-mass corrected p-values for contiguous above-threshold
periods.

This larger run is the minimum useful scale for subject-level statistical
testing. The 5-subject pilot is useful for smoke testing and early signal
checking; paper-facing claims should use the full staged manifest.

## Decoder Comparison

RepTrace supports standard probability-producing decoders with the `decoder`
manifest column or `--decoder` CLI option:

- `logistic`: balanced multinomial logistic regression;
- `lda`: linear discriminant analysis;
- `linear_svm`: calibrated balanced linear support vector machine.

Run the first-five-subject decoder comparison:

```bash
python -m reptrace.benchmark \
  benchmarks/nod_animate_decoders_first5.csv \
  --out-dir results/nod_animate_decoders_first5 \
  --aggregate-out results/nod_animate_decoders_first5/summary.csv \
  --plot-out results/nod_animate_decoders_first5/summary.png \
  --calibration-dir results/nod_animate_decoders_first5/calibration \
  --chance 0.5
```

When a manifest contains a `decoder` column, result files are named like
`sub-01_logistic_time_decode.csv`, and aggregate summaries preserve the
`decoder` column rather than averaging decoders together.

Generate a decoder comparison report:

```bash
python -m reptrace.report \
  results/nod_animate_decoders_first5/summary.csv \
  --out results/nod_animate_decoders_first5/report.md
```

For decoder comparisons, the report includes both raw effect-window accuracy
and effect minus baseline-window accuracy. The baseline-corrected value is the
more relevant paper-facing number when a decoder shows pre-stimulus bias.

Create a calibration-first decoder report and aggregate reliability bins:

```bash
python -m reptrace.calibration \
  results/nod_animate_decoders_first5/summary.csv \
  "results/nod_animate_decoders_first5/calibration/*_calibration_bins.csv" \
  --out-report results/nod_animate_decoders_first5/calibration_report.md \
  --out-bins results/nod_animate_decoders_first5/reliability_bins.csv
```

## Acceptance Target

The first useful milestone is not just above-chance accuracy. The benchmark
should produce stable probability traces and calibration metrics that can be
compared across subjects, sessions, and decoder variants.
