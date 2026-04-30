# Benchmarking

The first intended public benchmark is NOD-MEG/NOD-EEG. RepTrace does not
download large public datasets automatically; stage the relevant subject epochs
and metadata locally first.

## NOD Face/Object Pilot

Use the NOD preprocessed epochs file and the matching detailed events CSV. If
the metadata already contains a binary face/object label, pass it directly to
`reptrace.mne_time_decode`. If it contains a text category column instead,
derive a binary label first:

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

Then run the decoder:

```bash
python -m reptrace.mne_time_decode \
  --epochs data/nod/sub-01_epo.fif \
  --metadata-csv data/nod/sub-01_metadata_face_object.csv \
  --label-column condition \
  --group-column run \
  --tmin -0.1 \
  --tmax 0.8 \
  --window-ms 20 \
  --step-ms 10 \
  --out results/nod_sub-01_face_object.csv
```

The output CSV contains fold-wise accuracy, log loss, Brier score, and expected
calibration error for each time window.

Plot the single-subject result:

```bash
python -m reptrace.plot_time_decode \
  results/nod_sub-01_face_object.csv \
  --chance 0.5 \
  --title "NOD sub-01 face/object" \
  --out results/nod_sub-01_face_object.png
```

After running several subjects, aggregate across subjects:

```bash
python -m reptrace.results \
  results/nod_sub-01_face_object.csv \
  results/nod_sub-02_face_object.csv \
  results/nod_sub-03_face_object.csv \
  --out results/nod_face_object_summary.csv
```

Then plot the aggregate:

```bash
python -m reptrace.plot_time_decode \
  results/nod_face_object_summary.csv \
  --chance 0.5 \
  --title "NOD face/object summary" \
  --out results/nod_face_object_summary.png
```

## Manifest Runner

The same workflow can be run from a manifest:

```bash
python -m reptrace.benchmark \
  benchmarks/nod_face_object.csv \
  --out-dir results/nod \
  --aggregate-out results/nod_face_object_summary.csv \
  --plot-out results/nod_face_object_summary.png \
  --chance 0.5
```

Manifest paths are resolved relative to the manifest file. The example manifest
expects staged files under `data/nod/`.

## Acceptance Target

The first useful milestone is not just above-chance accuracy. The benchmark
should produce stable probability traces and calibration metrics that can be
compared across subjects, sessions, and decoder variants.
