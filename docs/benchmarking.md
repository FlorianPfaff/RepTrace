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

## Acceptance Target

The first useful milestone is not just above-chance accuracy. The benchmark
should produce stable probability traces and calibration metrics that can be
compared across subjects, sessions, and decoder variants.
