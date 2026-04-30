# NOD Benchmark Notes

RepTrace expects NOD data to be staged locally. The first pilot should use one
subject's epoched `.fif` file and matching events CSV.

Prepare a binary face/object metadata file:

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

Run time-resolved decoding:

```bash
python -m reptrace.mne_time_decode \
  --epochs data/nod/sub-01_epo.fif \
  --metadata-csv data/nod/sub-01_metadata_face_object.csv \
  --label-column condition \
  --group-column run \
  --out results/nod_sub-01_face_object.csv
```

Plot a subject:

```bash
python -m reptrace.plot_time_decode \
  results/nod_sub-01_face_object.csv \
  --chance 0.5 \
  --out results/nod_sub-01_face_object.png
```

Aggregate multiple subjects:

```bash
python -m reptrace.results \
  results/nod_sub-01_face_object.csv \
  results/nod_sub-02_face_object.csv \
  --out results/nod_face_object_summary.csv
```

Adjust `--source-column` and `--positive-pattern` to match the actual event
metadata columns available in the staged NOD files.
