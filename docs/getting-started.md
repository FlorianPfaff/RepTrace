# Getting Started

Install RepTrace from a source checkout:

```bash
poetry install --with dev
```

Run the first benchmark against an MNE epochs file:

```bash
python -m reptrace.mne_time_decode \
  --epochs path/to/sub-01_epo.fif \
  --label-column is_face \
  --out results/nod_sub-01_face_object.csv
```

Use `--metadata-csv` when the labels are stored outside the epochs metadata and
`--group-column` when cross-validation should keep sessions or runs separated.

If the metadata does not yet contain the binary decoding target, derive one from
a text column:

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
