# Data Staging

RepTrace does not download large public datasets automatically. Stage datasets
outside version control under `data/`, then validate the benchmark manifest
before running decoding.

## NOD-EEG and NOD-MEG

The first recommended public benchmark is NOD-EEG because it is smaller than
NOD-MEG and provides preprocessed epoch data.

- NOD-EEG: OpenNeuro `ds005811`
- NOD-MEG: OpenNeuro `ds005810`

For each subject, RepTrace expects:

- an epochs file named like `sub-01_epo.fif`;
- a detailed events CSV named like `sub-01_events.csv`;
- a manifest row pointing to both files.

The NOD records document detailed trial information under
`derivatives/detailed_events/sub-subID_events.csv`. The epoch data are stored in
the preprocessed epochs derivative area and named `sub-subID_epo.fif`.

## Manual Download

Use OpenNeuro's browser download, OpenNeuro CLI, or DataLad/git-annex. For a
small first pilot, download only one subject's epoch file and detailed events
file.

Example target layout:

```text
data/nod/sub-01_epo.fif
data/nod/sub-01_events.csv
```

Then update `benchmarks/nod_face_object.csv` if the staged filenames differ.

## Validate Before Decoding

Run:

```bash
python -m reptrace.validate_manifest \
  benchmarks/nod_face_object.csv \
  --report-out results/nod_manifest_validation.csv
```

Only run the benchmark after validation passes:

```bash
python -m reptrace.benchmark \
  benchmarks/nod_face_object.csv \
  --out-dir results/nod \
  --aggregate-out results/nod_face_object_summary.csv \
  --plot-out results/nod_face_object_summary.png \
  --chance 0.5
```
