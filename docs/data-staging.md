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
`derivatives/detailed_events/sub-subID_events.csv`. In NOD-EEG, epoch data are
stored in the preprocessed epochs derivative area and named like
`sub-01_eeg_epo.fif`; stage them under the simpler local filenames shown below.

## Manual Download

Use OpenNeuro's browser download, OpenNeuro CLI, or DataLad/git-annex. For a
small first pilot, download only one subject's epoch file and detailed events
file.

Example target layout:

```text
data/nod/sub-01_epo.fif
data/nod/sub-01_events.csv
```

Then update `benchmarks/nod_animate_sub01.csv` if the staged filenames differ.

## Validate Before Decoding

Run:

```bash
python -m reptrace.validate_manifest \
  benchmarks/nod_animate_sub01.csv \
  --report-out results/nod_animate_sub01_validation.csv
```

Only run the benchmark after validation passes:

```bash
python -m reptrace.benchmark \
  benchmarks/nod_animate_sub01.csv \
  --out-dir results/nod_animate_sub01 \
  --aggregate-out results/nod_animate_sub01_summary.csv \
  --plot-out results/nod_animate_sub01_summary.png \
  --chance 0.5
```
