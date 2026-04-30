# Data Staging

RepTrace does not download large public datasets automatically and does not
store staged data in git. Stage datasets outside version control under
`data/`, then validate the benchmark manifest before running decoding.

The repository tracks only code, documentation, and lightweight benchmark
manifests. `.gitignore` excludes `data/`, `results/`, and large local data
formats such as `*.fif`, `*.h5`, and `*.hdf5`.

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

## Recommended Pilot Download

For a small first pilot, download only one subject's epoch file and detailed
events file from OpenNeuro `ds005811`. The `openneuro-py` client supports
include filters, which avoids downloading the full dataset snapshot:

```bash
python -m pip install openneuro-py

openneuro-py download \
  --dataset=ds005811 \
  --target-dir=data/openneuro_ds005811_sub01 \
  --include="derivatives/preprocessed/epochs/sub-01_eeg_epo.fif" \
  --include="derivatives/detailed_events/sub-01_events.csv" \
  --max-concurrent-downloads=2 \
  --metadata-timeout=60
```

Then stage the files under the local layout expected by
`benchmarks/nod_animate_sub01.csv`:

```bash
mkdir -p data/nod
cp data/openneuro_ds005811_sub01/derivatives/preprocessed/epochs/sub-01_eeg_epo.fif \
  data/nod/sub-01_epo.fif
cp data/openneuro_ds005811_sub01/derivatives/detailed_events/sub-01_events.csv \
  data/nod/sub-01_events.csv
```

On Windows PowerShell, the staging commands are:

```powershell
New-Item -ItemType Directory -Force -Path data\nod
Copy-Item data\openneuro_ds005811_sub01\derivatives\preprocessed\epochs\sub-01_eeg_epo.fif `
  data\nod\sub-01_epo.fif
Copy-Item data\openneuro_ds005811_sub01\derivatives\detailed_events\sub-01_events.csv `
  data\nod\sub-01_events.csv
```

The expected staged file sizes for the sub-01 pilot are about 215 MB for the
epoch file and less than 1 MB for the events CSV.

## Alternative Download Methods

OpenNeuro's browser download, the Deno OpenNeuro CLI, or DataLad/git-annex can
also be used. The Deno CLI command downloads a dataset snapshot:

```bash
deno run -A jsr:@openneuro/cli download ds005811 data/openneuro_ds005811
```

After a full download, copy the same two files from the snapshot into
`data/nod/`.

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
