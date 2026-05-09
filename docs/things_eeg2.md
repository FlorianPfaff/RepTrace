# THINGS-EEG2 comparison workflow

RepTrace can run its calibration-first decoder comparison on THINGS-EEG2 after
the author-preprocessed arrays have been staged locally.  The workflow does not
download THINGS-EEG2 automatically because the dataset is large; use the THINGS,
OpenNeuro/NeMAR, OSF, or local lab mirror route to stage the data first.

## Expected staged data

The workflow expects the author-preprocessed THINGS-EEG2 files for each subject,
for example one of these layouts:

```text
<data_root>/eeg_dataset/preprocessed_data/sub-01/preprocessed_eeg_test.npy
<data_root>/eeg_dataset/preprocessed_data/sub-01/preprocessed_eeg_training.npy
```

It also accepts these equivalent subject roots:

```text
<data_root>/preprocessed_data/sub-01/
<data_root>/Preprocessed_data/sub-01/
<data_root>/derivatives/preprocessed_eeg/sub-01/
<data_root>/sub-01/
```

Each numpy file must contain the original dictionary keys
`preprocessed_eeg_data`, `ch_names`, and `times`.

## Label map

RepTrace's current benchmark runner evaluates supervised decoding labels.  For a
NOD-style semantic comparison, provide a CSV mapping THINGS-EEG2 image-condition
IDs to the labels you want to decode.  For a binary animate/inanimate task:

```csv
image_condition,label
1,animate
2,inanimate
3,animate
4,inanimate
```

The workflow default uses:

```text
label_map_key_column=image_condition
label_map_label_column=label
label_column=condition
group_column=image_condition
chance=0.5
```

Grouping by `image_condition` keeps repetitions of the same image out of both
train and test folds for semantic labels.  Do not use this grouped protocol for
exact image-condition classification unless the label definition has more than
one image condition per class; otherwise the grouped split holds out entire
classes.

## Manual workflow run

```bash
gh workflow run things-eeg2-comparison.yml \
  --repo IPS-Stuttgart/RepTrace \
  --ref main \
  -f data_root=../data/things_eeg2 \
  -f label_map_csv=../data/things_eeg2/things_eeg2_label_map.csv \
  -f output_dir=results/things_eeg2_animacy \
  -f target_labels=animate,inanimate \
  -f chance=0.5
```

For a smoke test, cap the number of conditions and repetitions:

```bash
gh workflow run things-eeg2-comparison.yml \
  --repo IPS-Stuttgart/RepTrace \
  --ref main \
  -f data_root=../data/things_eeg2 \
  -f label_map_csv=../data/things_eeg2/things_eeg2_label_map.csv \
  -f subjects="1 2" \
  -f max_conditions_per_label=10 \
  -f max_repetitions=10 \
  -f output_dir=results/things_eeg2_smoke
```

## Outputs

The workflow stages each subject into an MNE Epochs FIF file and metadata CSV,
writes a RepTrace manifest, then runs the existing benchmark, calibration,
paired-statistics, and inference reports.  Compact uploaded artifacts include
`summary.csv`, `summary.png`, reliability outputs, paired statistics, inference
CSV files, and the staged metadata CSVs.  Large staged FIF files are not uploaded.
