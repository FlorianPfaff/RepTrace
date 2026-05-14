# Probability-observation schema

RepTrace workflows exchange held-out decoder outputs as probability-observation
tables. The schema validator checks those tables before downstream onset,
stimulus-event, or temporal-state analyses.

## Minimal table

A minimal generic table contains a numeric `time` column and at least one
`prob_class_*` column.

```csv
subject,sequence_id,time,true_class,predicted_class,prob_class_0,prob_class_1,class_0,class_1
sub-01,trial-001,-0.100,face,object,0.60,0.40,object,face
sub-01,trial-001,-0.080,face,face,0.45,0.55,object,face
```

Recommended metadata columns are:

- `subject` for subject-level aggregation;
- `decoder` for comparing model families;
- `emission_mode` for calibrated/uncalibrated comparisons;
- `backend`, `split_id`, `seed`, `train_time`, `test_time`,
  `preprocessing_hash`, and `model_hash` for canonical reproducibility
  provenance;
- `sequence_id` or `sample_index` for trial-like temporal sequences;
- `stream_id` for long continuous stimulus streams;
- `class_*` columns for human-readable class names matching `prob_class_*`
  suffixes.

## Validation profiles

The validator supports four profiles.

| Profile | Intended consumer | Extra structural requirement |
| --- | --- | --- |
| `generic` | summaries, plotting, custom analyses | none beyond `time` and `prob_class_*` |
| `canonical` | backend/workflow interoperability and reproducible benchmarks | canonical provenance columns and prediction/probability consistency |
| `temporal-model` | sticky temporal-state models | `sequence_id` or `sample_index` |
| `stimulus-detection` | long-stream event detection | `stream_id`, `sequence_id`, or `sample_index` |

Structural errors fail validation. Quality and reproducibility concerns, such as
missing recommended metadata columns or probability row sums outside tolerance,
are warnings unless `--require-normalized` is used.

The `canonical` profile is stricter: probability row sums are treated as
errors, required provenance columns must be present and non-empty, `time` must
match `test_time`, `confidence` must equal the maximum `prob_class_*` value,
`predicted_label` must equal the probability argmax, `predicted_class` must
match `class_<predicted_label>` when class-name columns are present, and
`probability_true_class` must match `prob_class_<true_label>` when true-label
columns are present. Continuous-stream canonical rows with a non-empty
`stream_id` must also contain `sample_index` and `sequence_id` values.

## CLI

Validate a generic probability-observation CSV:

```bash
reptrace validate-observations observations.csv
```

Validate a table before temporal modeling and write machine-readable reports:

```bash
reptrace validate-observations observations.csv \
  --profile temporal-model \
  --report-out results/observation_validation.csv \
  --summary-out results/observation_summary.csv
```

Validate a long-stream stimulus table with an explicit stream identifier:

```bash
reptrace validate-observations stream_observations.csv \
  --profile stimulus-detection \
  --stream-column stream_id \
  --report-out results/stream_observation_validation.csv
```

Validate canonical probability observations before using them in a benchmark or
leaderboard:

```bash
reptrace validate-observations observations.csv \
  --profile canonical \
  --report-out results/canonical_observation_validation.csv
```

Write a row-normalized copy when classifier exports contain unnormalized scores
that are otherwise suitable probability-like emissions:

```bash
reptrace validate-observations observations.csv \
  --normalize-out results/observations_normalized.csv
```

The exit code is `0` when no blocking validation errors are found, `1` when
validation errors are found, and `2` when the command itself cannot read or
process the inputs. Use `--warn-only` to keep a CI or exploratory script moving
while still writing the validation report.

## Python API

```python
import pandas as pd
from reptrace.observation_schema import (
    read_validated_probability_observations,
    summarize_probability_observations,
    validate_probability_observations,
)

observations = pd.read_csv("observations.csv")
report = validate_probability_observations(observations, profile="temporal-model")
report.raise_for_errors()

summary = summarize_probability_observations(observations)
```

For reader-compatible defaults, use:

```python
observations = read_validated_probability_observations(
    ["observations.csv"],
    profile="temporal-model",
)
```
