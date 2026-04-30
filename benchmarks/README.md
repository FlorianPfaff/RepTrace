# Benchmark Manifests

Benchmark manifests describe one row per subject or session. Paths are resolved
relative to the manifest file.

Required columns:

- `subject`: subject or session identifier used in outputs;
- `epochs`: path to an MNE epochs `.fif` file;
- `label_column`: decoding target column in the metadata.

Use either `metadata_csv` for pre-labeled metadata or `events_csv`,
`source_column`, and `positive_pattern` to derive a binary label before
decoding.
