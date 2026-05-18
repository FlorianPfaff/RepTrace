from __future__ import annotations

import mne
import numpy as np
import pandas as pd

from reptrace import stage_things_eeg2_preprocessed as stage


def test_stage_subject_and_manifest_from_author_preprocessed_arrays(tmp_path):
    subject_dir = tmp_path / "eeg_dataset" / "preprocessed_data" / "sub-01"
    subject_dir.mkdir(parents=True)
    payload = {
        "preprocessed_eeg_data": np.random.default_rng(13).normal(size=(4, 3, 2, 20)).astype("float32"),
        "ch_names": ["O1", "O2"],
        "times": np.linspace(-0.1, 0.6, 20),
    }
    np.save(subject_dir / "preprocessed_eeg_test.npy", payload)
    label_map_path = tmp_path / "label_map.csv"
    pd.DataFrame(
        {
            "image_condition": [1, 2, 3, 4],
            "label": ["animate", "animate", "inanimate", "inanimate"],
        }
    ).to_csv(label_map_path, index=False)

    label_map = stage.load_label_map(label_map_path, key_column="image_condition", label_column="label")
    result = stage.stage_subject(
        things_root=tmp_path,
        staged_dir=tmp_path / "staged",
        subject="1",
        partition="test",
        label_map=label_map,
        label_column="condition",
        max_repetitions=2,
        overwrite=True,
    )

    metadata = pd.read_csv(result.metadata_path)
    epochs = mne.read_epochs(result.epochs_path, preload=False, verbose="error")
    assert result.subject == "sub-01"
    assert result.n_trials == 8
    assert result.n_conditions == 4
    assert len(epochs) == len(metadata) == 8
    assert metadata["condition"].value_counts().to_dict() == {"animate": 4, "inanimate": 4}

    manifest_path = tmp_path / "manifest.csv"
    stage.write_manifest(
        [result],
        manifest_path,
        decoders=["logistic", "linear_svm"],
        label_column="condition",
        group_column="image_condition",
        tmin=-0.1,
        tmax=0.6,
        window_ms=20,
        step_ms=10,
        n_splits=2,
    )
    manifest = pd.read_csv(manifest_path)
    assert manifest["decoder"].tolist() == ["logistic", "linear_svm"]
    assert manifest["group_column"].tolist() == ["image_condition", "image_condition"]
