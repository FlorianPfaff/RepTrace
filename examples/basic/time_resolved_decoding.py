from pathlib import Path

import mne
import numpy as np
import pandas as pd

from reptrace.mne_time_decode import run_time_resolved_decode


def main() -> None:
    rng = np.random.default_rng(13)
    n_epochs = 40
    n_channels = 4
    n_times = 50
    sfreq = 100.0

    labels = np.array(["object"] * (n_epochs // 2) + ["face"] * (n_epochs // 2))
    data = rng.normal(scale=0.2, size=(n_epochs, n_channels, n_times))
    data[labels == "face", 0, 20:30] += 1.0

    info = mne.create_info(
        ch_names=[f"MEG{i:03d}" for i in range(n_channels)],
        sfreq=sfreq,
        ch_types="mag",
    )
    metadata = pd.DataFrame(
        {
            "condition": labels,
            "run": np.repeat(np.arange(4), n_epochs // 4),
        }
    )
    epochs = mne.EpochsArray(data, info, tmin=-0.1, metadata=metadata, verbose="error")

    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    epochs_path = output_dir / "synthetic-epo.fif"
    out_path = output_dir / "synthetic_decoding.csv"
    epochs.save(epochs_path, overwrite=True, verbose="error")

    run_time_resolved_decode(
        epochs_path=epochs_path,
        label_column="condition",
        group_column="run",
        out_path=out_path,
        window_ms=50.0,
        step_ms=20.0,
        n_splits=4,
    )
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
