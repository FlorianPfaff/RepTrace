"""Backward-compatible wrapper for the installed THINGS-EEG2 staging command."""

from __future__ import annotations

from reptrace.stage_things_eeg2_preprocessed import (  # noqa: F401
    DEFAULT_DECODERS,
    PARTITION_ALIASES,
    SubjectStageResult,
    find_preprocessed_file,
    load_label_map,
    main,
    stage_subject,
    write_manifest,
)

if __name__ == "__main__":
    main()
