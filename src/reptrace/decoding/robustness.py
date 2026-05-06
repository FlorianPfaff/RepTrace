from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class RobustnessCondition:
    """Named analysis variant for robustness or control sweeps."""

    name: str
    label: str
    parameters: Mapping[str, object] = field(default_factory=dict)


RowCollection = Sequence[Mapping[str, object]] | pd.DataFrame | None
ArtifactRows = Mapping[str, RowCollection]


def annotate_condition_rows(
    rows: RowCollection,
    condition: RobustnessCondition,
    *,
    control_column: str = "control",
    label_column: str = "control_label",
) -> list[dict[str, object]]:
    """Attach robustness condition metadata to row dictionaries.

    Dataset-specific projects can emit plain result rows while RepTrace keeps
    condition labeling consistent across accuracy, prediction, and summary
    artifacts.
    """

    annotated: list[dict[str, object]] = []
    for row in _row_records(rows):
        record = dict(row)
        record.pop(control_column, None)
        record.pop(label_column, None)
        annotated.append(
            {
                control_column: condition.name,
                label_column: condition.label,
                **record,
            }
        )
    return annotated


def run_robustness_conditions(
    conditions: Sequence[RobustnessCondition],
    runner: Callable[[RobustnessCondition], ArtifactRows],
    *,
    progress: Callable[[str], Any] | None = None,
    control_column: str = "control",
    label_column: str = "control_label",
) -> dict[str, list[dict[str, object]]]:
    """Run condition-level robustness controls and concatenate named artifacts."""

    _validate_conditions(conditions)
    artifacts: dict[str, list[dict[str, object]]] = {}
    for condition in conditions:
        _emit_progress(progress, f"START control={condition.name}")
        condition_artifacts = runner(condition)
        _extend_artifacts(
            artifacts,
            condition_artifacts,
            condition,
            control_column=control_column,
            label_column=label_column,
        )
        _emit_progress(progress, f"DONE control={condition.name}")
    return artifacts


def run_participant_robustness_conditions(
    conditions: Sequence[RobustnessCondition],
    participants: Sequence[object],
    runner: Callable[[RobustnessCondition, object], ArtifactRows],
    *,
    progress: Callable[[str], Any] | None = None,
    control_column: str = "control",
    label_column: str = "control_label",
) -> dict[str, list[dict[str, object]]]:
    """Run robustness controls for each participant and concatenate artifacts."""

    _validate_conditions(conditions)
    artifacts: dict[str, list[dict[str, object]]] = {}
    for condition in conditions:
        _emit_progress(progress, f"START control={condition.name}")
        for participant in participants:
            _emit_progress(progress, f"START control={condition.name} participant={participant}")
            participant_artifacts = runner(condition, participant)
            _extend_artifacts(
                artifacts,
                participant_artifacts,
                condition,
                control_column=control_column,
                label_column=label_column,
            )
            _emit_progress(progress, f"DONE control={condition.name} participant={participant}")
        _emit_progress(progress, f"DONE control={condition.name}")
    return artifacts


def _extend_artifacts(
    artifacts: dict[str, list[dict[str, object]]],
    condition_artifacts: ArtifactRows,
    condition: RobustnessCondition,
    *,
    control_column: str,
    label_column: str,
) -> None:
    for artifact_name, rows in condition_artifacts.items():
        artifacts.setdefault(artifact_name, []).extend(
            annotate_condition_rows(
                rows,
                condition,
                control_column=control_column,
                label_column=label_column,
            )
        )


def _row_records(rows: RowCollection) -> list[dict[str, object]]:
    if rows is None:
        return []
    if isinstance(rows, pd.DataFrame):
        return rows.to_dict(orient="records")
    return [dict(row) for row in rows]


def _validate_conditions(conditions: Sequence[RobustnessCondition]) -> None:
    if not conditions:
        raise ValueError("Need at least one robustness condition.")
    names = [condition.name for condition in conditions]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        joined = ", ".join(duplicates)
        raise ValueError(f"Robustness condition names must be unique; duplicates: {joined}.")


def _emit_progress(progress: Callable[[str], Any] | None, message: str) -> None:
    if progress is not None:
        progress(message)
