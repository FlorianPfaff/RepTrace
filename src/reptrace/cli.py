"""Grouped command-line interface for RepTrace workflows."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from importlib import import_module

COMMAND_MODULES = {
    "benchmark": "reptrace.benchmark",
    "metadata": "reptrace.metadata",
    "mne-time-decode": "reptrace.mne_time_decode",
    "onset-detect": "reptrace.onset_detection",
    "onset-detection": "reptrace.onset_detection",
    "plot-time-decode": "reptrace.plot_time_decode",
    "results": "reptrace.results",
    "stimulus-detect": "reptrace.stimulus_detection",
    "stimulus-detection": "reptrace.stimulus_detection",
    "temporal-model": "reptrace.temporal_model",
    "temporal-state-workflow": "reptrace.temporal_state_workflow",
    "validate-manifest": "reptrace.validate_manifest",
}


def _run_module_main(command: str, argv: Sequence[str]) -> int:
    """Run a RepTrace module-level ``main`` as a grouped subcommand."""
    module = import_module(COMMAND_MODULES[command])
    module_main = getattr(module, "main", None)
    if module_main is None:
        raise RuntimeError(f"Command '{command}' is backed by {module.__name__}, which has no main() function.")

    original_argv = sys.argv
    sys.argv = [f"reptrace {command}", *argv]
    try:
        result = module_main()
    finally:
        sys.argv = original_argv

    return int(result) if isinstance(result, int) else 0


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch installed RepTrace subcommands."""
    if argv is None:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(description="RepTrace command-line interface.")
    parser.add_argument(
        "command",
        nargs="?",
        choices=sorted(COMMAND_MODULES),
        help="Workflow to run. Pass '<command> --help' for command-specific options.",
    )
    args, remaining = parser.parse_known_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    return _run_module_main(args.command, remaining)


if __name__ == "__main__":
    raise SystemExit(main())
