from __future__ import annotations

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-performance",
        action="store_true",
        default=False,
        help="Run performance benchmark tests.",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption("--run-performance"):
        return

    skip_performance = pytest.mark.skip(reason="need --run-performance to run performance benchmarks")
    for item in items:
        if "performance" in item.keywords:
            item.add_marker(skip_performance)
