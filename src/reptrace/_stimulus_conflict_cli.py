"""Compatibility hook for historical stimulus conflict CLI integration."""

from __future__ import annotations


def install() -> None:
    """Keep old imports valid without changing the public API."""
    return None
