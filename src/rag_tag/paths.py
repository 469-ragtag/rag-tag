"""Centralized path discovery for the rag-tag package."""

from __future__ import annotations

from pathlib import Path


def find_project_root(start_dir: Path | None = None) -> Path | None:
    """Find the project root by searching upwards for pyproject.toml."""
    if start_dir is None:
        start_dir = Path.cwd()
    for base in (start_dir, *start_dir.parents):
        if (base / "pyproject.toml").is_file():
            return base
    return None


def find_ifc_dir(start_dir: Path | None = None) -> Path | None:
    """Find an IFC-Files directory by searching upwards from start_dir."""
    if start_dir is None:
        start_dir = Path.cwd()
    for base in (start_dir, *start_dir.parents):
        candidate = base / "IFC-Files"
        if candidate.is_dir():
            return candidate
    return None
