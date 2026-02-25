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


def find_bsdd_rdf_path(start_dir: Path | None = None) -> Path:
    """
    Returns the default path where we store the downloaded bSDD RDF snapshot.

    We put it in output/metadata/bsdd/ifc43.ttl — next to the CSVs and
    databases we generate, but in its own folder so it doesn't get mixed up.

    The file won't exist until you run the refresh script for the first time.
    That's okay — the schema registry handles a missing file without crashing.
    """
    root = find_project_root(start_dir)
    if root is None:
        root = Path.cwd()
    return root / "output" / "metadata" / "bsdd" / "ifc43.ttl"
