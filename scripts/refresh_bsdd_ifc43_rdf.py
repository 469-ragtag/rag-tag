<<<<<<< HEAD
"""Download and pin a local bSDD IFC 4.3 RDF snapshot."""
=======
"""
refresh_bsdd_ifc43_rdf.py

Downloads a fresh copy of the bSDD / IFC-OWL RDF snapshot and saves it
to a local path so the schema registry can use it offline.

Why we need this
----------------
The schema registry (ifc43_schema_registry.py) has a built-in property
dictionary, but it can also read a proper OWL/RDF file to get the full
class hierarchy directly from the IFC standard.  That file lives online
so we need to pull it down once and pin it locally.

We also write a small sidecar JSON file next to the snapshot that records:
  - the URL we downloaded from
  - the timestamp of the download
  - the SHA256 hash of the file

This means we can always tell exactly which version of the schema we're
running against, and roll back to an older snapshot if something breaks.

Usage
-----
# using the entry point (after `uv pip install -e .`)
uv run rag-tag-refresh-ifc43-rdf --url <url> --out output/metadata/bsdd/ifc43.ttl

# or run the script directly
uv run python scripts/refresh_bsdd_ifc43_rdf.py --url <url>

Default URL is the IFC 4.3 OWL Turtle file from buildingSMART's GitHub.
Default output path is output/metadata/bsdd/ifc43.ttl (project root).
"""
>>>>>>> f8c7778f6548519e531b8d6d8c2dadcacd2beb62

from __future__ import annotations

import argparse
import hashlib
import json
<<<<<<< HEAD
import os
from datetime import UTC, datetime
from pathlib import Path
from urllib.request import urlopen

from rag_tag.paths import default_bsdd_ifc43_rdf_path, find_project_root


def _sha256(data: bytes) -> str:
    digest = hashlib.sha256()
    digest.update(data)
    return digest.hexdigest()


def refresh_snapshot(url: str, output_path: Path) -> tuple[Path, Path]:
    with urlopen(url, timeout=60) as response:
        data = response.read()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(data)

    metadata = {
        "source_url": url,
        "downloaded_at_utc": datetime.now(UTC).isoformat(),
        "sha256": _sha256(data),
        "size_bytes": len(data),
        "filename": output_path.name,
        "format_hint": output_path.suffix.lower().lstrip("."),
    }

    metadata_path = output_path.with_suffix(output_path.suffix + ".meta.json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return output_path, metadata_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and pin a local bSDD IFC 4.3 RDF snapshot."
    )
    parser.add_argument(
        "--url",
        type=str,
        default=os.getenv("BSDD_IFC43_RDF_URL"),
        help=(
            "RDF snapshot URL. Can also be provided via BSDD_IFC43_RDF_URL env var."
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help=(
            "Output file path for pinned snapshot "
            "(default: <project-root>/output/metadata/bsdd/ifc43.ttl)."
        ),
    )
    args = parser.parse_args()

    if not args.url:
        raise SystemExit(
            "Missing snapshot URL. Provide --url or set BSDD_IFC43_RDF_URL."
        )

    start_dir = Path(__file__).resolve().parent
    project_root = find_project_root(start_dir) or start_dir
    output_path = (args.out or default_bsdd_ifc43_rdf_path(project_root)).resolve()

    snapshot_path, metadata_path = refresh_snapshot(args.url, output_path)
    print(f"Snapshot written: {snapshot_path}")
    print(f"Metadata written: {metadata_path}")
=======
import logging
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# The canonical source for the IFC 4.3 OWL ontology (Turtle format).
# This is buildingSMART's official GitHub repo for the IFC 4.3 development.
DEFAULT_URL = (
    "https://raw.githubusercontent.com/buildingSMART/"
    "IFC4.3.x-development/master/docs/ifcOwlDocs/IFC4X3.ttl"
)

# Where to save the file by default — relative to the project root.
# We resolve the actual path at runtime using find_bsdd_rdf_path().
DEFAULT_OUT = "output/metadata/bsdd/ifc43.ttl"


def compute_sha256(file_path: Path) -> str:
    """Read a file in chunks and return its SHA256 hex digest."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, out_path: Path) -> None:
    """
    Download the RDF snapshot from url and save it to out_path.
    Also writes a sidecar .json file with metadata about the download.

    The sidecar file sits right next to the .ttl file with the same stem:
        ifc43.ttl      ← the actual schema data
        ifc43.json     ← metadata: url, timestamp, sha256, format
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading from:\n  %s", url)
    logger.info("Saving to:\n  %s", out_path)

    # stream the download — the OWL file can be a few MB
    try:
        with urllib.request.urlopen(url) as response:  # noqa: S310
            content = response.read()
    except Exception as exc:
        logger.error("Download failed: %s", exc)
        sys.exit(1)

    out_path.write_bytes(content)
    logger.info("Downloaded %d bytes", len(content))

    # compute SHA256 so we can verify the file later
    sha256 = compute_sha256(out_path)
    logger.info("SHA256: %s", sha256)

    # write the sidecar metadata JSON
    sidecar_path = out_path.with_suffix(".json")
    metadata = {
        "url": url,
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "sha256": sha256,
        "format": "text/turtle",
        "file": out_path.name,
    }
    sidecar_path.write_text(json.dumps(metadata, indent=2))
    logger.info("Metadata written to: %s", sidecar_path)

    print(f"\nSnapshot ready at: {out_path}")
    print(f"SHA256: {sha256}")
    print(
        "\nTo use it, run the parser with:\n"
        f"  uv run rag-tag-ifc-to-csv --bsdd-rdf-path {out_path}"
    )


def main() -> None:
    # we import here so the script works even if rag_tag isn't installed
    # (the entry point will have it installed, but someone might run this directly)
    try:
        from rag_tag.paths import find_bsdd_rdf_path  # noqa: PLC0415
        default_out = find_bsdd_rdf_path()
    except ImportError:
        default_out = Path(DEFAULT_OUT)

    ap = argparse.ArgumentParser(
        description="Download and pin a local bSDD IFC 4.3 RDF snapshot.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  # download using default URL and path
  uv run rag-tag-refresh-ifc43-rdf

  # custom URL
  uv run rag-tag-refresh-ifc43-rdf --url https://example.com/ifc4x3.ttl

  # custom output path
  uv run rag-tag-refresh-ifc43-rdf --out /tmp/ifc43.ttl
        """,
    )
    ap.add_argument(
        "--url",
        default=DEFAULT_URL,
        help=f"URL to download the RDF snapshot from (default: {DEFAULT_URL})",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=default_out,
        help=f"Where to save the snapshot (default: {default_out})",
    )
    args = ap.parse_args()

    download(url=args.url, out_path=args.out.expanduser().resolve())
>>>>>>> f8c7778f6548519e531b8d6d8c2dadcacd2beb62


if __name__ == "__main__":
    main()
