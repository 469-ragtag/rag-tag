"""Download and pin a local bSDD/IFC 4.3 RDF snapshot.

The command saves the Turtle file and writes a sidecar JSON file with
download source, timestamp, and SHA256 for reproducibility.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from rag_tag.paths import find_bsdd_rdf_path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Canonical IFC 4.3 OWL Turtle source from buildingSMART's repository.
DEFAULT_URL = (
    "https://raw.githubusercontent.com/buildingSMART/"
    "IFC4.3.x-development/master/docs/ifcOwlDocs/IFC4X3.ttl"
)


def compute_sha256(file_path: Path) -> str:
    """Read a file in chunks and return its SHA256 hex digest."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, out_path: Path) -> None:
    """Download the RDF snapshot and write a sidecar metadata JSON file."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading from:\n  %s", url)
    logger.info("Saving to:\n  %s", out_path)

    try:
        with urllib.request.urlopen(url) as response:  # noqa: S310
            content = response.read()
    except Exception as exc:
        logger.error("Download failed: %s", exc)
        sys.exit(1)

    out_path.write_bytes(content)
    logger.info("Downloaded %d bytes", len(content))

    sha256 = compute_sha256(out_path)
    logger.info("SHA256: %s", sha256)

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
        "\nTo use it in the JSONL pipeline, run:\n"
        f"  uv run rag-tag-generate-ontology-map --rdf {out_path}\n"
        "  uv run rag-tag-ifc-to-jsonl"
    )


def main() -> None:
    """Parse CLI args and refresh the local RDF snapshot."""
    default_out = find_bsdd_rdf_path()

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


if __name__ == "__main__":
    main()
