"""Download and pin a local bSDD IFC 4.3 RDF snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
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


if __name__ == "__main__":
    main()
