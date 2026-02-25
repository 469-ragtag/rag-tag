"""
generate_ontology_map.py

Run once to export the IFC class hierarchy and known Pset definitions into
a lightweight JSON file.

Delegates to parse_bsdd_to_map.build_ontology_map() which:
  - Uses a bSDD / IFC-OWL RDF/TTL snapshot (--rdf) for richer hierarchy
  - Auto-detects output/metadata/bsdd/ifc43.ttl when --rdf is omitted
  - Falls back silently to embedded STANDARD_PSETS on any error

The JSON is written to src/rag_tag/parser/ifc_ontology_map.json by default.

Usage:
  uv run rag-tag-generate-ontology-map
  uv run rag-tag-generate-ontology-map --rdf output/metadata/bsdd/ifc43.ttl
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from rag_tag.paths import find_bsdd_rdf_path, find_project_root

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _default_out_path() -> Path:
    """Resolve the default output path relative to the project root."""
    root = find_project_root() or Path.cwd()
    return root / "src" / "rag_tag" / "parser" / "ifc_ontology_map.json"


def main() -> None:
    default_out = _default_out_path()
    default_rdf = find_bsdd_rdf_path()

    ap = argparse.ArgumentParser(
        description=(
            "Export IFC ontology map to JSON.  "
            "Augments the embedded registry with bSDD RDF hierarchy when "
            "a Turtle snapshot is available."
        ),
    )
    ap.add_argument(
        "--rdf",
        type=Path,
        default=None,
        dest="rdf_path",
        help=(
            "Path to a bSDD / IFC-OWL Turtle (.ttl) file.  "
            f"Auto-detected default: {default_rdf} (used if it exists)."
        ),
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=default_out,
        help=f"Output path (default: {default_out})",
    )
    args = ap.parse_args()

    from rag_tag.parser.parse_bsdd_to_map import (  # noqa: PLC0415
        build_ontology_map,
    )

    rdf_path = args.rdf_path.expanduser().resolve() if args.rdf_path else None
    out_path = args.out.expanduser().resolve()

    logger.info("Building ontology map...")
    ontology_map = build_ontology_map(rdf_path)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(ontology_map, indent=2), encoding="utf-8")
    logger.info("Ontology map written to %s (%d entries)", out_path, len(ontology_map))
    print(f"\nOntology map ready at: {out_path}")


if __name__ == "__main__":
    main()
