"""
generate_ontology_map.py

Run once to export the IFC class hierarchy and known Pset definitions
from ifc43_schema_registry.py into a lightweight JSON file.

The JSON is written to src/rag_tag/parser/ifc_ontology_map.json and can be
inspected or modified by developers. ifc_to_jsonl.py uses the registry
directly at runtime (not this file), but the JSON is useful for debugging.

Usage:
  uv run rag-tag-generate-ontology-map
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

_SCRIPT_DIR = Path(__file__).resolve().parent
_DEFAULT_OUT = (
    _SCRIPT_DIR.parent
    / "src"
    / "rag_tag"
    / "parser"
    / "ifc_ontology_map.json"
)


def build_ontology_map() -> dict:
    """Export STANDARD_PSETS + hierarchy from the registry to a plain dict."""
    from rag_tag.parser.ifc43_schema_registry import (  # noqa: PLC0415
        STANDARD_PSETS,
        get_registry,
    )

    registry = get_registry()
    result: dict = {}

    # For each known class record its ancestors and valid pset names
    for cls, psets in STANDARD_PSETS.items():
        norm = registry.normalize_class(cls)
        result[cls] = {
            "BaseClasses": norm.get("ancestors", []),
            "ValidPsets": list(psets.keys()),
        }

    # For each pset record its property names (type info not available here,
    # so we use "IfcLabel" as a placeholder for all — the real types are in
    # the IFC spec; this is just for reference)
    for _cls, psets in STANDARD_PSETS.items():
        for pset_name, prop_names in psets.items():
            if pset_name not in result:
                result[pset_name] = {p: "IfcLabel" for p in prop_names}

    return result


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Export IFC ontology map from registry to JSON."
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=_DEFAULT_OUT,
        help=f"Output path (default: {_DEFAULT_OUT})",
    )
    args = ap.parse_args()

    out_path = args.out.expanduser().resolve()
    logger.info("Building ontology map from registry...")

    ontology_map = build_ontology_map()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(ontology_map, indent=2), encoding="utf-8")
    logger.info(
        "Ontology map written to %s (%d entries)", out_path, len(ontology_map)
    )
    print(f"\nOntology map ready at: {out_path}")


if __name__ == "__main__":
    main()
