"""
parse_bsdd_to_map.py

Parse a bSDD / IFC-OWL RDF/TTL snapshot and generate an offline ontology-map
JSON file.  The map records:

  IFC class entries:  { "BaseClasses": [...], "ValidPsets": [...] }
  Pset entries:       { "PropertyName": "IfcTypeName", ... }

If the RDF file is missing, unparseable, or rdflib is unavailable the module
falls back gracefully to the embedded STANDARD_PSETS dict so downstream
tooling is never blocked.

Usage (standalone):
    uv run python -m rag_tag.parser.parse_bsdd_to_map \\
        --rdf-path output/metadata/bsdd/ifc43.ttl \\
        --out src/rag_tag/parser/ifc_ontology_map.json

Public API:
    build_ontology_map(rdf_path)  ->  dict
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_OUT: Path = Path(__file__).resolve().parent / "ifc_ontology_map.json"

# RDF URI prefixes that indicate an IFC schema entity vs an OWL bookkeeping node.
_IFC_PREFIXES = ("Ifc", "Pset_", "Qto_")


# ---------------------------------------------------------------------------
# Low-level RDF helpers
# ---------------------------------------------------------------------------


def _local_name(node) -> str:
    """
    Extract the short token from an RDF URI, BNode, or Literal.

    URIs:    .../IfcWall  or  ...#IfcWall  -> "IfcWall"
    BNodes:  blank nodes have no useful name  -> ""
    Literals: return the string value as-is.
    """
    try:
        import rdflib  # noqa: PLC0415

        if isinstance(node, rdflib.BNode):
            return ""
        if isinstance(node, rdflib.Literal):
            return str(node)
    except ImportError:
        pass
    s = str(node)
    if "#" in s:
        return s.rsplit("#", 1)[-1]
    if "/" in s:
        return s.rsplit("/", 1)[-1]
    return s


def _is_ifc_token(name: str) -> bool:
    return any(name.startswith(p) for p in _IFC_PREFIXES)


# ---------------------------------------------------------------------------
# Hierarchy expansion
# ---------------------------------------------------------------------------


def _expand_ancestors(
    direct_parents: dict[str, list[str]],
) -> dict[str, list[str]]:
    """
    Expand direct-parent maps into full ordered ancestor chains.

    The chain for a class is ordered nearest-first (direct parent first,
    then grandparent, etc.).  Cycles are silently broken.
    """
    cache: dict[str, list[str]] = {}

    def _walk(cls: str, visiting: frozenset[str]) -> list[str]:
        if cls in cache:
            return cache[cls]
        if cls in visiting:
            return []
        visiting = visiting | {cls}
        chain: list[str] = []
        for parent in direct_parents.get(cls, []):
            if parent not in chain:
                chain.append(parent)
            for anc in _walk(parent, visiting):
                if anc not in chain:
                    chain.append(anc)
        cache[cls] = chain
        return chain

    return {cls: _walk(cls, frozenset()) for cls in direct_parents}


# ---------------------------------------------------------------------------
# RDF parsing
# ---------------------------------------------------------------------------


def _parse_hierarchy_from_rdf(rdf_path: Path) -> dict[str, list[str]] | None:
    """
    Load a bSDD / IFC-OWL Turtle file and return a full ancestor-chain map.

    Reads only ``rdfs:subClassOf`` triples where both ends look like IFC
    tokens (start with Ifc, Pset_, or Qto_).  Returns None if rdflib is
    unavailable or the file cannot be parsed.
    """
    try:
        import rdflib  # noqa: PLC0415
        import rdflib.namespace as rdfns  # noqa: PLC0415
    except ImportError:
        logger.warning(
            "rdflib not installed; cannot parse RDF. "
            "Run `uv add rdflib` to enable RDF augmentation."
        )
        return None

    try:
        g = rdflib.Graph()
        g.parse(str(rdf_path))
    except Exception as exc:
        logger.warning(
            "Failed to parse RDF at %s: %s. "
            "Check that the file is a valid Turtle/RDF document.",
            rdf_path,
            exc,
        )
        return None

    RDFS = rdfns.RDFS
    direct_parents: dict[str, list[str]] = {}

    for cls_node, _, parent_node in g.triples((None, RDFS.subClassOf, None)):
        cls_name = _local_name(cls_node)
        parent_name = _local_name(parent_node)
        if not cls_name or not parent_name:
            continue
        if not (_is_ifc_token(cls_name) and _is_ifc_token(parent_name)):
            continue
        parents = direct_parents.setdefault(cls_name, [])
        if parent_name not in parents:
            parents.append(parent_name)

    if not direct_parents:
        logger.warning(
            "RDF at %s yielded no usable subClassOf triples. "
            "Schema may use an unsupported format or namespace.",
            rdf_path,
        )
        return None

    logger.info(
        "Parsed RDF hierarchy from %s: %d triples -> %d classes indexed",
        rdf_path,
        len(g),
        len(direct_parents),
    )
    return _expand_ancestors(direct_parents)


# ---------------------------------------------------------------------------
# Map builder
# ---------------------------------------------------------------------------


def build_ontology_map_from_registry(
    rdf_ancestors: dict[str, list[str]] | None = None,
) -> dict:
    """
    Build the ontology map from the embedded STANDARD_PSETS registry.

    If *rdf_ancestors* is supplied, it overrides the ifcopenshell-derived
    ancestor chains for classes that appear in the RDF (richer coverage).
    Classes not found in the RDF fall back to the ifcopenshell chain.

    Returns a dict with two entry shapes:

      IFC class entries:
          { "BaseClasses": [<ancestor>, ...], "ValidPsets": [<pset_name>, ...] }

      Pset entries:
          { <property_name>: "IfcLabel", ... }
    """
    from rag_tag.parser.ifc43_schema_registry import (  # noqa: PLC0415
        STANDARD_PSETS,
        get_registry,
    )

    registry = get_registry()
    result: dict = {}

    # ---- IFC class entries ----
    for cls, psets in STANDARD_PSETS.items():
        if rdf_ancestors is not None and cls in rdf_ancestors:
            ancestors: list[str] = rdf_ancestors[cls]
        else:
            norm = registry.normalize_class(cls)
            ancestors = norm.get("ancestors", [])

        result[cls] = {
            "BaseClasses": ancestors,
            "ValidPsets": list(psets.keys()),
        }

    # ---- Pset entries ----
    # Property type strings are not yet encoded in the RDF so we use
    # "IfcLabel" as a well-known placeholder consistent with the IFC spec.
    for _cls, psets in STANDARD_PSETS.items():
        for pset_name, prop_names in psets.items():
            if pset_name not in result:
                result[pset_name] = {p: "IfcLabel" for p in prop_names}

    return result


def build_ontology_map(rdf_path: Path | None = None) -> dict:
    """
    Build the IFC ontology map, optionally augmented by an RDF snapshot.

    Resolution order:
      1. If *rdf_path* is given and the file exists, parse it for hierarchy.
      2. If *rdf_path* is None, check the default bSDD path (find_bsdd_rdf_path).
      3. On any failure (missing file, parse error, no rdflib) fall back to the
         embedded STANDARD_PSETS registry.  Nothing crashes.

    Args:
        rdf_path: Path to a bSDD / IFC-OWL Turtle file, or None.

    Returns:
        Dict conforming to the schema described in build_ontology_map_from_registry.
    """
    rdf_ancestors: dict[str, list[str]] | None = None
    resolved_rdf: Path | None = rdf_path

    if resolved_rdf is None:
        try:
            from rag_tag.paths import find_bsdd_rdf_path  # noqa: PLC0415

            candidate = find_bsdd_rdf_path()
            if candidate.exists():
                resolved_rdf = candidate
                logger.info("Auto-detected bSDD RDF at %s", resolved_rdf)
        except Exception:
            pass

    if resolved_rdf is not None:
        if not resolved_rdf.exists():
            logger.warning(
                "RDF file not found: %s. "
                "Run `uv run rag-tag-refresh-ifc43-rdf` to download it. "
                "Using embedded registry.",
                resolved_rdf,
            )
        else:
            rdf_ancestors = _parse_hierarchy_from_rdf(resolved_rdf)

    if rdf_ancestors is None:
        logger.info("Building ontology map from embedded STANDARD_PSETS registry.")
    else:
        logger.info(
            "Building ontology map using RDF hierarchy (%d classes).",
            len(rdf_ancestors),
        )

    return build_ontology_map_from_registry(rdf_ancestors)


# ---------------------------------------------------------------------------
# CLI entry point (standalone use)
# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    try:
        from rag_tag.paths import find_bsdd_rdf_path, find_project_root  # noqa: PLC0415

        default_rdf: Path | None = find_bsdd_rdf_path()
        project_root = find_project_root() or Path.cwd()
    except Exception:
        default_rdf = None
        project_root = Path.cwd()

    default_out = project_root / "src" / "rag_tag" / "parser" / "ifc_ontology_map.json"

    ap = argparse.ArgumentParser(
        description=(
            "Parse a bSDD / IFC-OWL RDF/TTL snapshot and write an offline "
            "ontology map JSON file.  Falls back to embedded STANDARD_PSETS "
            "if the RDF file is unavailable."
        ),
    )
    ap.add_argument(
        "--rdf-path",
        type=Path,
        default=None,
        help=(
            "bSDD / IFC-OWL Turtle file to parse.  "
            f"Auto-detected default: {default_rdf}"
        ),
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=default_out,
        help=f"Output JSON path (default: {default_out}).",
    )
    args = ap.parse_args()

    rdf_path = args.rdf_path.expanduser().resolve() if args.rdf_path else None
    out_path = args.out.expanduser().resolve()

    ontology_map = build_ontology_map(rdf_path)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(ontology_map, indent=2), encoding="utf-8")
    logger.info(
        "Ontology map written to %s (%d entries).",
        out_path,
        len(ontology_map),
    )
    print(f"\nOntology map ready at: {out_path}")


if __name__ == "__main__":
    main()
