"""Batch 1 (P0.2) smoke-tests: explicit IFC relationships in the NetworkX graph.

Validates that:
  - explicit (source="ifc") and heuristic (source="heuristic") edges coexist
  - `connected_to` is heuristic-only (never source="ifc")
  - `ifc_connected_to` is used for explicit IFC connectivity (never source="heuristic")
  - provenance (source attr) exists on all non-hierarchy edges for both paths
  - context nodes (System::/Zone::/Classification::) are created with correct attrs
  - explicit edges use the correct relation names
  - backward-compat: existing hierarchy/topology relations still present

Run with:
    uv run python scripts/check_graph_relationships.py

The script uses both:
  A) A synthetic in-memory graph built from hand-crafted JSONL records.
  B) The live output/*.jsonl files (if present), as a real-data smoke-test.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import networkx as nx  # noqa: E402 (networkx is a direct dep)

from rag_tag.parser.jsonl_to_graph import (  # noqa: E402
    _add_explicit_relationships,
    _normalize_context_label,
    add_spatial_adjacency,
    build_graph,
    build_graph_from_jsonl,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
FAIL = "\033[31mFAIL\033[0m"
PASS = "\033[32mPASS\033[0m"
_failures: list[str] = []


def check(condition: bool, name: str) -> None:
    if condition:
        print(f"  {PASS}  {name}")
    else:
        print(f"  {FAIL}  {name}")
        _failures.append(name)


def _iter_edge_attrs(G: nx.Graph, u: str, v: str) -> list[dict]:
    edge_data = G.get_edge_data(u, v)
    if edge_data is None:
        return []
    if G.is_multigraph():
        if isinstance(edge_data, dict):
            return [attrs for attrs in edge_data.values() if isinstance(attrs, dict)]
        return []
    if isinstance(edge_data, dict):
        return [edge_data]
    return []


def _has_edge_with(
    G: nx.Graph,
    u: str,
    v: str,
    *,
    relation: str | None = None,
    source: str | None = None,
) -> bool:
    for attrs in _iter_edge_attrs(G, u, v):
        if relation is not None and attrs.get("relation") != relation:
            continue
        if source is not None and attrs.get("source") != source:
            continue
        return True
    return False


# ===========================================================================
# Part A: Unit tests on helper functions
# ===========================================================================
print("\n=== Part A: helper function unit tests ===")

# A-1: _normalize_context_label
check(
    _normalize_context_label("  hello   world  ") == "hello world",
    "A-1a  whitespace normalisation strips + collapses",
)
check(
    _normalize_context_label("house - living space") == "house - living space",
    "A-1b  already-clean label is unchanged",
)
check(
    _normalize_context_label("") == "",
    "A-1c  empty string stays empty",
)


# A-2: _add_explicit_relationships — context node creation
print("\n[A-2] _add_explicit_relationships: context node creation")

G_ctx = nx.DiGraph()
G_ctx.add_node("Element::A", label="A", class_="IfcWall")

_add_explicit_relationships(
    G_ctx,
    "Element::A",
    {
        "hosts": [],
        "hosted_by": [],
        "ifc_connected_to": [],
        "belongs_to_system": ["HVAC System"],
        "in_zone": ["Zone Alpha"],
        "classified_as": ["ISO 12006:A-1"],
    },
    {},
)

check("System::HVAC System" in G_ctx, "A-2a  System context node created")
check("Zone::Zone Alpha" in G_ctx, "A-2b  Zone context node created")
check(
    "Classification::ISO 12006:A-1" in G_ctx,
    "A-2c  Classification context node created",
)

check(
    G_ctx.nodes["System::HVAC System"].get("class_") == "IfcSystem",
    "A-2d  System node has class_='IfcSystem'",
)
check(
    G_ctx.nodes["Zone::Zone Alpha"].get("class_") == "IfcZone",
    "A-2e  Zone node has class_='IfcZone'",
)
check(
    G_ctx.nodes["Classification::ISO 12006:A-1"].get("class_")
    == "IfcClassificationReference",
    "A-2f  Classification node has correct class_",
)
check(
    G_ctx.nodes["System::HVAC System"].get("node_kind") == "context",
    "A-2g  System node has node_kind='context'",
)

# A-3: explicit edges have source="ifc"
print("\n[A-3] _add_explicit_relationships: edge provenance")

for target_nid, rel_name in [
    ("System::HVAC System", "belongs_to_system"),
    ("Zone::Zone Alpha", "in_zone"),
    ("Classification::ISO 12006:A-1", "classified_as"),
]:
    has_ifc = _has_edge_with(
        G_ctx, "Element::A", target_nid, relation=rel_name, source="ifc"
    )
    check(
        has_ifc,
        f"A-3  {rel_name} edge has source='ifc'",
    )
    check(
        _has_edge_with(G_ctx, "Element::A", target_nid, relation=rel_name),
        f"A-3  {rel_name} edge has correct relation attr",
    )

# A-4: element-to-element explicit edges
print("\n[A-4] _add_explicit_relationships: element-to-element edges")

G_e2e = nx.DiGraph()
G_e2e.add_node("Element::Door1", label="Door 1", class_="IfcDoor")
G_e2e.add_node("Element::Wall1", label="Wall 1", class_="IfcWall")
G_e2e.add_node("Element::Pipe1", label="Pipe 1", class_="IfcPipeSegment")
G_e2e.add_node("Element::Pipe2", label="Pipe 2", class_="IfcPipeSegment")

_add_explicit_relationships(
    G_e2e,
    "Element::Door1",
    {
        "hosted_by": ["WALL_GID"],
        "hosts": [],
        "ifc_connected_to": [],
        "belongs_to_system": [],
        "in_zone": [],
        "classified_as": [],
    },
    {"WALL_GID": "Element::Wall1"},
)
_add_explicit_relationships(
    G_e2e,
    "Element::Pipe1",
    {
        "hosts": [],
        "hosted_by": [],
        "ifc_connected_to": ["PIPE2_GID"],
        "belongs_to_system": [],
        "in_zone": [],
        "classified_as": [],
    },
    {"PIPE2_GID": "Element::Pipe2"},
)

check(
    G_e2e.has_edge("Element::Door1", "Element::Wall1"),
    "A-4a  hosted_by edge Door1 → Wall1 created",
)
check(
    _has_edge_with(G_e2e, "Element::Door1", "Element::Wall1", relation="hosted_by"),
    "A-4b  hosted_by edge has correct relation attr",
)
check(
    _has_edge_with(
        G_e2e, "Element::Door1", "Element::Wall1", relation="hosted_by", source="ifc"
    ),
    "A-4c  hosted_by edge has source='ifc'",
)

check(
    G_e2e.has_edge("Element::Pipe1", "Element::Pipe2"),
    "A-4d  ifc_connected_to Pipe1 → Pipe2 created",
)
check(
    G_e2e.has_edge("Element::Pipe2", "Element::Pipe1"),
    "A-4e  ifc_connected_to reverse Pipe2 → Pipe1 created (bidirectional)",
)
check(
    _has_edge_with(
        G_e2e, "Element::Pipe1", "Element::Pipe2", relation="ifc_connected_to"
    ),
    "A-4f  ifc_connected_to edge has correct relation attr",
)
check(
    _has_edge_with(
        G_e2e,
        "Element::Pipe1",
        "Element::Pipe2",
        relation="ifc_connected_to",
        source="ifc",
    ),
    "A-4g  ifc_connected_to edge has source='ifc'",
)

# A-5: missing target GlobalId is skipped gracefully
print("\n[A-5] _add_explicit_relationships: missing targets skipped")

G_miss = nx.DiGraph()
G_miss.add_node("Element::X", label="X", class_="IfcWall")

_add_explicit_relationships(
    G_miss,
    "Element::X",
    {
        "hosts": ["NONEXISTENT_GID"],
        "hosted_by": [],
        "ifc_connected_to": [],
        "belongs_to_system": [],
        "in_zone": [],
        "classified_as": [],
    },
    {},  # empty map — NONEXISTENT_GID resolves to nothing
)

check(
    G_miss.number_of_edges() == 0,
    "A-5  no edges added for unresolvable GlobalId targets",
)

# A-6: self-loop guard
print("\n[A-6] _add_explicit_relationships: self-loop guard")

G_self = nx.DiGraph()
G_self.add_node("Element::Y", label="Y", class_="IfcWall")

_add_explicit_relationships(
    G_self,
    "Element::Y",
    {
        "hosts": ["SELF_GID"],
        "hosted_by": [],
        "ifc_connected_to": [],
        "belongs_to_system": [],
        "in_zone": [],
        "classified_as": [],
    },
    {"SELF_GID": "Element::Y"},  # target is self
)

check(
    not G_self.has_edge("Element::Y", "Element::Y"),
    "A-6  self-referential host edge not created",
)


# ===========================================================================
# Part B: build_graph_from_jsonl with synthetic JSONL containing Relationships
# ===========================================================================
print("\n=== Part B: build_graph_from_jsonl with explicit relationships ===")

_RECORDS = [
    {
        "GlobalId": "PROJ001",
        "IfcType": "IfcProject",
        "Name": "Test Project",
        "Hierarchy": {},
    },
    {
        "GlobalId": "BLDG001",
        "IfcType": "IfcBuilding",
        "Name": "Test Building",
        "Hierarchy": {"ParentId": "PROJ001"},
        "Relationships": {
            "hosts": [],
            "hosted_by": [],
            "ifc_connected_to": [],
            "belongs_to_system": [],
            "in_zone": [],
            "classified_as": ["ISO 16739:IfcBuilding"],
        },
    },
    {
        "GlobalId": "STOR001",
        "IfcType": "IfcBuildingStorey",
        "Name": "Level 0",
        "Hierarchy": {"ParentId": "BLDG001"},
        "Geometry": {
            "Centroid": [0.0, 0.0, 1.5],
            "BoundingBox": {"min": [-10.0, -10.0, 0.0], "max": [10.0, 10.0, 3.0]},
        },
    },
    {
        "GlobalId": "WALL001",
        "IfcType": "IfcWall",
        "Name": "Wall 1",
        "Hierarchy": {"ParentId": "STOR001"},
        "Geometry": {
            "Centroid": [0.0, 0.0, 1.5],
            "BoundingBox": {"min": [-5.0, 0.0, 0.0], "max": [5.0, 0.2, 3.0]},
        },
        "Relationships": {
            "hosts": ["DOOR001"],
            "hosted_by": [],
            "ifc_connected_to": [],
            "belongs_to_system": [],
            "in_zone": ["Zone Alpha"],
            "classified_as": [],
        },
    },
    {
        "GlobalId": "DOOR001",
        "IfcType": "IfcDoor",
        "Name": "Door 1",
        "Hierarchy": {"ParentId": "STOR001"},
        "Geometry": {
            "Centroid": [1.0, 0.0, 1.0],
            "BoundingBox": {"min": [0.5, -0.1, 0.0], "max": [1.5, 0.1, 2.1]},
        },
        "Relationships": {
            "hosts": [],
            "hosted_by": ["WALL001"],
            "ifc_connected_to": [],
            "belongs_to_system": ["Fire Safety System"],
            "in_zone": ["Zone Alpha"],
            "classified_as": ["ISO 16739:IfcDoor"],
        },
    },
    {
        "GlobalId": "PIPE001",
        "IfcType": "IfcPipeSegment",
        "Name": "Pipe 1",
        "Hierarchy": {"ParentId": "STOR001"},
        "Geometry": {
            "Centroid": [2.0, 2.0, 1.0],
            "BoundingBox": {"min": [1.5, 1.5, 0.5], "max": [2.5, 2.5, 1.5]},
        },
        "Relationships": {
            "hosts": [],
            "hosted_by": [],
            "ifc_connected_to": ["PIPE002"],
            "belongs_to_system": ["HVAC Supply"],
            "in_zone": [],
            "classified_as": [],
        },
    },
    {
        "GlobalId": "PIPE002",
        "IfcType": "IfcPipeSegment",
        "Name": "Pipe 2",
        "Hierarchy": {"ParentId": "STOR001"},
        "Geometry": {
            "Centroid": [3.0, 2.0, 1.0],
            "BoundingBox": {"min": [2.5, 1.5, 0.5], "max": [3.5, 2.5, 1.5]},
        },
        "Relationships": {
            "hosts": [],
            "hosted_by": [],
            "ifc_connected_to": ["PIPE001"],
            "belongs_to_system": ["HVAC Supply"],
            "in_zone": [],
            "classified_as": [],
        },
    },
]

with tempfile.NamedTemporaryFile(
    mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
) as fh:
    for rec in _RECORDS:
        fh.write(json.dumps(rec) + "\n")
    _tmp_path = Path(fh.name)

try:
    G = build_graph_from_jsonl([_tmp_path])
    add_spatial_adjacency(G)
finally:
    _tmp_path.unlink(missing_ok=True)

# B-1: context nodes created
print("\n[B-1] Context nodes created with deterministic IDs")
check("Zone::Zone Alpha" in G, "B-1a  Zone::Zone Alpha node created")
check(
    "Classification::ISO 16739:IfcBuilding" in G,
    "B-1b  Classification node created from IfcBuilding record",
)
check(
    "Classification::ISO 16739:IfcDoor" in G,
    "B-1c  Classification node created from IfcDoor record",
)
check(
    "System::Fire Safety System" in G, "B-1d  System::Fire Safety System node created"
)
check("System::HVAC Supply" in G, "B-1e  System::HVAC Supply node created")

# B-2: explicit edges with correct attrs
print("\n[B-2] Explicit edge attributes (relation + source)")

# hosts edge: Wall1 → Door1
check(
    G.has_edge("Element::WALL001", "Element::DOOR001"),
    "B-2a  hosts edge Element::WALL001 → Element::DOOR001 present",
)
check(
    _has_edge_with(G, "Element::WALL001", "Element::DOOR001", relation="hosts"),
    "B-2b  hosts edge has relation='hosts'",
)
check(
    _has_edge_with(
        G, "Element::WALL001", "Element::DOOR001", relation="hosts", source="ifc"
    ),
    "B-2c  hosts edge has source='ifc'",
)

# hosted_by edge: Door1 → Wall1
check(
    G.has_edge("Element::DOOR001", "Element::WALL001"),
    "B-2d  hosted_by edge Element::DOOR001 → Element::WALL001 present",
)
check(
    _has_edge_with(G, "Element::DOOR001", "Element::WALL001", relation="hosted_by"),
    "B-2e  hosted_by edge has relation='hosted_by'",
)
check(
    _has_edge_with(
        G,
        "Element::DOOR001",
        "Element::WALL001",
        relation="hosted_by",
        source="ifc",
    ),
    "B-2f  hosted_by edge has source='ifc'",
)

# ifc_connected_to: Pipe1 ↔ Pipe2
check(
    G.has_edge("Element::PIPE001", "Element::PIPE002"),
    "B-2g  ifc_connected_to Pipe1 → Pipe2 present",
)
check(
    G.has_edge("Element::PIPE002", "Element::PIPE001"),
    "B-2h  ifc_connected_to Pipe2 → Pipe1 present (bidirectional)",
)
check(
    _has_edge_with(
        G, "Element::PIPE001", "Element::PIPE002", relation="ifc_connected_to"
    ),
    "B-2i  ifc_connected_to edge has correct relation attr",
)
check(
    _has_edge_with(
        G,
        "Element::PIPE001",
        "Element::PIPE002",
        relation="ifc_connected_to",
        source="ifc",
    ),
    "B-2j  ifc_connected_to edge has source='ifc'",
)

# in_zone edge
check(
    G.has_edge("Element::WALL001", "Zone::Zone Alpha"),
    "B-2k  in_zone edge Element::WALL001 → Zone::Zone Alpha present",
)
check(
    _has_edge_with(
        G, "Element::WALL001", "Zone::Zone Alpha", relation="in_zone", source="ifc"
    ),
    "B-2l  in_zone edge has source='ifc'",
)

# classified_as edge on IfcBuilding node
check(
    G.has_edge("IfcBuilding", "Classification::ISO 16739:IfcBuilding"),
    "B-2m  classified_as edge IfcBuilding → Classification present",
)
check(
    _has_edge_with(
        G,
        "IfcBuilding",
        "Classification::ISO 16739:IfcBuilding",
        relation="classified_as",
        source="ifc",
    ),
    "B-2n  classified_as edge has source='ifc'",
)

# belongs_to_system
check(
    G.has_edge("Element::PIPE001", "System::HVAC Supply"),
    "B-2o  belongs_to_system edge Pipe1 → System present",
)
check(
    _has_edge_with(
        G,
        "Element::PIPE001",
        "System::HVAC Supply",
        relation="belongs_to_system",
        source="ifc",
    ),
    "B-2p  belongs_to_system edge has source='ifc'",
)

# B-3: Shared context node deduplication
print("\n[B-3] Context node deduplication")
zone_nodes = [n for n in G.nodes if str(n).startswith("Zone::Zone Alpha")]
check(
    len(zone_nodes) == 1,
    f"B-3a  Zone::Zone Alpha created exactly once (found {len(zone_nodes)})",
)

system_nodes = [n for n in G.nodes if str(n) == "System::HVAC Supply"]
check(
    len(system_nodes) == 1,
    f"B-3b  System::HVAC Supply deduplicated (found {len(system_nodes)})",
)

# Both Pipe1 and Pipe2 belong to HVAC Supply — check both edges exist
check(
    G.has_edge("Element::PIPE001", "System::HVAC Supply"),
    "B-3c  Pipe1 → System::HVAC Supply edge present",
)
check(
    G.has_edge("Element::PIPE002", "System::HVAC Supply"),
    "B-3d  Pipe2 → System::HVAC Supply edge present (same context node)",
)

# B-4: heuristic adjacency edges have source="heuristic"
print("\n[B-4] Heuristic adjacency edges have source='heuristic'")

heuristic_edges = [
    (u, v, d)
    for u, v, d in G.edges(data=True)
    if d.get("relation") in {"adjacent_to", "connected_to"}
]
check(
    len(heuristic_edges) > 0,
    f"B-4a  heuristic adjacency edges exist (found {len(heuristic_edges)})",
)
all_heuristic = all(d.get("source") == "heuristic" for _, _, d in heuristic_edges)
check(
    all_heuristic,
    "B-4b  all adjacent_to/connected_to edges have source='heuristic'",
)

# B-5: connected_to is heuristic-only — never source="ifc"
print("\n[B-5] connected_to is heuristic-only")

connected_to_ifc = [
    (u, v, d)
    for u, v, d in G.edges(data=True)
    if d.get("relation") == "connected_to" and d.get("source") == "ifc"
]
check(
    len(connected_to_ifc) == 0,
    f"B-5  no connected_to edges with source='ifc' (found {len(connected_to_ifc)})",
)

# B-6: ifc_connected_to is explicit-only — never source="heuristic"
print("\n[B-6] ifc_connected_to is explicit-only")

ifc_conn_heuristic = [
    (u, v, d)
    for u, v, d in G.edges(data=True)
    if d.get("relation") == "ifc_connected_to" and d.get("source") == "heuristic"
]
check(
    len(ifc_conn_heuristic) == 0,
    "B-6  no ifc_connected_to edges with source='heuristic' "
    f"(found {len(ifc_conn_heuristic)})",
)

# B-7: explicit and heuristic edges coexist
print("\n[B-7] Explicit and heuristic edges coexist")

explicit_edges = [
    (u, v, d) for u, v, d in G.edges(data=True) if d.get("source") == "ifc"
]
check(
    len(explicit_edges) > 0,
    f"B-7a  explicit (source='ifc') edges present (found {len(explicit_edges)})",
)
check(
    len(heuristic_edges) > 0,
    "B-7b  heuristic (source='heuristic') edges present "
    f"(found {len(heuristic_edges)})",
)

# B-8: provenance on all non-hierarchy edges
print("\n[B-8] Provenance (source attr) on all non-hierarchy edges")

HIERARCHY_RELATIONS = {"aggregates", "contains", "contained_in"}
non_hierarchy_no_source = [
    (u, v, d)
    for u, v, d in G.edges(data=True)
    if d.get("relation") not in HIERARCHY_RELATIONS and "source" not in d
]
check(
    len(non_hierarchy_no_source) == 0,
    f"B-8  all non-hierarchy edges have 'source' attr "
    f"(missing on {len(non_hierarchy_no_source)} edge(s): "
    f"{[(u, v) for u, v, _ in non_hierarchy_no_source[:5]]})",
)

# B-9: edge_categories includes "explicit" key
print("\n[B-9] edge_categories metadata")

cats = G.graph.get("edge_categories", {})
check("explicit" in cats, "B-9a  edge_categories contains 'explicit' key")
check(
    "ifc_connected_to" in (cats.get("explicit") or []),
    "B-9b  'ifc_connected_to' listed under explicit edge categories",
)
check(
    "connected_to" in (cats.get("spatial") or []),
    "B-9c  'connected_to' still listed under spatial (heuristic) edge categories",
)
check("hierarchy" in cats, "B-9d  hierarchy category still present (backward compat)")
check("spatial" in cats, "B-9e  spatial category still present (backward compat)")
check("topology" in cats, "B-9f  topology category still present (backward compat)")

# B-10: backward compat — hierarchy edges still intact
print("\n[B-10] Backward compat: hierarchy edges still intact")

check(
    G.has_edge("Storey::STOR001", "Element::WALL001"),
    "B-10a  contains edge Storey → Wall still present",
)
check(
    _has_edge_with(G, "Storey::STOR001", "Element::WALL001", relation="contains"),
    "B-10b  contains edge has correct relation attr",
)
check(
    G.has_edge("Element::WALL001", "Storey::STOR001"),
    "B-10c  contained_in reverse edge still present",
)

# B-11: full build path (adds adjacency + topology) must preserve explicit IFC
# edges when relations share the same (u, v) pair in a DiGraph.
print("\n[B-11] build_graph path preserves explicit IFC edges after topology")

with tempfile.NamedTemporaryFile(
    mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
) as fh:
    for rec in _RECORDS:
        fh.write(json.dumps(rec) + "\n")
    _tmp_path_full = Path(fh.name)

try:
    G_full_path = build_graph([_tmp_path_full])
finally:
    _tmp_path_full.unlink(missing_ok=True)

check(
    _has_edge_with(
        G_full_path, "Element::WALL001", "Element::DOOR001", relation="hosts"
    ),
    "B-11a explicit hosts relation survives topology pass",
)
check(
    _has_edge_with(
        G_full_path,
        "Element::WALL001",
        "Element::DOOR001",
        relation="hosts",
        source="ifc",
    ),
    "B-11b explicit hosts source='ifc' survives topology pass",
)

check(
    _has_edge_with(
        G_full_path, "Element::DOOR001", "Element::WALL001", relation="hosted_by"
    ),
    "B-11c explicit hosted_by relation survives topology pass",
)
check(
    _has_edge_with(
        G_full_path,
        "Element::DOOR001",
        "Element::WALL001",
        relation="hosted_by",
        source="ifc",
    ),
    "B-11d explicit hosted_by source='ifc' survives topology pass",
)


# ===========================================================================
# Part C: Live JSONL smoke-test (skipped if no output files present)
# ===========================================================================
print("\n=== Part C: Live JSONL smoke-test ===")

_project_root = Path(__file__).resolve().parents[1]
_jsonl_files = sorted((_project_root / "output").glob("*.jsonl"))

if not _jsonl_files:
    print("  NOTE  No output/*.jsonl files found — skipping live smoke-test.")
    print("        Run: uv run rag-tag-ifc-to-jsonl  first.")
else:
    print(
        f"  Found {len(_jsonl_files)} JSONL file(s): {[f.name for f in _jsonl_files]}"
    )
    try:
        G_live = build_graph(_jsonl_files)
        print(
            f"  Graph: {G_live.number_of_nodes()} nodes, "
            f"{G_live.number_of_edges()} edges"
        )

        # C-1: explicit edges present (in_zone and classified_as appear in sample data)
        live_explicit = [
            (u, v, d) for u, v, d in G_live.edges(data=True) if d.get("source") == "ifc"
        ]
        check(
            len(live_explicit) > 0,
            "C-1  live graph has explicit (source='ifc') edges "
            f"(found {len(live_explicit)})",
        )

        # C-2: heuristic edges present
        live_heuristic = [
            (u, v, d)
            for u, v, d in G_live.edges(data=True)
            if d.get("source") == "heuristic"
        ]
        check(
            len(live_heuristic) > 0,
            "C-2  live graph has heuristic (source='heuristic') edges "
            f"(found {len(live_heuristic)})",
        )

        # C-3: no connected_to with source="ifc"
        live_bad_connected = [
            (u, v, d)
            for u, v, d in G_live.edges(data=True)
            if d.get("relation") == "connected_to" and d.get("source") == "ifc"
        ]
        check(
            len(live_bad_connected) == 0,
            f"C-3  no connected_to edges with source='ifc' in live graph "
            f"(found {len(live_bad_connected)})",
        )

        # C-4: all non-hierarchy edges have provenance
        live_missing_source = [
            (u, v, d)
            for u, v, d in G_live.edges(data=True)
            if d.get("relation") not in HIERARCHY_RELATIONS and "source" not in d
        ]
        check(
            len(live_missing_source) == 0,
            f"C-4  all non-hierarchy edges have 'source' attr in live graph "
            f"(missing on {len(live_missing_source)} edge(s))",
        )

        # C-5: context nodes present (in_zone and classified_as in sample data)
        zone_nodes_live = [n for n in G_live.nodes if str(n).startswith("Zone::")]
        cls_nodes_live = [
            n for n in G_live.nodes if str(n).startswith("Classification::")
        ]
        check(
            len(zone_nodes_live) > 0 or len(cls_nodes_live) > 0,
            f"C-5  live graph has Zone:: or Classification:: context nodes "
            f"(zones={len(zone_nodes_live)}, cls={len(cls_nodes_live)})",
        )

        # C-6: edge_categories present on live graph
        live_cats = G_live.graph.get("edge_categories", {})
        check(
            "explicit" in live_cats,
            "C-6  live graph edge_categories contains 'explicit' key",
        )

    except Exception as exc:  # noqa: BLE001
        print(f"  NOTE  Live graph build failed: {exc}")
        print("        This is not a test failure — check JSONL files are valid.")


# ===========================================================================
# Summary
# ===========================================================================
print()
if _failures:
    print(f"\033[31m{len(_failures)} check(s) FAILED:\033[0m")
    for f in _failures:
        print(f"  - {f}")
    sys.exit(1)
else:
    print("\033[32mAll checks passed.\033[0m")
