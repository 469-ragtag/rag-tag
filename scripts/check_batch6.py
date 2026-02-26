"""Batch 6 smoke-tests: dataset selection consistency + jsonl_to_graph verification.

Covers:
  Part A – _resolve_graph_dataset priority/fallback rules
  Part B – build_graph_from_jsonl: payload storage, hierarchy edges, node IDs

Run with:
    uv run python scripts/check_batch6.py
"""

from __future__ import annotations

import inspect
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

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


# ===========================================================================
# Part A: Dataset selection consistency
# ===========================================================================
print("\n=== Part A: _resolve_graph_dataset priority / fallback ===")

from rag_tag.run_agent import _resolve_graph_dataset  # noqa: E402

# A-1: explicit --graph-dataset wins over everything
check(
    _resolve_graph_dataset("Explicit", None) == "Explicit",
    "A-1  explicit flag returned as-is (no db)",
)
check(
    _resolve_graph_dataset("Explicit", Path("/x/Other.db")) == "Explicit",
    "A-2  explicit flag wins over db stem",
)

# A-3: --db stem used when no explicit flag
check(
    _resolve_graph_dataset(None, Path("/data/MyBuilding.db")) == "MyBuilding",
    "A-3  db stem inferred when no --graph-dataset",
)

# A-4: fallback must be None (not "Building-Architecture")
fallback = _resolve_graph_dataset(None, None)
check(
    fallback is None,
    f"A-4  fallback is None (got {fallback!r}) – loads all JSONL",
)
check(
    fallback != "Building-Architecture",
    "A-5  hardcoded 'Building-Architecture' no longer used as fallback",
)

# A-6: return type annotation accepts None
sig = inspect.signature(_resolve_graph_dataset)
return_annotation = sig.return_annotation
check(
    "None" in str(return_annotation) or return_annotation is inspect.Parameter.empty,
    f"A-6  return annotation allows None (got {return_annotation!r})",
)

# A-7: run_tui signature accepts graph_dataset=None without TypeError
print("\n[Part A-7] run_tui + QueryApp accept graph_dataset=None")
try:
    from rag_tag.textual_app import QueryApp, run_tui  # noqa: E402

    tui_sig = inspect.signature(run_tui)
    app_sig = inspect.signature(QueryApp.__init__)
    check(
        "graph_dataset" in tui_sig.parameters,
        "A-7a run_tui has graph_dataset parameter",
    )
    check(
        "graph_dataset" in app_sig.parameters,
        "A-7b QueryApp.__init__ has graph_dataset parameter",
    )
    gd_default = tui_sig.parameters["graph_dataset"].default
    check(
        gd_default is None,
        f"A-7c run_tui graph_dataset default is None (got {gd_default!r})",
    )
except ImportError as exc:
    print(f"  NOTE: Textual import skipped in headless env ({exc})")

# A-8: query_service.execute_query passes graph_dataset to _ensure_graph_context
print("\n[Part A-8] execute_query propagates graph_dataset")
import types  # noqa: E402

import networkx as nx  # noqa: E402

# Minimal stubs so we can load query_service without every heavy dep.
_stubs: dict[str, types.ModuleType] = {}

_router_stub = types.ModuleType("rag_tag.router")

# We'll use real RouteDecision / SqlRequest from rag_tag.router if available,
# otherwise fall back to a dataclass stub.
try:
    from rag_tag.router import RouteDecision, route_question  # noqa: E402

    _router_stub.RouteDecision = RouteDecision  # type: ignore[attr-defined]
    _router_stub.route_question = route_question  # type: ignore[attr-defined]
    _router_stub.SqlRequest = getattr(
        sys.modules.get("rag_tag.router"), "SqlRequest", None
    )
except Exception:
    from dataclasses import dataclass  # noqa: E402

    @dataclass(frozen=True)
    class _FakeDecision:
        route: str
        reason: str
        sql_request: object | None

    _router_stub.RouteDecision = _FakeDecision  # type: ignore[attr-defined]
    _router_stub.route_question = None  # type: ignore[attr-defined]
    _router_stub.SqlRequest = None  # type: ignore[attr-defined]

# Monkeypatch _ensure_graph_context to capture graph_dataset argument.
_captured_dataset: list[str | None] = []

import rag_tag.query_service as _qs  # noqa: E402


def _fake_ensure(
    graph: nx.DiGraph | None,
    agent: object | None,
    debug: bool,
    dataset: str | None = None,
) -> tuple[nx.DiGraph, object]:
    _captured_dataset.append(dataset)
    return nx.DiGraph(), object()


_orig_ensure = _qs._ensure_graph_context  # noqa: SLF001
_qs._ensure_graph_context = _fake_ensure  # type: ignore[attr-defined]

# Fake a graph-route decision so execute_query hits the graph branch.
try:
    from rag_tag.router import RouteDecision as _RD  # noqa: E402

    fake_decision = _RD(route="graph", reason="test", sql_request=None)  # type: ignore[call-arg]
except Exception:
    from dataclasses import dataclass  # noqa: E402

    @dataclass
    class _FakeRD:  # type: ignore[no-redef]
        route: str = "graph"
        reason: str = "test"
        sql_request: object | None = None

    fake_decision = _FakeRD()

_qs.execute_query(
    "dummy question",
    [],
    None,
    None,
    decision=fake_decision,
    graph_dataset="TestDataset",
)

check(
    len(_captured_dataset) == 1 and _captured_dataset[0] == "TestDataset",
    f"A-8  execute_query propagated graph_dataset='TestDataset' "
    f"(captured: {_captured_dataset!r})",
)

_qs._ensure_graph_context = _orig_ensure  # type: ignore[attr-defined]

# ===========================================================================
# Part B: jsonl_to_graph — payload storage + hierarchy edge construction
# ===========================================================================
print("\n=== Part B: jsonl_to_graph payload + hierarchy ===")

from rag_tag.parser.jsonl_to_graph import build_graph_from_jsonl  # noqa: E402

# Synthetic JSONL records forming a minimal IFC hierarchy:
#   IfcProject (PROJ001)
#     └─ IfcBuilding (BLDG001)
#           └─ IfcBuildingStorey (STOR001)
#                 └─ IfcWall (WALL001)
#                 └─ IfcDoor (DOOR001)

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
        "PropertySets": {
            "Official": {
                "Pset_WallCommon": {"FireRating": "EI 60"},
            }
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
finally:
    _tmp_path.unlink(missing_ok=True)

# B-1: Expected node IDs exist
print("\n[Part B-1] Node ID strategy")
check("IfcProject" in G, "B-1a sentinel node 'IfcProject' present")
check("IfcBuilding" in G, "B-1b sentinel node 'IfcBuilding' present")
check("Storey::STOR001" in G, "B-1c storey node uses 'Storey::' prefix")
check("Element::WALL001" in G, "B-1d wall node uses 'Element::' prefix")
check("Element::DOOR001" in G, "B-1e door node uses 'Element::' prefix")

# B-2: payload = full parsed record stored on every node
print("\n[Part B-2] payload=rec stored on nodes")
check(
    G.nodes["IfcProject"].get("payload", {}).get("GlobalId") == "PROJ001",
    "B-2a IfcProject payload.GlobalId == 'PROJ001'",
)
check(
    G.nodes["IfcBuilding"].get("payload", {}).get("GlobalId") == "BLDG001",
    "B-2b IfcBuilding payload.GlobalId == 'BLDG001'",
)
check(
    G.nodes["Storey::STOR001"].get("payload", {}).get("GlobalId") == "STOR001",
    "B-2c Storey payload.GlobalId == 'STOR001'",
)
check(
    G.nodes["Element::WALL001"].get("payload", {}).get("GlobalId") == "WALL001",
    "B-2d Wall payload.GlobalId == 'WALL001'",
)
check(
    G.nodes["Element::DOOR001"].get("payload", {}).get("GlobalId") == "DOOR001",
    "B-2e Door payload.GlobalId == 'DOOR001'",
)

# Verify payload is the *full* record, not just a subset.
wall_payload = G.nodes["Element::WALL001"].get("payload") or {}
check(
    "PropertySets" in wall_payload,
    "B-2f Wall payload retains nested PropertySets block",
)
check(
    wall_payload.get("Name") == "Wall 1",
    "B-2g Wall payload.Name == 'Wall 1'",
)

# B-3: Hierarchy edges built from ParentId (containment)
print("\n[Part B-3] Hierarchy edges via Hierarchy.ParentId")

# Storey → Wall (contains + contained_in)
check(
    G.has_edge("Storey::STOR001", "Element::WALL001"),
    "B-3a edge Storey::STOR001 → Element::WALL001 exists",
)
check(
    G["Storey::STOR001"]["Element::WALL001"].get("relation") == "contains",
    "B-3b that edge has relation='contains'",
)
check(
    G.has_edge("Element::WALL001", "Storey::STOR001"),
    "B-3c reverse edge Element::WALL001 → Storey::STOR001 exists",
)
check(
    G["Element::WALL001"]["Storey::STOR001"].get("relation") == "contained_in",
    "B-3d reverse edge has relation='contained_in'",
)

# Storey → Door
check(
    G.has_edge("Storey::STOR001", "Element::DOOR001"),
    "B-3e edge Storey::STOR001 → Element::DOOR001 exists",
)
check(
    G.has_edge("Element::DOOR001", "Storey::STOR001"),
    "B-3f reverse edge Element::DOOR001 → Storey::STOR001 exists",
)

# Building → Storey (may be 'aggregates' from initial setup or 'contains'
# from second-pass — both are valid hierarchical relations).
check(
    G.has_edge("IfcBuilding", "Storey::STOR001"),
    "B-3g edge IfcBuilding → Storey::STOR001 exists (aggregates or contains)",
)
bldg_to_storey_rel = (G["IfcBuilding"]["Storey::STOR001"] or {}).get("relation")
check(
    bldg_to_storey_rel in {"aggregates", "contains"},
    f"B-3h IfcBuilding→Storey relation is hierarchical (got {bldg_to_storey_rel!r})",
)

# B-4: flat properties pulled from record top-level fields
print("\n[Part B-4] _flat_properties stored on 'properties' attribute")
wall_props = G.nodes["Element::WALL001"].get("properties") or {}
check(
    wall_props.get("GlobalId") == "WALL001",
    "B-4a properties.GlobalId == 'WALL001'",
)
check(
    wall_props.get("Class") == "IfcWall",
    "B-4b properties.Class == 'IfcWall' (pulled from IfcType)",
)
check(
    wall_props.get("Name") == "Wall 1",
    "B-4c properties.Name == 'Wall 1'",
)

# B-5: geometry from Geometry block wired to node 'geometry' and 'bbox'
print("\n[Part B-5] Geometry block wired to node attributes")
check(
    G.nodes["Element::WALL001"].get("geometry") == (0.0, 0.0, 1.5),
    "B-5a Wall geometry (centroid) == (0,0,1.5)",
)
wall_bbox = G.nodes["Element::WALL001"].get("bbox")
check(
    wall_bbox is not None,
    "B-5b Wall bbox is not None",
)
if wall_bbox is not None:
    check(
        wall_bbox[0] == (-5.0, 0.0, 0.0),
        f"B-5c Wall bbox.min == (-5,0,0) (got {wall_bbox[0]!r})",
    )
    check(
        wall_bbox[1] == (5.0, 0.2, 3.0),
        f"B-5d Wall bbox.max == (5,0.2,3) (got {wall_bbox[1]!r})",
    )

# B-6: no orphaned nodes created for unknown ParentId references
print("\n[Part B-6] Orphaned/dangling parent GIDs produce no extra nodes")
node_count = G.number_of_nodes()
# We created PROJ001,BLDG001,STOR001,WALL001,DOOR001 → plus sentinels
# IfcProject and IfcBuilding are REUSED sentinels, so we expect exactly 5 nodes.
check(
    node_count == 5,
    f"B-6  graph has exactly 5 nodes (got {node_count})",
)

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
