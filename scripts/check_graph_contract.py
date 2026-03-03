"""Batch 4 smoke-tests: canonical graph contract invariants.

Run with:
    uv run python scripts/check_graph_contract.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import networkx as nx

from rag_tag.graph_contract import (
    CANONICAL_RELATION_SET,
    EXPLICIT_IFC_RELATIONS,
    HIERARCHY_RELATIONS,
    KNOWN_RELATION_SOURCES,
    SPATIAL_RELATIONS,
    TOPOLOGY_RELATIONS,
    has_valid_envelope_shape,
    missing_required_action_fields,
    normalize_relation_name,
    normalize_relation_source,
    relation_bucket,
)
from rag_tag.ifc_graph_tool import query_ifc_graph

FAIL = "\033[31mFAIL\033[0m"
PASS = "\033[32mPASS\033[0m"
_failures: list[str] = []


def check(condition: bool, name: str) -> None:
    if condition:
        print(f"  {PASS}  {name}")
    else:
        print(f"  {FAIL}  {name}")
        _failures.append(name)


def _build_graph() -> nx.DiGraph:
    G = nx.DiGraph()

    G.add_node(
        "IfcBuilding",
        label="Building A",
        class_="IfcBuilding",
        properties={"GlobalId": "BLDG1", "Name": "Building A"},
        payload={"PropertySets": {"Official": {}, "Custom": {}}},
    )

    G.add_node(
        "Storey::STOREY1",
        label="Level 0",
        class_="IfcBuildingStorey",
        properties={"GlobalId": "STOREY1", "Name": "Level 0"},
        payload={"PropertySets": {"Official": {}, "Custom": {}}},
    )

    G.add_node(
        "Storey::STOREY2",
        label="Level 1",
        class_="IfcBuildingStorey",
        properties={"GlobalId": "STOREY2", "Name": "Level 1"},
        payload={"PropertySets": {"Official": {}, "Custom": {}}},
    )

    G.add_node(
        "Element::WALL1",
        label="Wall 1",
        class_="IfcWall",
        properties={"GlobalId": "WALL1", "Name": "Wall 1"},
        payload={
            "PropertySets": {
                "Official": {"Pset_WallCommon": {"FireRating": "EI 90"}},
                "Custom": {},
            },
            "Quantities": {"Qto_WallBaseQuantities": {"Length": 5.0}},
        },
    )

    G.add_node(
        "Element::DOOR1",
        label="Door 1",
        class_="IfcDoor",
        properties={"GlobalId": "DOOR1", "Name": "Door 1"},
        payload={
            "PropertySets": {
                "Official": {"Pset_DoorCommon": {"FireRating": "EI 60"}},
                "Custom": {},
            }
        },
    )

    G.add_node(
        "Element::COLUMN1",
        label="Column 1",
        class_="IfcColumn",
        properties={"GlobalId": "COLUMN1", "Name": "Column 1"},
        payload={"PropertySets": {"Official": {}, "Custom": {}}},
    )

    G.add_node(
        "Element::SLAB1",
        label="Slab 1",
        class_="IfcSlab",
        properties={"GlobalId": "SLAB1", "Name": "Slab 1"},
        payload={"PropertySets": {"Official": {}, "Custom": {}}},
    )

    G.add_node(
        "Element::WALL2",
        label="Wall 2",
        class_="IfcWall",
        properties={"GlobalId": "WALL2", "Name": "Wall 2"},
        payload={"PropertySets": {"Official": {}, "Custom": {}}},
    )

    G.add_node(
        "Element::PIPE1",
        label="Pipe 1",
        class_="IfcPipeSegment",
        properties={"GlobalId": "PIPE1", "Name": "Pipe 1"},
        payload={"PropertySets": {"Official": {}, "Custom": {}}},
    )

    G.add_node(
        "Element::TERMINAL1",
        label="Terminal 1",
        class_="IfcFlowTerminal",
        properties={"GlobalId": "TERM1", "Name": "Terminal 1"},
        payload={"PropertySets": {"Official": {}, "Custom": {}}},
    )

    G.add_node(
        "System::SYS1",
        label="System 1",
        class_="IfcSystem",
        properties={"GlobalId": "SYS1", "Name": "System 1"},
        payload={"PropertySets": {"Official": {}, "Custom": {}}},
    )

    # Hierarchy edges
    G.add_edge("IfcBuilding", "Storey::STOREY1", relation="contains")
    G.add_edge("Storey::STOREY1", "IfcBuilding", relation="contained_in")
    G.add_edge("IfcBuilding", "Storey::STOREY2", relation="contains")
    G.add_edge("Storey::STOREY2", "IfcBuilding", relation="contained_in")

    G.add_edge(
        "Storey::STOREY1",
        "Element::WALL1",
        relation="contains",
        source="ifc",
    )
    G.add_edge(
        "Element::WALL1",
        "Storey::STOREY1",
        relation="contained_in",
        source="topology",
    )
    G.add_edge("Storey::STOREY1", "Element::DOOR1", relation="contains")
    G.add_edge("Element::DOOR1", "Storey::STOREY1", relation="contained_in")
    G.add_edge("Storey::STOREY1", "Element::COLUMN1", relation="contains")
    G.add_edge("Element::COLUMN1", "Storey::STOREY1", relation="contained_in")
    G.add_edge("Storey::STOREY1", "Element::SLAB1", relation="contains")
    G.add_edge("Element::SLAB1", "Storey::STOREY1", relation="contained_in")
    G.add_edge("Storey::STOREY1", "Element::PIPE1", relation="contains")
    G.add_edge("Element::PIPE1", "Storey::STOREY1", relation="contained_in")
    G.add_edge("Storey::STOREY1", "Element::TERMINAL1", relation="contains")
    G.add_edge("Element::TERMINAL1", "Storey::STOREY1", relation="contained_in")
    G.add_edge("Storey::STOREY2", "Element::WALL2", relation="contains")
    G.add_edge("Element::WALL2", "Storey::STOREY2", relation="contained_in")

    # Spatial edges
    G.add_edge(
        "Element::WALL1",
        "Element::DOOR1",
        relation="adjacent_to",
        distance=0.8,
        source="ifc",
    )
    G.add_edge(
        "Element::DOOR1",
        "Element::WALL1",
        relation="adjacent_to",
        distance=0.8,
        source="ifc",
    )

    # Topology edges
    G.add_edge(
        "Element::WALL1",
        "Element::COLUMN1",
        relation="above",
        vertical_gap=0.5,
        source="heuristic",
    )
    G.add_edge(
        "Element::COLUMN1",
        "Element::WALL1",
        relation="below",
        vertical_gap=0.5,
        source="heuristic",
    )
    G.add_edge(
        "Element::WALL1",
        "Element::SLAB1",
        relation="intersects_3d",
        intersection_volume=1.2,
        contact_area=0.4,
        source="heuristic",
    )
    G.add_edge(
        "Element::SLAB1",
        "Element::WALL1",
        relation="intersects_3d",
        intersection_volume=1.2,
        contact_area=0.4,
        source="heuristic",
    )

    # Explicit IFC relationship edges
    G.add_edge(
        "Element::WALL1",
        "Element::PIPE1",
        relation="hosts",
        source="topology",
    )
    G.add_edge(
        "Element::PIPE1",
        "Element::WALL1",
        relation="hosted_by",
        source="topology",
    )
    G.add_edge(
        "Element::PIPE1",
        "Element::TERMINAL1",
        relation="ifc_connected_to",
        source="topology",
    )
    G.add_edge(
        "Element::TERMINAL1",
        "System::SYS1",
        relation="belongs_to_system",
        source="topology",
    )

    return G


def _assert_ok_action(
    G: nx.DiGraph,
    action: str,
    params: dict[str, Any],
) -> dict[str, Any]:
    result = query_ifc_graph(G, action, params)
    check(has_valid_envelope_shape(result), f"{action}: envelope keys are stable")
    check(result.get("status") == "ok", f"{action}: status=ok")
    missing = missing_required_action_fields(action, result.get("data"))
    check(not missing, f"{action}: required data fields present ({missing or 'ok'})")
    return result


def _source_matches_bucket_semantics(item: dict[str, Any]) -> bool:
    relation = normalize_relation_name(item.get("relation"))
    source = normalize_relation_source(item.get("source"))
    bucket = relation_bucket(relation)
    if bucket == "explicit_ifc":
        return source == "ifc"
    if bucket == "topology":
        return source == "topology"
    if bucket == "spatial":
        return source == "heuristic"
    if bucket == "hierarchy":
        return source is None
    return False


print("\n[Contract] Envelope + required field invariants")
G = _build_graph()

res_storey = _assert_ok_action(G, "get_elements_in_storey", {"storey": "Level 0"})
res_storey_prefixed = _assert_ok_action(
    G,
    "get_elements_in_storey",
    {"storey": "Storey::STOREY1"},
)
_assert_ok_action(G, "find_elements_by_class", {"class": "IfcWall"})
res_adj = _assert_ok_action(
    G, "get_adjacent_elements", {"element_id": "Element::DOOR1"}
)
res_topo = _assert_ok_action(
    G,
    "get_topology_neighbors",
    {"element_id": "Element::WALL1", "relation": "above"},
)
_assert_ok_action(G, "get_intersections_3d", {"element_id": "Element::WALL1"})
_assert_ok_action(
    G,
    "find_nodes",
    {"class": "IfcWall", "property_filters": {"Pset_WallCommon.FireRating": "EI 90"}},
)
res_traverse = _assert_ok_action(
    G,
    "traverse",
    {"start": "Storey::STOREY1", "relation": "contains", "depth": 1},
)
res_traverse_contained_in = _assert_ok_action(
    G,
    "traverse",
    {"start": "Storey::STOREY1", "relation": "contained_in", "depth": 1},
)
res_hosts = _assert_ok_action(
    G,
    "traverse",
    {"start": "Element::WALL1", "relation": "hosts", "depth": 1},
)
res_ifc_connected = _assert_ok_action(
    G,
    "traverse",
    {"start": "Element::PIPE1", "relation": "ifc_connected_to", "depth": 1},
)
res_belongs_system = _assert_ok_action(
    G,
    "traverse",
    {"start": "Element::TERMINAL1", "relation": "belongs_to_system", "depth": 1},
)
res_spatial = _assert_ok_action(
    G,
    "spatial_query",
    {"near": "WALL1", "max_distance": 2.0},
)
res_above = _assert_ok_action(
    G, "find_elements_above", {"element_id": "Element::WALL1"}
)
res_below = _assert_ok_action(
    G,
    "find_elements_below",
    {"element_id": "Element::COLUMN1"},
)
_assert_ok_action(G, "get_element_properties", {"element_id": "Element::WALL1"})
res_keys = _assert_ok_action(
    G,
    "list_property_keys",
    {"class": "IfcWall", "sample_values": True},
)
res_keys_echo = _assert_ok_action(
    G,
    "list_property_keys",
    {"class": "Wall", "sample_values": False},
)

print("\n[Contract] Relation taxonomy + source semantics")

adjacent = (res_adj.get("data") or {}).get("adjacent", [])
check(len(adjacent) > 0, "get_adjacent_elements returned at least one neighbor")
adjacent_ok = all(
    item.get("relation") in SPATIAL_RELATIONS and _source_matches_bucket_semantics(item)
    for item in adjacent
)
check(
    adjacent_ok,
    "get_adjacent_elements uses spatial relations with source coerced to contract",
)

topo_neighbors = (res_topo.get("data") or {}).get("neighbors", [])
check(len(topo_neighbors) > 0, "get_topology_neighbors returned at least one neighbor")
topo_ok = all(
    item.get("relation") in TOPOLOGY_RELATIONS
    and _source_matches_bucket_semantics(item)
    for item in topo_neighbors
)
check(
    topo_ok,
    "get_topology_neighbors uses topology relations with source coerced",
)

spatial_results = (res_spatial.get("data") or {}).get("results", [])
check(len(spatial_results) > 0, "spatial_query returned at least one result")
spatial_ok = all(
    item.get("relation") in SPATIAL_RELATIONS and _source_matches_bucket_semantics(item)
    for item in spatial_results
)
check(spatial_ok, "spatial_query uses spatial relations with source coerced")

above_results = (res_above.get("data") or {}).get("results", [])
below_results = (res_below.get("data") or {}).get("results", [])
check(len(above_results) > 0, "find_elements_above returned at least one result")
check(len(below_results) > 0, "find_elements_below returned at least one result")
check(
    all(
        item.get("relation") == "above" and _source_matches_bucket_semantics(item)
        for item in above_results
    ),
    "find_elements_above relation/source semantics follow taxonomy",
)
check(
    all(
        item.get("relation") == "below" and _source_matches_bucket_semantics(item)
        for item in below_results
    ),
    "find_elements_below relation/source semantics follow taxonomy",
)

traverse_results = (res_traverse.get("data") or {}).get("results", [])
check(len(traverse_results) > 0, "traverse contains fixture returned edges")
check(
    all(item.get("relation") == "contains" for item in traverse_results),
    "traverse(relation='contains') keeps strict contains direction",
)
traverse_rel_ok = all(
    normalize_relation_name(item.get("relation")) in CANONICAL_RELATION_SET
    for item in traverse_results
)
check(traverse_rel_ok, "traverse emits canonical taxonomy relations")
check(
    all(_source_matches_bucket_semantics(item) for item in traverse_results),
    "traverse enforces bucket-based source coercion (hierarchy -> null)",
)

traverse_contained_results = (res_traverse_contained_in.get("data") or {}).get(
    "results", []
)
check(
    len(traverse_contained_results) > 0,
    "traverse contained_in fixture returned edges",
)
check(
    all(item.get("relation") == "contained_in" for item in traverse_contained_results),
    "traverse(relation='contained_in') keeps strict contained_in direction",
)
contained_targets = {item.get("to") for item in traverse_contained_results}
check(
    contained_targets == {"IfcBuilding"},
    "traverse(relation='contained_in') targets only intended container node(s)",
)
check(
    all(
        not str(item.get("to", "")).startswith("Element::")
        for item in traverse_contained_results
    ),
    "traverse(relation='contained_in') has no element-node leakage",
)

hosts_results = (res_hosts.get("data") or {}).get("results", [])
check(len(hosts_results) > 0, "traverse hosts fixture returned edges")
hosts_ok = all(
    item.get("relation") == "hosts" and item.get("source") == "ifc"
    for item in hosts_results
)
check(hosts_ok, "traverse emits hosts edges with source='ifc'")

ifc_connected_results = (res_ifc_connected.get("data") or {}).get("results", [])
check(
    len(ifc_connected_results) > 0,
    "traverse ifc_connected_to fixture returned edges",
)
ifc_connected_ok = all(
    item.get("relation") == "ifc_connected_to" and item.get("source") == "ifc"
    for item in ifc_connected_results
)
check(
    ifc_connected_ok,
    "traverse emits ifc_connected_to edges with source='ifc'",
)

belongs_system_results = (res_belongs_system.get("data") or {}).get("results", [])
check(
    len(belongs_system_results) > 0,
    "traverse belongs_to_system fixture returned edges",
)
belongs_system_ok = all(
    item.get("relation") == "belongs_to_system" and item.get("source") == "ifc"
    for item in belongs_system_results
)
check(
    belongs_system_ok,
    "traverse emits belongs_to_system edges with source='ifc'",
)

explicit_relations_ok = all(
    item.get("relation") in EXPLICIT_IFC_RELATIONS
    for item in (hosts_results + ifc_connected_results + belongs_system_results)
)
check(explicit_relations_ok, "explicit IFC relation outputs stay canonical")
check(
    adjacent_ok and topo_ok and hosts_ok and ifc_connected_ok and belongs_system_ok,
    "source/relation mismatches are normalized to contract semantics",
)

# Additional edge-level sanity: non-hierarchy edges should have known source.
non_hierarchy_sources_ok = True
for _, _, edge in G.edges(data=True):
    relation = normalize_relation_name(edge.get("relation"))
    source = normalize_relation_source(edge.get("source"))
    if relation in HIERARCHY_RELATIONS:
        continue
    if source not in KNOWN_RELATION_SOURCES:
        non_hierarchy_sources_ok = False
        break
check(
    non_hierarchy_sources_ok, "non-hierarchy graph edges carry known source semantics"
)

print("\n[Contract] list_property_keys + error-path checks")
key_data = res_keys.get("data") or {}
keys = set(key_data.get("keys") or [])
check("GlobalId" in keys, "list_property_keys includes flat keys")
check(
    "Pset_WallCommon.FireRating" in keys,
    "list_property_keys includes dotted keys",
)
check("samples" in key_data, "list_property_keys returns samples when requested")

key_echo_data = res_keys_echo.get("data") or {}
check(
    key_echo_data.get("class_filter") == "IfcWall",
    "list_property_keys returns normalized class_filter",
)
check(
    key_echo_data.get("class_filter_raw") == "Wall",
    "list_property_keys preserves raw class filter input",
)

unknown = query_ifc_graph(G, "unknown_action", {})
check(has_valid_envelope_shape(unknown), "unknown action still uses stable envelope")
check(unknown.get("status") == "error", "unknown action returns status=error")
check(
    ((unknown.get("error") or {}).get("code") == "unknown_action"),
    "unknown action returns code='unknown_action'",
)

bad_topology = query_ifc_graph(
    G,
    "get_topology_neighbors",
    {"element_id": "Element::WALL1", "relation": "diagonal_to"},
)
check(bad_topology.get("status") == "error", "invalid topology relation returns error")
check(
    ((bad_topology.get("error") or {}).get("code") == "invalid"),
    "invalid topology relation returns code='invalid'",
)

bad_traverse = query_ifc_graph(
    G,
    "traverse",
    {"start": "Storey::STOREY1", "relation": "diagonal_to", "depth": 1},
)
check(bad_traverse.get("status") == "error", "invalid traverse relation returns error")
check(
    ((bad_traverse.get("error") or {}).get("code") == "invalid"),
    "invalid traverse relation returns code='invalid'",
)
bad_traverse_details = (bad_traverse.get("error") or {}).get("details") or {}
allowed_relations = bad_traverse_details.get("allowed_relations") or []
check(
    sorted(allowed_relations) == sorted(CANONICAL_RELATION_SET),
    "invalid traverse relation returns canonical allowed_relations list",
)

bad_props_type = query_ifc_graph(
    G,
    "get_element_properties",
    {"element_id": ["Element::WALL1"]},
)
check(
    has_valid_envelope_shape(bad_props_type),
    "get_element_properties invalid type keeps stable envelope",
)
check(
    bad_props_type.get("status") == "error",
    "get_element_properties invalid type returns status=error",
)
check(
    ((bad_props_type.get("error") or {}).get("code") == "invalid"),
    "get_element_properties invalid type returns code='invalid'",
)

check(
    len((res_storey.get("data") or {}).get("elements", [])) >= 1,
    "get_elements_in_storey returned at least one element",
)
check(
    len((res_storey_prefixed.get("data") or {}).get("elements", [])) >= 1,
    "get_elements_in_storey accepts canonical prefixed storey id",
)
storey_ids = {
    item.get("id") for item in ((res_storey.get("data") or {}).get("elements") or [])
}
check(
    "Element::WALL2" not in storey_ids,
    "get_elements_in_storey excludes sibling-storey elements",
)

print()
if _failures:
    print(f"\033[31m{len(_failures)} check(s) FAILED:\033[0m")
    for failure in _failures:
        print(f"  - {failure}")
    sys.exit(1)
else:
    print("\033[32mAll checks passed.\033[0m")
