from __future__ import annotations

import os

import networkx as nx
import pytest

from rag_tag.graph.backends.neo4j_backend import Neo4jBackend
from rag_tag.graph_contract import CANONICAL_ACTION_SET
from rag_tag.ifc_graph_tool import query_ifc_graph
from rag_tag.parser.jsonl_to_neo4j import import_networkx_graph

try:
    from neo4j import GraphDatabase  # noqa: F401
except Exception:
    GraphDatabase = None


NEO4J_ENV = all(
    os.environ.get(k) for k in ("NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD")
)


@pytest.mark.skipif(GraphDatabase is None, reason="neo4j package not installed")
@pytest.mark.skipif(not NEO4J_ENV, reason="Neo4j env not configured")
def test_neo4j_parity_smoke() -> None:
    G = _build_graph()
    import_networkx_graph(G, batch_size=500)

    backend = Neo4jBackend()

    for action in CANONICAL_ACTION_SET:
        params = _default_params(action)
        if params is None:
            continue
        nx_result = query_ifc_graph(G, action, params)
        neo_result = backend.query(action, params)
        assert nx_result["status"] == neo_result["status"]

        if action == "find_elements_by_class":
            assert _ids(nx_result["data"]["elements"]) == _ids(
                neo_result["data"]["elements"]
            )
        if action == "get_elements_in_storey":
            assert _ids(nx_result["data"]["elements"]) == _ids(
                neo_result["data"]["elements"]
            )
        if action == "get_adjacent_elements":
            assert _ids(nx_result["data"]["adjacent"]) == _ids(
                neo_result["data"]["adjacent"]
            )
        if action == "get_topology_neighbors":
            assert _ids(nx_result["data"]["neighbors"]) == _ids(
                neo_result["data"]["neighbors"]
            )
        if action == "get_intersections_3d":
            assert _ids(nx_result["data"]["intersections_3d"]) == _ids(
                neo_result["data"]["intersections_3d"]
            )
        if action == "traverse":
            assert _ids([r["node"] for r in nx_result["data"]["results"]]) == _ids(
                [r["node"] for r in neo_result["data"]["results"]]
            )

    backend.close()


def _build_graph() -> nx.MultiDiGraph:
    G = nx.MultiDiGraph()
    G.add_node(
        "IfcBuilding",
        label="Building A",
        class_="IfcBuilding",
        properties={"GlobalId": "BLDG1", "Name": "Building A"},
        payload={},
    )
    G.add_node(
        "Storey::S1",
        label="Level 0",
        class_="IfcBuildingStorey",
        properties={"GlobalId": "S1", "Name": "Level 0"},
        payload={},
    )
    G.add_node(
        "Element::W1",
        label="Wall 1",
        class_="IfcWall",
        properties={"GlobalId": "W1", "Name": "Wall 1"},
        payload={"PropertySets": {"Official": {}, "Custom": {}}},
    )
    G.add_node(
        "Element::D1",
        label="Door 1",
        class_="IfcDoor",
        properties={"GlobalId": "D1", "Name": "Door 1"},
        payload={"PropertySets": {"Official": {}, "Custom": {}}},
    )
    G.add_node(
        "Element::S1",
        label="Slab 1",
        class_="IfcSlab",
        properties={"GlobalId": "S1", "Name": "Slab 1"},
        payload={"PropertySets": {"Official": {}, "Custom": {}}},
    )

    G.add_edge("IfcBuilding", "Storey::S1", relation="contains")
    G.add_edge("Storey::S1", "IfcBuilding", relation="contained_in")
    G.add_edge("Storey::S1", "Element::W1", relation="contains")
    G.add_edge("Element::W1", "Storey::S1", relation="contained_in")
    G.add_edge("Storey::S1", "Element::D1", relation="contains")
    G.add_edge("Element::D1", "Storey::S1", relation="contained_in")

    G.add_edge(
        "Element::W1",
        "Element::D1",
        relation="adjacent_to",
        distance=0.5,
        source="heuristic",
    )
    G.add_edge(
        "Element::W1",
        "Element::S1",
        relation="above",
        vertical_gap=0.25,
        source="topology",
    )
    G.add_edge(
        "Element::W1",
        "Element::S1",
        relation="intersects_3d",
        intersection_volume=1.0,
        contact_area=0.5,
        source="topology",
    )
    return G


def _default_params(action: str) -> dict | None:
    if action == "get_elements_in_storey":
        return {"storey": "Level 0"}
    if action == "find_elements_by_class":
        return {"class": "IfcWall"}
    if action == "get_adjacent_elements":
        return {"element_id": "Element::W1"}
    if action == "get_topology_neighbors":
        return {"element_id": "Element::W1", "relation": "above"}
    if action == "get_intersections_3d":
        return {"element_id": "Element::W1"}
    if action == "traverse":
        return {"start": "Storey::S1", "relation": "contains", "depth": 1}
    if action == "find_nodes":
        return {"class": "IfcWall", "property_filters": {"Name": "Wall 1"}}
    if action == "list_property_keys":
        return {"class": "IfcWall", "sample_values": False}
    if action == "get_element_properties":
        return {"element_id": "Element::W1"}
    if action == "spatial_query":
        return {"near": "Element::W1", "max_distance": 1.0}
    if action == "find_elements_above":
        return {"element_id": "Element::W1"}
    if action == "find_elements_below":
        return {"element_id": "Element::S1"}
    return None


def _ids(items: list) -> list[str]:
    ids: list[str] = []
    for item in items:
        if isinstance(item, dict):
            if "id" in item:
                ids.append(item["id"])
        elif isinstance(item, str):
            ids.append(item)
    return sorted(ids)
