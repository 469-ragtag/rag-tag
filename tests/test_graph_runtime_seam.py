from __future__ import annotations

import networkx as nx

from rag_tag.graph import GraphRuntime, wrap_networkx_graph


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
        "Element::D1",
        "Element::W1",
        relation="adjacent_to",
        distance=0.5,
        source="heuristic",
    )
    return G


def test_graph_runtime_networkx_smoke() -> None:
    G = _build_graph()
    runtime = wrap_networkx_graph(G)

    res_storey = runtime.query("get_elements_in_storey", {"storey": "Level 0"})
    assert res_storey["status"] == "ok"
    assert len(res_storey["data"]["elements"]) == 2

    res_class = runtime.query("find_elements_by_class", {"class": "IfcWall"})
    assert res_class["status"] == "ok"
    assert len(res_class["data"]["elements"]) == 1

    res_adj = runtime.query("get_adjacent_elements", {"element_id": "Element::W1"})
    assert res_adj["status"] == "ok"
    assert len(res_adj["data"]["adjacent"]) >= 1

    res_traverse = runtime.query(
        "traverse", {"start": "Storey::S1", "relation": "contains", "depth": 1}
    )
    assert res_traverse["status"] == "ok"
    assert len(res_traverse["data"]["results"]) == 2
