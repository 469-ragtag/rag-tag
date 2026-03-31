from __future__ import annotations

import networkx as nx

from rag_tag.graph_contract import (
    has_valid_envelope_shape,
    missing_required_action_fields,
)
from rag_tag.ifc_graph_tool import query_ifc_graph


def _build_batch2_graph() -> nx.MultiDiGraph:
    graph = nx.MultiDiGraph()
    graph.add_node(
        "Element::space-101",
        label="Room 101",
        class_="IfcSpace",
        properties={"GlobalId": "SPACE101"},
    )
    graph.add_node(
        "Element::pipe-a",
        label="Pipe A",
        class_="IfcPipeSegment",
        properties={"GlobalId": "PIPEA"},
    )
    graph.add_node(
        "Element::pipe-b",
        label="Pipe B",
        class_="IfcPipeSegment",
        properties={"GlobalId": "PIPEB"},
    )
    graph.add_node(
        "Element::terminal-1",
        label="Diffuser 1",
        class_="IfcFlowTerminal",
        properties={"GlobalId": "TERM1"},
    )
    graph.add_node(
        "Element::ahu-1",
        label="AHU 1",
        class_="IfcUnitaryEquipment",
        properties={"GlobalId": "AHU1"},
    )
    graph.add_node(
        "System::Supply Air",
        label="Supply Air",
        class_="IfcSystem",
        node_kind="context",
    )
    graph.add_node(
        "Classification::Pr_70_70_63",
        label="Pr_70_70_63",
        class_="IfcClassificationReference",
        node_kind="context",
    )

    graph.add_edge(
        "Element::space-101",
        "Element::terminal-1",
        relation="contains",
    )
    graph.add_edge(
        "Element::pipe-a",
        "Element::pipe-b",
        relation="ifc_connected_to",
        source="ifc",
    )
    graph.add_edge(
        "Element::pipe-b",
        "Element::pipe-a",
        relation="ifc_connected_to",
        source="ifc",
    )
    graph.add_edge(
        "Element::pipe-b",
        "Element::terminal-1",
        relation="ifc_connected_to",
        source="ifc",
    )
    graph.add_edge(
        "Element::terminal-1",
        "Element::pipe-b",
        relation="ifc_connected_to",
        source="ifc",
    )
    graph.add_edge(
        "Element::terminal-1",
        "System::Supply Air",
        relation="belongs_to_system",
        source="ifc",
    )
    graph.add_edge(
        "Element::ahu-1",
        "System::Supply Air",
        relation="belongs_to_system",
        source="ifc",
    )
    graph.add_edge(
        "Element::ahu-1",
        "Classification::Pr_70_70_63",
        relation="classified_as",
        source="ifc",
    )
    graph.add_edge(
        "Element::pipe-a",
        "Classification::Pr_70_70_63",
        relation="classified_as",
        source="ifc",
    )
    return graph


def _build_directional_topology_graph() -> nx.MultiDiGraph:
    graph = nx.MultiDiGraph()
    graph.add_node(
        "Element::wall-1",
        label="Plumbing Wall",
        class_="IfcWall",
        properties={"GlobalId": "WALL1"},
    )
    graph.add_node(
        "Element::sand-1",
        label="Sand Bedding",
        class_="IfcGeotechnicalStratum",
        properties={"GlobalId": "SAND1"},
    )
    graph.add_node(
        "Element::slab-1",
        label="Slab 1",
        class_="IfcSlab",
        properties={"GlobalId": "SLAB1"},
    )

    graph.add_edge(
        "Element::wall-1",
        "Element::sand-1",
        relation="above",
        source="topology",
        vertical_gap=0.15,
    )
    graph.add_edge(
        "Element::sand-1",
        "Element::wall-1",
        relation="below",
        source="topology",
        vertical_gap=0.15,
    )
    graph.add_edge(
        "Element::wall-1",
        "Element::slab-1",
        relation="intersects_bbox",
        source="topology",
    )
    graph.add_edge(
        "Element::slab-1",
        "Element::wall-1",
        relation="intersects_bbox",
        source="topology",
    )
    return graph


def test_trace_distribution_network_returns_bounded_grounded_results() -> None:
    graph = _build_batch2_graph()

    result = query_ifc_graph(
        graph,
        "trace_distribution_network",
        {
            "start": "Element::pipe-a",
            "max_depth": 2,
            "relations": ["ifc_connected_to"],
            "max_results": 10,
        },
    )

    assert has_valid_envelope_shape(result)
    assert result["status"] == "ok"
    assert not missing_required_action_fields(
        "trace_distribution_network", result["data"]
    )
    data = result["data"]
    assert [item["id"] for item in data["results"]] == [
        "Element::pipe-b",
        "Element::terminal-1",
    ]
    assert data["results"][1]["path"][-1]["id"] == "Element::terminal-1"
    assert data["results"][1]["via_relation"] == "ifc_connected_to"
    assert data["evidence"][0]["id"] == "Element::pipe-a"
    assert data["evidence"][0]["match_reason"] == "start_node"


def test_trace_distribution_network_orders_same_depth_results_deterministically() -> (
    None
):
    graph = nx.MultiDiGraph()
    graph.add_node(
        "Element::root",
        label="Root",
        class_="IfcPipeSegment",
        properties={"GlobalId": "ROOT"},
    )
    graph.add_node(
        "Element::zulu",
        label="Zulu branch",
        class_="IfcPipeSegment",
        properties={"GlobalId": "ZULU"},
    )
    graph.add_node(
        "Element::alpha",
        label="Alpha branch",
        class_="IfcPipeSegment",
        properties={"GlobalId": "ALPHA"},
    )
    graph.add_edge(
        "Element::root",
        "Element::zulu",
        relation="ifc_connected_to",
        source="ifc",
    )
    graph.add_edge(
        "Element::root",
        "Element::alpha",
        relation="ifc_connected_to",
        source="ifc",
    )

    result = query_ifc_graph(
        graph,
        "trace_distribution_network",
        {
            "start": "Element::root",
            "max_depth": 1,
            "relations": ["ifc_connected_to"],
        },
    )

    assert result["status"] == "ok"
    assert [item["id"] for item in result["data"]["results"]] == [
        "Element::alpha",
        "Element::zulu",
    ]


def test_trace_distribution_network_exposes_truncation_metadata() -> None:
    graph = nx.MultiDiGraph()
    graph.add_node(
        "Element::root",
        label="Root",
        class_="IfcPipeSegment",
        properties={"GlobalId": "ROOT"},
    )
    graph.add_node(
        "Element::zulu",
        label="Zulu branch",
        class_="IfcPipeSegment",
        properties={"GlobalId": "ZULU"},
    )
    graph.add_node(
        "Element::alpha",
        label="Alpha branch",
        class_="IfcPipeSegment",
        properties={"GlobalId": "ALPHA"},
    )
    graph.add_edge(
        "Element::root",
        "Element::zulu",
        relation="ifc_connected_to",
        source="ifc",
    )
    graph.add_edge(
        "Element::root",
        "Element::alpha",
        relation="ifc_connected_to",
        source="ifc",
    )

    result = query_ifc_graph(
        graph,
        "trace_distribution_network",
        {
            "start": "Element::root",
            "max_depth": 1,
            "relations": ["ifc_connected_to"],
            "max_results": 1,
        },
    )

    assert result["status"] == "ok"
    data = result["data"]
    assert [item["id"] for item in data["results"]] == ["Element::alpha"]
    assert data["results"][0]["path"][-1]["id"] == "Element::alpha"
    assert data["visited_count"] == 2
    assert data["total_found"] == 2
    assert data["returned_count"] == 1
    assert data["truncated"] is True
    assert data["truncation_reason"] == (
        "Results truncated to 1 item(s) to stay bounded."
    )
    assert data["warnings"] == [data["truncation_reason"]]


def test_find_shortest_path_returns_ordered_path_and_steps() -> None:
    graph = _build_batch2_graph()

    result = query_ifc_graph(
        graph,
        "find_shortest_path",
        {
            "start": "PIPEA",
            "end": "AHU1",
            "relations": ["ifc_connected_to", "belongs_to_system"],
            "max_path_length": 6,
        },
    )

    assert has_valid_envelope_shape(result)
    assert result["status"] == "ok"
    assert not missing_required_action_fields("find_shortest_path", result["data"])
    data = result["data"]
    assert [item["id"] for item in data["path"]] == [
        "Element::pipe-a",
        "Element::pipe-b",
        "Element::terminal-1",
        "System::Supply Air",
        "Element::ahu-1",
    ]
    assert [step["relation"] for step in data["steps"]] == [
        "ifc_connected_to",
        "ifc_connected_to",
        "belongs_to_system",
        "belongs_to_system",
    ]
    assert data["path_length"] == 4
    assert data["evidence"][0] == {
        "global_id": "PIPEA",
        "id": "Element::pipe-a",
        "label": "Pipe A",
        "class_": "IfcPipeSegment",
        "source_tool": "find_shortest_path",
        "match_reason": "start_node",
    }
    assert data["evidence"][1] == {
        "global_id": "AHU1",
        "id": "Element::ahu-1",
        "label": "AHU 1",
        "class_": "IfcUnitaryEquipment",
        "source_tool": "find_shortest_path",
        "match_reason": "end_node",
    }
    assert any(
        item
        == {
            "global_id": "PIPEB",
            "id": "Element::pipe-b",
            "label": "Pipe B",
            "class_": "IfcPipeSegment",
            "source_tool": "find_shortest_path",
            "match_reason": "step=1",
        }
        for item in data["evidence"]
    )


def test_find_by_classification_matches_normalized_label_and_context_evidence() -> None:
    graph = _build_batch2_graph()

    result = query_ifc_graph(
        graph,
        "find_by_classification",
        {"classification": "pr 70 70 63", "max_results": 10},
    )

    assert has_valid_envelope_shape(result)
    assert result["status"] == "ok"
    assert not missing_required_action_fields("find_by_classification", result["data"])
    data = result["data"]
    assert [item["id"] for item in data["elements"]] == [
        "Element::ahu-1",
        "Element::pipe-a",
    ]
    assert data["matched_classifications"][0]["id"] == "Classification::Pr_70_70_63"
    assert any(
        item.get("id") == "Classification::Pr_70_70_63" for item in data["evidence"]
    )
    assert any(
        item.get("id") == "Classification::Pr_70_70_63"
        and item.get("match_reason") == "normalized_label"
        and item.get("source_tool") == "find_by_classification"
        for item in data["evidence"]
    )


def test_find_by_classification_exposes_truncation_metadata() -> None:
    graph = nx.MultiDiGraph()
    graph.add_node(
        "Classification::Pr_70_70_63",
        label="Pr_70_70_63",
        class_="IfcClassificationReference",
        node_kind="context",
    )
    for node_id, label, global_id in [
        ("Element::ahu-a", "AHU Alpha", "AHUA"),
        ("Element::ahu-b", "AHU Bravo", "AHUB"),
        ("Element::ahu-c", "AHU Charlie", "AHUC"),
    ]:
        graph.add_node(
            node_id,
            label=label,
            class_="IfcUnitaryEquipment",
            properties={"GlobalId": global_id},
        )
        graph.add_edge(
            node_id,
            "Classification::Pr_70_70_63",
            relation="classified_as",
            source="ifc",
        )

    result = query_ifc_graph(
        graph,
        "find_by_classification",
        {"classification": "pr 70 70 63", "max_results": 2},
    )

    assert result["status"] == "ok"
    data = result["data"]
    assert [item["id"] for item in data["elements"]] == [
        "Element::ahu-a",
        "Element::ahu-b",
    ]
    assert data["matched_classifications"] == [
        {
            "id": "Classification::Pr_70_70_63",
            "global_id": None,
            "label": "Pr_70_70_63",
            "class_": "IfcClassificationReference",
            "match_reason": "normalized_label",
            "match_score": 2,
        }
    ]
    assert data["total"] == 3
    assert data["total_found"] == 3
    assert data["returned_count"] == 2
    assert data["truncated"] is True
    assert data["truncation_reason"] == (
        "Results truncated to 2 item(s) to stay bounded."
    )
    assert data["warnings"] == [data["truncation_reason"]]


def test_find_equipment_serving_space_finds_upstream_equipment() -> None:
    graph = _build_batch2_graph()

    result = query_ifc_graph(
        graph,
        "find_equipment_serving_space",
        {"space": "Room 101", "max_depth": 4, "max_results": 5},
    )

    assert has_valid_envelope_shape(result)
    assert result["status"] == "ok"
    assert not missing_required_action_fields(
        "find_equipment_serving_space", result["data"]
    )
    data = result["data"]
    assert data["space"] == "Element::space-101"
    assert data["equipment"][0]["id"] == "Element::ahu-1"
    assert data["equipment"][0]["candidate_type"] == "equipment"
    assert data["equipment"][0]["support_strength"] in {"strong", "moderate"}
    assert [item["id"] for item in data["equipment"][0]["path"]] == [
        "Element::space-101",
        "Element::terminal-1",
        "System::Supply Air",
        "Element::ahu-1",
    ]
    assert data["evidence"][0] == {
        "global_id": "SPACE101",
        "id": "Element::space-101",
        "label": "Room 101",
        "class_": "IfcSpace",
        "source_tool": "find_equipment_serving_space",
        "match_reason": "space_anchor",
    }


def test_find_equipment_serving_space_exposes_truncation_metadata() -> None:
    graph = nx.MultiDiGraph()
    graph.add_node(
        "Element::space-401",
        label="Room 401",
        class_="IfcSpace",
        properties={"GlobalId": "SPACE401"},
    )
    for node_id, label, global_id in [
        ("Element::ahu-a", "AHU Alpha", "AHUA"),
        ("Element::ahu-b", "AHU Bravo", "AHUB"),
        ("Element::ahu-c", "AHU Charlie", "AHUC"),
    ]:
        graph.add_node(
            node_id,
            label=label,
            class_="IfcUnitaryEquipment",
            properties={"GlobalId": global_id},
        )
        graph.add_edge("Element::space-401", node_id, relation="contains")

    result = query_ifc_graph(
        graph,
        "find_equipment_serving_space",
        {"space": "Element::space-401", "max_results": 2},
    )

    assert result["status"] == "ok"
    data = result["data"]
    assert [item["id"] for item in data["equipment"]] == [
        "Element::ahu-a",
        "Element::ahu-b",
    ]
    assert [item["candidate_type"] for item in data["equipment"]] == [
        "equipment",
        "equipment",
    ]
    assert [item["path"][-1]["id"] for item in data["equipment"]] == [
        "Element::ahu-a",
        "Element::ahu-b",
    ]
    assert data["seed_count"] == 3
    assert data["total_found"] == 3
    assert data["returned_count"] == 2
    assert data["truncated"] is True
    assert data["truncation_reason"] == (
        "Results truncated to 2 item(s) to stay bounded."
    )
    assert data["warnings"] == [data["truncation_reason"]]


def test_find_equipment_serving_space_returns_terminal_fallback_warning() -> None:
    graph = nx.MultiDiGraph()
    graph.add_node(
        "Element::space-201",
        label="Room 201",
        class_="IfcSpace",
        properties={"GlobalId": "SPACE201"},
    )
    graph.add_node(
        "Element::terminal-201",
        label="Diffuser 201",
        class_="IfcFlowTerminal",
        properties={"GlobalId": "TERM201"},
    )
    graph.add_edge(
        "Element::space-201",
        "Element::terminal-201",
        relation="contains",
    )

    result = query_ifc_graph(
        graph,
        "find_equipment_serving_space",
        {"space": "Element::space-201"},
    )

    assert result["status"] == "ok"
    data = result["data"]
    assert data["equipment"][0]["id"] == "Element::terminal-201"
    assert data["equipment"][0]["candidate_type"] == "terminal"
    assert any("returning terminal-level" in warning for warning in data["warnings"])


def test_find_equipment_serving_space_replaces_weaker_path_with_stronger_one() -> None:
    graph = nx.MultiDiGraph()
    graph.add_node(
        "Element::space-301",
        label="Room 301",
        class_="IfcSpace",
        properties={"GlobalId": "SPACE301"},
    )
    graph.add_node(
        "Element::ahu-301",
        label="AHU 301",
        class_="IfcUnitaryEquipment",
        properties={"GlobalId": "AHU301"},
    )
    graph.add_node(
        "Element::terminal-301",
        label="Diffuser 301",
        class_="IfcFlowTerminal",
        properties={"GlobalId": "TERM301"},
    )

    graph.add_edge(
        "Element::space-301",
        "Element::ahu-301",
        relation="contains",
    )
    graph.add_edge(
        "Element::space-301",
        "Element::terminal-301",
        relation="contains",
    )
    graph.add_edge(
        "Element::terminal-301",
        "Element::ahu-301",
        relation="ifc_connected_to",
        source="ifc",
    )

    result = query_ifc_graph(
        graph,
        "find_equipment_serving_space",
        {"space": "Element::space-301", "max_depth": 3, "max_results": 5},
    )

    assert result["status"] == "ok"
    equipment = result["data"]["equipment"]
    assert equipment[0]["id"] == "Element::ahu-301"
    assert equipment[0]["score"] == 7
    assert [item["id"] for item in equipment[0]["path"]] == [
        "Element::space-301",
        "Element::terminal-301",
        "Element::ahu-301",
    ]


def test_directional_topology_filters_do_not_treat_above_below_as_undirected() -> None:
    graph = _build_directional_topology_graph()

    above_neighbors = query_ifc_graph(
        graph,
        "get_topology_neighbors",
        {"element_id": "Element::wall-1", "relation": "above"},
    )
    below_neighbors = query_ifc_graph(
        graph,
        "get_topology_neighbors",
        {"element_id": "Element::wall-1", "relation": "below"},
    )

    assert above_neighbors["status"] == "ok"
    assert below_neighbors["status"] == "ok"
    assert [item["id"] for item in above_neighbors["data"]["neighbors"]] == [
        "Element::sand-1"
    ]
    assert below_neighbors["data"]["neighbors"] == []


def test_find_elements_above_and_below_use_anchor_relative_direction() -> None:
    graph = _build_directional_topology_graph()

    above = query_ifc_graph(
        graph,
        "find_elements_above",
        {"element_id": "Element::wall-1"},
    )
    below = query_ifc_graph(
        graph,
        "find_elements_below",
        {"element_id": "Element::wall-1"},
    )
    intersects = query_ifc_graph(
        graph,
        "get_topology_neighbors",
        {"element_id": "Element::wall-1", "relation": "intersects_bbox"},
    )

    assert above["status"] == "ok"
    assert below["status"] == "ok"
    assert intersects["status"] == "ok"

    assert above["data"]["results"] == []
    assert below["data"]["results"] == [
        {
            "id": "Element::sand-1",
            "global_id": "SAND1",
            "label": "Sand Bedding",
            "class_": "IfcGeotechnicalStratum",
            "relation": "below",
            "vertical_gap": 0.15,
            "source": "topology",
        }
    ]
    assert [item["id"] for item in intersects["data"]["neighbors"]] == [
        "Element::slab-1"
    ]
