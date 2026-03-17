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
