from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import networkx as nx

from rag_tag.graph import wrap_networkx_graph
from rag_tag.graph_contract import (
    has_valid_envelope_shape,
    missing_required_action_fields,
)
from rag_tag.ifc_graph_tool import query_ifc_graph
from rag_tag.parser.sql_schema import SCHEMA_SQL


def _typed(value: object) -> str:
    return f"json:{json.dumps(value, separators=(',', ':'))}"


def _build_batch3_graph() -> nx.MultiDiGraph:
    graph = nx.MultiDiGraph()
    graph.add_node(
        "Element::wall-a",
        label="Wall A",
        class_="IfcWall",
        properties={
            "GlobalId": "WALLA",
            "ExpressId": 1,
            "Level": "Level 1",
            "Name": "Wall A",
        },
    )
    graph.add_node(
        "Element::wall-b",
        label="Wall B",
        class_="IfcWall",
        properties={
            "GlobalId": "WALLB",
            "ExpressId": 2,
            "Level": "Level 2",
            "Name": "Wall B",
        },
    )
    graph.add_node(
        "Element::door-a",
        label="Door A",
        class_="IfcDoor",
        properties={
            "GlobalId": "DOORA",
            "ExpressId": 3,
            "Level": "Level 1",
            "Name": "Door A",
        },
    )
    return graph


def _write_batch3_db(path: Path) -> None:
    conn = sqlite3.connect(str(path))
    try:
        conn.executescript(SCHEMA_SQL)
        conn.executemany(
            "INSERT INTO elements "
            "(express_id, global_id, ifc_class, predefined_type, name, description, "
            "object_type, tag, level, level_key, type_name) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    1,
                    "WALLA",
                    "IfcWall",
                    None,
                    "Wall A",
                    None,
                    None,
                    None,
                    "Level 1",
                    None,
                    None,
                ),
                (
                    2,
                    "WALLB",
                    "IfcWall",
                    None,
                    "Wall B",
                    None,
                    None,
                    None,
                    "Level 2",
                    None,
                    None,
                ),
                (
                    3,
                    "DOORA",
                    "IfcDoor",
                    None,
                    "Door A",
                    None,
                    None,
                    None,
                    "Level 1",
                    None,
                    None,
                ),
            ],
        )
        conn.executemany(
            "INSERT INTO properties "
            "(element_id, pset_name, property_name, value, is_official) "
            "VALUES (?, ?, ?, ?, ?)",
            [
                (1, "Pset_Custom", "UValue", _typed(0.28), 0),
                (2, "Pset_Custom", "UValue", _typed(0.35), 0),
                (1, "Pset_WallCommon", "FireRating", _typed("EI 60"), 1),
                (2, "Pset_WallCommon", "FireRating", _typed("EI 90"), 1),
            ],
        )
        conn.executemany(
            "INSERT INTO quantities "
            "(element_id, qto_name, quantity_name, value, is_official) "
            "VALUES (?, ?, ?, ?, ?)",
            [
                (1, "Qto_WallBaseQuantities", "NetVolume", 12.5, 1),
                (2, "Qto_WallBaseQuantities", "NetVolume", 7.5, 1),
            ],
        )
        conn.commit()
    finally:
        conn.close()


def _runtime_with_db(tmp_path: Path):
    db_path = tmp_path / "bridge.db"
    _write_batch3_db(db_path)
    return wrap_networkx_graph(_build_batch3_graph(), context_db_path=db_path)


def test_aggregate_elements_counts_exact_ids_and_reports_unmatched(
    tmp_path: Path,
) -> None:
    runtime = _runtime_with_db(tmp_path)

    result = query_ifc_graph(
        runtime,
        "aggregate_elements",
        {
            "element_ids": ["Element::wall-a", "WALLB", "missing-id"],
            "metric": "count",
        },
    )

    assert has_valid_envelope_shape(result)
    assert result["status"] == "ok"
    assert not missing_required_action_fields("aggregate_elements", result["data"])
    data = result["data"]
    assert data["aggregate_value"] == 2
    assert data["matched_element_count"] == 2
    assert data["unmatched_element_count"] == 1
    assert data["unmatched_element_ids"] == ["missing-id"]
    assert data["evidence"][0]["global_id"] == "WALLA"


def test_aggregate_elements_supports_quantity_sum(tmp_path: Path) -> None:
    runtime = _runtime_with_db(tmp_path)

    result = query_ifc_graph(
        runtime,
        "aggregate_elements",
        {
            "element_ids": ["Element::wall-a", "Element::wall-b"],
            "metric": "sum",
            "field": "Qto_WallBaseQuantities.NetVolume",
        },
    )

    assert result["status"] == "ok"
    data = result["data"]
    assert data["field_source"] == "quantity"
    assert data["aggregate_value"] == 20.0
    assert data["missing_value_count"] == 0
    assert data["sample"][0]["field_value"] == 12.5


def test_aggregate_elements_supports_property_average(tmp_path: Path) -> None:
    runtime = _runtime_with_db(tmp_path)

    result = query_ifc_graph(
        runtime,
        "aggregate_elements",
        {
            "element_ids": ["WALLA", "WALLB"],
            "metric": "avg",
            "field": "Pset_Custom.UValue",
        },
    )

    assert result["status"] == "ok"
    data = result["data"]
    assert data["field_source"] == "property"
    assert data["aggregate_value"] == 0.315


def test_group_elements_by_property_groups_by_core_column(tmp_path: Path) -> None:
    runtime = _runtime_with_db(tmp_path)

    result = query_ifc_graph(
        runtime,
        "group_elements_by_property",
        {
            "element_ids": ["Element::wall-a", "Element::wall-b", "Element::door-a"],
            "property_key": "Level",
        },
    )

    assert has_valid_envelope_shape(result)
    assert result["status"] == "ok"
    assert not missing_required_action_fields(
        "group_elements_by_property", result["data"]
    )
    data = result["data"]
    assert data["field_source"] == "core"
    assert [(group["value"], group["count"]) for group in data["groups"]] == [
        ("Level 1", 2),
        ("Level 2", 1),
    ]
    assert data["groups"][0]["sample"][0]["field_value"] == "Level 1"
    assert data["evidence"][0]["global_id"] == "WALLA"


def test_bridge_tools_require_sqlite_context_db() -> None:
    runtime = wrap_networkx_graph(_build_batch3_graph())

    result = query_ifc_graph(
        runtime,
        "aggregate_elements",
        {
            "element_ids": ["Element::wall-a"],
            "metric": "count",
        },
    )

    assert result["status"] == "error"
    assert result["error"]["code"] == "missing_context_db"


def test_aggregate_elements_rejects_non_numeric_metric_on_text_field(
    tmp_path: Path,
) -> None:
    runtime = _runtime_with_db(tmp_path)

    result = query_ifc_graph(
        runtime,
        "aggregate_elements",
        {
            "element_ids": ["Element::wall-a", "Element::wall-b"],
            "metric": "avg",
            "field": "Pset_WallCommon.FireRating",
        },
    )

    assert result["status"] == "error"
    assert result["error"]["code"] == "invalid"
    assert "requires numeric values" in result["error"]["message"]
