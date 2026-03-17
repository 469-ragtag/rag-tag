from __future__ import annotations

import json
from pathlib import Path

import pytest

from rag_tag.ifc_sql_tool import SqlQueryError, query_ifc_sql
from rag_tag.parser.jsonl_to_sql import jsonl_to_sql
from rag_tag.router.models import SqlFieldRef, SqlRequest, SqlValueFilter


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def _build_test_db(tmp_path: Path) -> Path:
    jsonl_path = tmp_path / "rich_queries.jsonl"
    db_path = tmp_path / "rich_queries.db"
    _write_jsonl(
        jsonl_path,
        [
            {
                "ExpressId": 1,
                "GlobalId": "wall-l2-a",
                "IfcType": "IfcWall",
                "Name": "Wall L2 A",
                "Hierarchy": {"Level": "Level 2"},
                "Quantities": {
                    "Qto_WallBaseQuantities": {"NetVolume": 10.0, "NetArea": 12.0}
                },
            },
            {
                "ExpressId": 2,
                "GlobalId": "wall-l2-b",
                "IfcType": "IfcWall",
                "Name": "Wall L2 B",
                "Hierarchy": {"Level": "Level 2"},
                "Quantities": {
                    "Qto_WallBaseQuantities": {"NetVolume": 15.5, "NetArea": 14.0}
                },
            },
            {
                "ExpressId": 3,
                "GlobalId": "wall-ground-a",
                "IfcType": "IfcWall",
                "Name": "Wall Ground A",
                "Hierarchy": {"Level": "Ground Floor"},
                "Quantities": {
                    "Qto_WallBaseQuantities": {"NetVolume": 7.0, "NetArea": 9.0}
                },
            },
            {
                "ExpressId": 4,
                "GlobalId": "door-l1-a",
                "IfcType": "IfcDoor",
                "Name": "Door L1 A",
                "Hierarchy": {"Level": "Level 1"},
                "PropertySets": {
                    "Official": {"Pset_DoorCommon": {"FireRating": "EI30"}}
                },
            },
            {
                "ExpressId": 5,
                "GlobalId": "door-l1-b",
                "IfcType": "IfcDoor",
                "Name": "Door L1 B",
                "Hierarchy": {"Level": "Level 1"},
                "PropertySets": {
                    "Official": {"Pset_DoorCommon": {"FireRating": "EI60"}}
                },
            },
            {
                "ExpressId": 6,
                "GlobalId": "door-l1-c",
                "IfcType": "IfcDoor",
                "Name": "Door L1 C",
                "Hierarchy": {"Level": "Level 1"},
                "PropertySets": {
                    "Official": {"Pset_DoorCommon": {"FireRating": "EI30"}}
                },
            },
            {
                "ExpressId": 7,
                "GlobalId": "window-ground-a",
                "IfcType": "IfcWindow",
                "Name": "Window Ground A",
                "Hierarchy": {"Level": "Ground Floor"},
                "PropertySets": {"Official": {"Pset_WindowCommon": {"UValue": 1.2}}},
            },
            {
                "ExpressId": 8,
                "GlobalId": "window-ground-b",
                "IfcType": "IfcWindow",
                "Name": "Window Ground B",
                "Hierarchy": {"Level": "Ground Floor"},
                "PropertySets": {"Official": {"Pset_WindowCommon": {"UValue": 0.8}}},
            },
            {
                "ExpressId": 9,
                "GlobalId": "window-l2-a",
                "IfcType": "IfcWindow",
                "Name": "Window L2 A",
                "Hierarchy": {"Level": "Level 2"},
                "PropertySets": {"Official": {"Pset_WindowCommon": {"UValue": 1.5}}},
            },
        ],
    )
    jsonl_to_sql(jsonl_path, db_path)
    return db_path


def test_sum_quantity_aggregate_is_deterministic(tmp_path: Path) -> None:
    db_path = _build_test_db(tmp_path)

    result = query_ifc_sql(
        db_path,
        SqlRequest(
            intent="aggregate",
            ifc_class="IfcWall",
            level_like="Level 2",
            aggregate_op="sum",
            aggregate_field=SqlFieldRef(source="quantity", field="NetVolume"),
        ),
    )

    assert result["data"]["aggregate_value"] == 25.5
    assert result["data"]["matched_value_count"] == 2
    assert result["data"]["missing_value_count"] == 0
    assert result["data"]["aggregate_field"] == {
        "source": "quantity",
        "field": "NetVolume",
    }


def test_group_by_property_is_deterministic(tmp_path: Path) -> None:
    db_path = _build_test_db(tmp_path)

    result = query_ifc_sql(
        db_path,
        SqlRequest(
            intent="group",
            ifc_class="IfcDoor",
            level_like="Level 1",
            group_by=SqlFieldRef(source="property", field="FireRating"),
            limit=10,
        ),
    )

    assert result["data"]["groups"] == [
        {"group": "EI30", "count": 2},
        {"group": "EI60", "count": 1},
    ]
    assert result["data"]["matched_element_count"] == 3
    assert result["data"]["missing_value_count"] == 0


def test_avg_property_aggregate_and_filters_work(tmp_path: Path) -> None:
    db_path = _build_test_db(tmp_path)

    aggregate_result = query_ifc_sql(
        db_path,
        SqlRequest(
            intent="aggregate",
            ifc_class="IfcWindow",
            level_like="ground floor",
            aggregate_op="avg",
            aggregate_field=SqlFieldRef(source="property", field="UValue"),
        ),
    )
    assert aggregate_result["data"]["aggregate_value"] == 1.0
    assert aggregate_result["data"]["matched_value_count"] == 2

    filtered_result = query_ifc_sql(
        db_path,
        SqlRequest(
            intent="count",
            ifc_class="IfcWindow",
            level_like="ground floor",
            property_filters=(SqlValueFilter(field="UValue", op="lte", value=1.0),),
        ),
    )
    assert filtered_result["data"]["count"] == 1


def test_quantity_filters_work_for_count_queries(tmp_path: Path) -> None:
    db_path = _build_test_db(tmp_path)

    result = query_ifc_sql(
        db_path,
        SqlRequest(
            intent="count",
            ifc_class="IfcWall",
            level_like="Level 2",
            quantity_filters=(SqlValueFilter(field="NetVolume", op="gte", value=12.0),),
        ),
    )

    assert result["data"]["count"] == 1


def test_non_numeric_property_aggregate_raises_clear_error(tmp_path: Path) -> None:
    db_path = _build_test_db(tmp_path)

    with pytest.raises(SqlQueryError, match="is not numeric"):
        query_ifc_sql(
            db_path,
            SqlRequest(
                intent="aggregate",
                ifc_class="IfcDoor",
                level_like="Level 1",
                aggregate_op="avg",
                aggregate_field=SqlFieldRef(source="property", field="FireRating"),
            ),
        )
