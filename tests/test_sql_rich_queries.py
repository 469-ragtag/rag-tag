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
                "TypeName": "Generic Wall A",
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
                "TypeName": "Generic Wall B",
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
                "TypeName": "Curtain Wall:Exterior Curtain Wall",
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
                "TypeName": "Single Flush 900 x 2100",
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
                "TypeName": "Double Glass 1800 x 2100",
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
                "TypeName": "Single Flush 900 x 2100",
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
                "TypeName": "Fixed 1200 x 1500",
                "Hierarchy": {"Level": "Ground Floor"},
                "PropertySets": {"Official": {"Pset_WindowCommon": {"UValue": 1.2}}},
            },
            {
                "ExpressId": 8,
                "GlobalId": "window-ground-b",
                "IfcType": "IfcWindow",
                "Name": "Window Ground B",
                "TypeName": "Sliding 1800 x 1500",
                "Hierarchy": {"Level": "Ground Floor"},
                "PropertySets": {"Official": {"Pset_WindowCommon": {"UValue": 0.8}}},
            },
            {
                "ExpressId": 9,
                "GlobalId": "window-l2-a",
                "IfcType": "IfcWindow",
                "Name": "Window L2 A",
                "TypeName": "Fixed 1200 x 1500",
                "Hierarchy": {"Level": "Level 2"},
                "PropertySets": {"Official": {"Pset_WindowCommon": {"UValue": 1.5}}},
            },
            {
                "ExpressId": 10,
                "GlobalId": "curtain-wall-ground-b",
                "IfcType": "IfcWall",
                "Name": "Curtain Wall Ground B",
                "TypeName": "Curtain Wall:Exterior Curtain Wall",
                "Hierarchy": {"Level": "Ground Floor"},
            },
            {
                "ExpressId": 11,
                "GlobalId": "curtain-wall-ground-c",
                "IfcType": "IfcWall",
                "Name": "Curtain Wall Ground C",
                "TypeName": "Curtain Wall:Storefront",
                "Hierarchy": {"Level": "Ground Floor"},
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
            property_filters=(
                SqlValueFilter(
                    source="property",
                    field="UValue",
                    op="lte",
                    value=1.0,
                ),
            ),
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
            quantity_filters=(
                SqlValueFilter(
                    source="quantity",
                    field="NetVolume",
                    op="gte",
                    value=12.0,
                ),
            ),
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


def test_element_name_filter_uses_elements_column_and_excludes_type_rows(
    tmp_path: Path,
) -> None:
    jsonl_path = tmp_path / "name_filter.jsonl"
    db_path = tmp_path / "name_filter.db"
    _write_jsonl(
        jsonl_path,
        [
            {
                "ExpressId": 1,
                "GlobalId": "roof-occurrence",
                "IfcType": "IfcRoof",
                "Name": "Main Roof",
            },
            {
                "ExpressId": 2,
                "GlobalId": "roof-type",
                "IfcType": "IfcRoofType",
                "Name": "Main Roof Type",
            },
            {
                "ExpressId": 3,
                "GlobalId": "roof-type-object",
                "IfcType": "IfcTypeObject",
                "Name": "Generic Roof Family",
            },
        ],
    )
    jsonl_to_sql(jsonl_path, db_path)

    result = query_ifc_sql(
        db_path,
        SqlRequest(
            intent="count",
            ifc_class=None,
            level_like=None,
            element_filters=(
                SqlValueFilter(
                    source="element",
                    field="name",
                    op="like",
                    value="%roof%",
                ),
            ),
        ),
    )

    assert result["data"]["count"] == 1
    assert "FROM properties" not in result["data"]["sql"]["query"]
    assert "e.name" in result["data"]["sql"]["query"]


def test_group_by_type_name_is_deterministic(tmp_path: Path) -> None:
    db_path = _build_test_db(tmp_path)

    result = query_ifc_sql(
        db_path,
        SqlRequest(
            intent="group",
            ifc_class="IfcWall",
            level_like="ground floor",
            group_by=SqlFieldRef(source="element", field="type_name"),
            limit=10,
        ),
    )

    assert result["data"]["groups"] == [
        {"group": "Curtain Wall:Exterior Curtain Wall", "count": 2},
        {"group": "Curtain Wall:Storefront", "count": 1},
    ]
    assert result["data"]["matched_element_count"] == 3
    assert result["data"]["missing_value_count"] == 0


def test_text_match_counts_descriptive_compound_occurrences_not_type_rows(
    tmp_path: Path,
) -> None:
    jsonl_path = tmp_path / "parking_spaces.jsonl"
    db_path = tmp_path / "parking_spaces.db"
    _write_jsonl(
        jsonl_path,
        [
            {
                "ExpressId": 1,
                "GlobalId": "parking-1",
                "IfcType": "IfcBuildingElementProxy",
                "Name": "Parking Space 1",
                "TypeName": "M_Parking Space:5480 x 2740mm - 90 deg",
            },
            {
                "ExpressId": 2,
                "GlobalId": "parking-2",
                "IfcType": "IfcBuildingElementProxy",
                "Name": "Parking Space 2",
                "TypeName": "M_Parking Space:5480 x 2740mm - 90 deg",
            },
            {
                "ExpressId": 3,
                "GlobalId": "parking-type",
                "IfcType": "IfcBuildingElementProxyType",
                "Name": "Parking Space Type",
                "TypeName": "M_Parking Space:5480 x 2740mm - 90 deg",
            },
        ],
    )
    jsonl_to_sql(jsonl_path, db_path)

    result = query_ifc_sql(
        db_path,
        SqlRequest(
            intent="count",
            ifc_class=None,
            level_like=None,
            text_match="parking space",
        ),
    )

    assert result["data"]["count"] == 2
    assert "e.type_name" in result["data"]["sql"]["query"]
    assert "IfcBuildingElementProxyType" not in result["data"]["summary"]


def test_class_plus_text_match_preserves_descriptive_subtype_constraint(
    tmp_path: Path,
) -> None:
    jsonl_path = tmp_path / "round_columns.jsonl"
    db_path = tmp_path / "round_columns.db"
    _write_jsonl(
        jsonl_path,
        [
            {
                "ExpressId": 1,
                "GlobalId": "column-round-1",
                "IfcType": "IfcColumn",
                "Name": "Round Column 1",
                "TypeName": "M_Concrete-Round-Column:300mm",
                "ObjectType": "Concrete Round Column",
            },
            {
                "ExpressId": 2,
                "GlobalId": "column-round-2",
                "IfcType": "IfcColumn",
                "Name": "Round Column 2",
                "TypeName": "M_Concrete-Round-Column:300mm",
                "ObjectType": "Concrete Round Column",
            },
            {
                "ExpressId": 3,
                "GlobalId": "column-square",
                "IfcType": "IfcColumn",
                "Name": "Square Column 1",
                "TypeName": "M_Concrete-Square-Column:300mm",
                "ObjectType": "Concrete Square Column",
            },
            {
                "ExpressId": 4,
                "GlobalId": "column-steel-round",
                "IfcType": "IfcColumn",
                "Name": "Steel Round Column 1",
                "TypeName": "M_Steel-Round-Column:300mm",
                "ObjectType": "Steel Round Column",
            },
        ],
    )
    jsonl_to_sql(jsonl_path, db_path)

    result = query_ifc_sql(
        db_path,
        SqlRequest(
            intent="count",
            ifc_class="IfcColumn",
            level_like=None,
            text_match="round concrete",
        ),
    )

    assert result["data"]["count"] == 2
    assert "e.object_type" in result["data"]["sql"]["query"]
    assert result["data"]["summary"] == "Found 2 IfcColumn matching 'round concrete'."


def test_text_match_is_token_aware_for_ground_round_false_positive(
    tmp_path: Path,
) -> None:
    jsonl_path = tmp_path / "ground_round.jsonl"
    db_path = tmp_path / "ground_round.db"
    _write_jsonl(
        jsonl_path,
        [
            {
                "ExpressId": 1,
                "GlobalId": "wall-ground",
                "IfcType": "IfcWall",
                "Name": "Ground Wall 1",
                "ObjectType": "Ground Floor Wall",
            },
            {
                "ExpressId": 2,
                "GlobalId": "column-round",
                "IfcType": "IfcColumn",
                "Name": "Round Column 1",
                "ObjectType": "Round Concrete Column",
            },
        ],
    )
    jsonl_to_sql(jsonl_path, db_path)

    result = query_ifc_sql(
        db_path,
        SqlRequest(
            intent="count",
            ifc_class=None,
            level_like=None,
            text_match="round",
        ),
    )

    assert result["data"]["count"] == 1
    assert result["data"]["evidence"][0]["global_id"] == "column-round"


def test_text_match_is_token_aware_for_street_tree_false_positive(
    tmp_path: Path,
) -> None:
    jsonl_path = tmp_path / "street_tree.jsonl"
    db_path = tmp_path / "street_tree.db"
    _write_jsonl(
        jsonl_path,
        [
            {
                "ExpressId": 1,
                "GlobalId": "street-light",
                "IfcType": "IfcLightFixture",
                "Name": "Street Light 1",
                "Description": "Street lighting pole",
            },
            {
                "ExpressId": 2,
                "GlobalId": "tree-proxy",
                "IfcType": "IfcBuildingElementProxy",
                "Name": "Tree 1",
                "Description": "Deciduous tree",
            },
        ],
    )
    jsonl_to_sql(jsonl_path, db_path)

    result = query_ifc_sql(
        db_path,
        SqlRequest(
            intent="count",
            ifc_class=None,
            level_like=None,
            text_match="tree",
        ),
    )

    assert result["data"]["count"] == 1
    assert result["data"]["evidence"][0]["global_id"] == "tree-proxy"
