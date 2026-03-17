from __future__ import annotations

import json
from pathlib import Path

from rag_tag.ifc_sql_tool import query_ifc_sql
from rag_tag.level_normalization import canonicalize_level
from rag_tag.parser.jsonl_to_sql import jsonl_to_sql
from rag_tag.router.models import SqlRequest


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def test_canonicalize_level_normalizes_numeric_and_ground_floor_variants() -> None:
    assert canonicalize_level("Level 02") == "level 2"
    assert canonicalize_level("floor 2") == "level 2"
    assert canonicalize_level("Ground Level") == "ground floor"


def test_sql_level_filter_is_exact_after_normalization(tmp_path: Path) -> None:
    jsonl_path = tmp_path / "levels.jsonl"
    db_path = tmp_path / "levels.db"
    _write_jsonl(
        jsonl_path,
        [
            {
                "ExpressId": 1,
                "GlobalId": "gid-level-2",
                "IfcType": "IfcDoor",
                "Name": "Door L2",
                "Hierarchy": {"Level": "Level 2"},
            },
            {
                "ExpressId": 2,
                "GlobalId": "gid-level-12",
                "IfcType": "IfcDoor",
                "Name": "Door L12",
                "Hierarchy": {"Level": "Level 12"},
            },
            {
                "ExpressId": 3,
                "GlobalId": "gid-ground",
                "IfcType": "IfcDoor",
                "Name": "Door Ground",
                "Hierarchy": {"Level": "Ground Level"},
            },
        ],
    )

    jsonl_to_sql(jsonl_path, db_path)

    level_2 = query_ifc_sql(
        db_path,
        SqlRequest(intent="count", ifc_class="IfcDoor", level_like="Level 2", limit=0),
    )
    assert level_2["data"]["count"] == 1
    assert level_2["data"]["evidence"] == [
        {
            "global_id": "gid-level-2",
            "id": 1,
            "label": "Door L2",
            "class_": "IfcDoor",
            "source_tool": "query_ifc_sql",
            "match_reason": "representative_match",
        }
    ]

    floor_2 = query_ifc_sql(
        db_path,
        SqlRequest(intent="count", ifc_class="IfcDoor", level_like="floor 2", limit=0),
    )
    assert floor_2["data"]["count"] == 1

    ground_floor = query_ifc_sql(
        db_path,
        SqlRequest(
            intent="count",
            ifc_class="IfcDoor",
            level_like="ground floor",
            limit=0,
        ),
    )
    assert ground_floor["data"]["count"] == 1

    listed = query_ifc_sql(
        db_path,
        SqlRequest(intent="list", ifc_class="IfcDoor", level_like="Level 2", limit=5),
    )
    assert listed["data"]["items"] == [
        {
            "express_id": 1,
            "global_id": "gid-level-2",
            "ifc_class": "IfcDoor",
            "name": "Door L2",
            "level": "Level 2",
            "type_name": None,
        }
    ]
    assert listed["data"]["evidence"] == [
        {
            "global_id": "gid-level-2",
            "id": 1,
            "label": "Door L2",
            "class_": "IfcDoor",
            "source_tool": "query_ifc_sql",
        }
    ]
