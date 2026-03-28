from __future__ import annotations

from rag_tag.query_presentation import build_query_presentation


def test_build_query_presentation_formats_sql_list_rows() -> None:
    result = {
        "route": "sql",
        "answer": "Found 2 walls, showing 1.",
        "data": {
            "intent": "list",
            "items": [
                {
                    "name": "Wall A",
                    "ifc_class": "IfcWall",
                    "level": "Level 1",
                }
            ],
            "total_count": 2,
        },
    }

    presentation = build_query_presentation(result)

    assert presentation["answer"] == "Found 2 walls, showing 1."
    assert presentation["sql_items"] == [
        "1. Name: Wall A | Class: IfcWall | Level: Level 1"
    ]
    assert presentation["sql_more_count"] == 1
    assert '"route": "sql"' in presentation["details_json"]


def test_build_query_presentation_preserves_graph_samples_and_errors() -> None:
    result = {
        "route": "graph",
        "error": "graph failed",
        "data": {"sample": ["Wall A", "Wall B"]},
    }

    presentation = build_query_presentation(result)

    assert presentation["error"] == "graph failed"
    assert presentation["answer"] is None
    assert presentation["graph_sample"] == ["Wall A", "Wall B"]
