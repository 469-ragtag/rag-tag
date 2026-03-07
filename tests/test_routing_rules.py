from __future__ import annotations

from rag_tag.router.rules import route_question_rule


def test_ground_floor_window_count_routes_to_sql() -> None:
    decision = route_question_rule("Count windows on the ground floor.")

    assert decision.route == "sql"
    assert decision.sql_request is not None
    assert decision.sql_request.ifc_class == "IfcWindow"
    assert decision.sql_request.level_like == "ground floor"


def test_building_existence_question_does_not_trigger_multi_class_graph_route() -> None:
    decision = route_question_rule("Are there any windows in the building?")

    assert decision.route == "sql"
    assert decision.sql_request is not None
    assert decision.sql_request.ifc_class == "IfcWindow"
    assert decision.sql_request.level_like is None


def test_ground_floor_door_list_routes_to_sql() -> None:
    decision = route_question_rule("List all doors on the ground floor.")

    assert decision.route == "sql"
    assert decision.sql_request is not None
    assert decision.sql_request.ifc_class == "IfcDoor"
    assert decision.sql_request.level_like == "ground floor"


def test_spatial_cue_still_routes_to_graph() -> None:
    decision = route_question_rule("Find doors near the stair core.")

    assert decision.route == "graph"
