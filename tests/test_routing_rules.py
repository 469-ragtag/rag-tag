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


def test_total_net_volume_routes_to_sql_aggregate() -> None:
    decision = route_question_rule("What is the total net volume of walls on level 2?")

    assert decision.route == "sql"
    assert decision.sql_request is not None
    assert decision.sql_request.intent == "aggregate"
    assert decision.sql_request.ifc_class == "IfcWall"
    assert decision.sql_request.level_like == "level 2"
    assert decision.sql_request.aggregate_op == "sum"
    assert decision.sql_request.aggregate_field is not None
    assert decision.sql_request.aggregate_field.source == "quantity"
    assert decision.sql_request.aggregate_field.field == "NetVolume"


def test_group_by_property_routes_to_sql_group() -> None:
    decision = route_question_rule("Group the doors on level 1 by fire rating.")

    assert decision.route == "sql"
    assert decision.sql_request is not None
    assert decision.sql_request.intent == "group"
    assert decision.sql_request.ifc_class == "IfcDoor"
    assert decision.sql_request.level_like == "level 1"
    assert decision.sql_request.group_by is not None
    assert decision.sql_request.group_by.source == "property"
    assert decision.sql_request.group_by.field == "FireRating"


def test_average_property_routes_to_sql_aggregate() -> None:
    decision = route_question_rule("Average UValue of windows on the ground floor.")

    assert decision.route == "sql"
    assert decision.sql_request is not None
    assert decision.sql_request.intent == "aggregate"
    assert decision.sql_request.ifc_class == "IfcWindow"
    assert decision.sql_request.level_like == "ground floor"
    assert decision.sql_request.aggregate_op == "avg"
    assert decision.sql_request.aggregate_field is not None
    assert decision.sql_request.aggregate_field.source == "property"
    assert decision.sql_request.aggregate_field.field == "UValue"
