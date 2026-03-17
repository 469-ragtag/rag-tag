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


def test_served_by_question_routes_to_graph_macro_path() -> None:
    decision = route_question_rule("Which equipment serves Room 101?")

    assert decision.route == "graph"


def test_classification_question_routes_to_graph_macro_path() -> None:
    decision = route_question_rule("Which walls are classified as Pr_70_70_63?")

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


def test_predefined_type_filter_routes_to_sql_without_greedy_capture() -> None:
    decision = route_question_rule(
        "List doors with predefined type SINGLE_SWING_LEFT on level 1."
    )

    assert decision.route == "sql"
    assert decision.sql_request is not None
    assert decision.sql_request.ifc_class == "IfcDoor"
    assert decision.sql_request.predefined_type == "SINGLE_SWING_LEFT"
    assert decision.sql_request.level_like == "level 1"


def test_type_name_filter_stops_at_clause_boundaries() -> None:
    decision = route_question_rule(
        "List windows with type name Triple Glazed Unit, where property UValue <= 1.0"
    )

    assert decision.route == "sql"
    assert decision.sql_request is not None
    assert decision.sql_request.ifc_class == "IfcWindow"
    assert decision.sql_request.type_name == "Triple Glazed Unit"
    assert decision.sql_request.property_filters[0].field == "UValue"
    assert decision.sql_request.property_filters[0].op == "lte"
    assert decision.sql_request.property_filters[0].value == 1.0


def test_explicit_property_filter_routes_to_sql() -> None:
    decision = route_question_rule(
        "Count doors where property FireRating = EI 60 on level 1."
    )

    assert decision.route == "sql"
    assert decision.sql_request is not None
    assert decision.sql_request.intent == "count"
    assert decision.sql_request.ifc_class == "IfcDoor"
    assert decision.sql_request.level_like == "level 1"
    assert decision.sql_request.property_filters[0].field == "FireRating"
    assert decision.sql_request.property_filters[0].op == "eq"
    assert decision.sql_request.property_filters[0].value == "EI 60"


def test_explicit_quantity_filter_routes_to_sql() -> None:
    decision = route_question_rule(
        "List walls where quantity NetVolume >= 10 on level 2."
    )

    assert decision.route == "sql"
    assert decision.sql_request is not None
    assert decision.sql_request.intent == "list"
    assert decision.sql_request.ifc_class == "IfcWall"
    assert decision.sql_request.level_like == "level 2"
    assert decision.sql_request.quantity_filters[0].field == "NetVolume"
    assert decision.sql_request.quantity_filters[0].op == "gte"
    assert decision.sql_request.quantity_filters[0].value == 10
