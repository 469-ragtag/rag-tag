from __future__ import annotations

from rag_tag.router.llm import _build_system_prompt
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


def test_materials_question_routes_to_graph() -> None:
    decision = route_question_rule(
        "What are the materials of all the walls located on the groundfloor?"
    )

    assert decision.route == "graph"
    assert "materials/color" in decision.reason


def test_named_element_comparison_routes_to_graph() -> None:
    decision = route_question_rule(
        "Compare the net volume of the right roof slab and the left roof slab. "
        "Which is larger?"
    )

    assert decision.route == "graph"
    assert "comparison between specific named elements" in decision.reason


def test_color_fuzzy_named_object_lookup_routes_to_graph() -> None:
    decision = route_question_rule(
        "What is the color (RGB or Material) of the geo-reference element?"
    )

    assert decision.route == "graph"
    assert "fuzzy named-object lookup" in decision.reason


def test_room_containment_question_routes_to_graph() -> None:
    decision = route_question_rule("Which doors are in the kitchen?")

    assert decision.route == "graph"
    assert "room/space containment" in decision.reason


def test_zone_question_routes_to_graph() -> None:
    decision = route_question_rule("Which spaces are in the fire zone A?")

    assert decision.route == "graph"
    assert "systems/serving/classification/zone membership" in decision.reason


def test_ifc_space_with_level_routes_to_graph_for_correctness() -> None:
    decision = route_question_rule("Count spaces on level 2")

    assert decision.route == "graph"
    assert "IfcSpace + level/storey questions" in decision.reason


def test_bare_space_count_stays_ifc_space_sql() -> None:
    decision = route_question_rule("How many spaces are there?")

    assert decision.route == "sql"
    assert decision.sql_request is not None
    assert decision.sql_request.intent == "count"
    assert decision.sql_request.ifc_class == "IfcSpace"
    assert decision.sql_request.text_match is None


def test_parking_spaces_route_to_text_match_not_ifc_space() -> None:
    decision = route_question_rule("How many parking spaces are there?")

    assert decision.route == "sql"
    assert decision.sql_request is not None
    assert decision.sql_request.intent == "count"
    assert decision.sql_request.ifc_class is None
    assert decision.sql_request.text_match == "parking space"


def test_type_presence_question_routes_to_group_by_type_name() -> None:
    decision = route_question_rule("What curtain wall types are present?")

    assert decision.route == "sql"
    assert decision.sql_request is not None
    assert decision.sql_request.intent == "group"
    assert decision.sql_request.ifc_class == "IfcCurtainWall"
    assert decision.sql_request.group_by is not None
    assert decision.sql_request.group_by.source == "element"
    assert decision.sql_request.group_by.field == "type_name"


def test_longer_multiword_alias_wins_over_shorter_overlap() -> None:
    decision = route_question_rule("List curtain walls in the building.")

    assert decision.route == "sql"
    assert decision.sql_request is not None
    assert decision.sql_request.intent == "list"
    assert decision.sql_request.ifc_class == "IfcCurtainWall"


def test_family_question_routes_to_group_by_type_name() -> None:
    decision = route_question_rule("Which window families exist?")

    assert decision.route == "sql"
    assert decision.sql_request is not None
    assert decision.sql_request.intent == "group"
    assert decision.sql_request.ifc_class == "IfcWindow"
    assert decision.sql_request.group_by is not None
    assert decision.sql_request.group_by.source == "element"
    assert decision.sql_request.group_by.field == "type_name"


def test_kinds_question_routes_to_group_by_type_name() -> None:
    decision = route_question_rule("What kinds of columns are there?")

    assert decision.route == "sql"
    assert decision.sql_request is not None
    assert decision.sql_request.intent == "group"
    assert decision.sql_request.ifc_class == "IfcColumn"
    assert decision.sql_request.group_by is not None
    assert decision.sql_request.group_by.source == "element"
    assert decision.sql_request.group_by.field == "type_name"


def test_name_word_filter_count_stays_sql() -> None:
    decision = route_question_rule(
        "How many elements have the word roof in their name?"
    )

    assert decision.route == "sql"
    assert decision.sql_request is not None
    assert decision.sql_request.intent == "count"
    assert decision.sql_request.ifc_class is None
    assert decision.sql_request.element_filters[0].source == "element"
    assert decision.sql_request.element_filters[0].field == "name"
    assert decision.sql_request.element_filters[0].op == "like"
    assert decision.sql_request.element_filters[0].value == "%roof%"


def test_llm_prompt_mentions_shared_capability_matrix() -> None:
    prompt = _build_system_prompt()

    assert "Shared capability matrix:" in prompt
    assert "Graph-first categories" in prompt
    assert "room/space containment membership" in prompt
    assert "materials/color (unsupported by current SQLite schema)" in prompt
    assert "deterministic count/list/aggregate/group" in prompt
    assert "text_match" in prompt
    assert "types/families/kinds" in prompt
