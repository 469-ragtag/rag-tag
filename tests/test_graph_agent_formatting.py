from __future__ import annotations

from rag_tag.agent.graph_agent import (
    _SCHEMA_CORRECTION_HINT,
    SYSTEM_PROMPT,
    _sanitize_model_text,
)


def test_sanitize_model_text_preserves_markdown_line_breaks() -> None:
    value = (
        "<co:assistant>## Wall\r\n\r\n"
        "Length: 3800 mm\r\n"
        "Adjacent elements:\r\n"
        "- kitchen\r\n"
        "- living room</co:assistant>"
    )

    sanitized = _sanitize_model_text(value)

    assert sanitized == (
        "## Wall\n\nLength: 3800 mm\nAdjacent elements:\n- kitchen\n- living room"
    )


def test_system_prompt_allows_lightweight_markdown_answers() -> None:
    assert "Use lightweight Markdown in `answer`" in SYSTEM_PROMPT
    assert "Do not use ASCII-art tables" in SYSTEM_PROMPT
    assert "Lightweight Markdown\nis allowed inside the `answer` field" in SYSTEM_PROMPT


def test_system_prompt_makes_macro_first_preferences_explicit_and_ordered() -> None:
    assert "### Macro-first defaults" in SYSTEM_PROMPT
    assert "use it before generic `traverse` or manual multi-step" in SYSTEM_PROMPT
    assert "Prefer macro/helper tools first" in SYSTEM_PROMPT

    ordered_preferences = [
        "1. `trace_distribution_network`",
        "2. `find_shortest_path`",
        "3. `find_by_classification`",
        "4. `find_equipment_serving_space`",
        "5. `aggregate_elements` / `group_elements_by_property`",
    ]
    positions = [SYSTEM_PROMPT.index(item) for item in ordered_preferences]

    assert positions == sorted(positions)
    assert "find_container_elements_excluding" in SYSTEM_PROMPT
    assert "trace_distribution_network" in SYSTEM_PROMPT
    assert "find_shortest_path" in SYSTEM_PROMPT
    assert "find_by_classification" in SYSTEM_PROMPT
    assert "find_equipment_serving_space" in SYSTEM_PROMPT
    assert "aggregate_elements" in SYSTEM_PROMPT
    assert "group_elements_by_property" in SYSTEM_PROMPT
    assert "do not count or sum mentally" in SYSTEM_PROMPT
    assert (
        "do not count, sum,\n   average, min/max, or group in-context" in SYSTEM_PROMPT
    )


def test_system_prompt_guides_generic_container_anchor_discipline() -> None:
    assert "single best canonical container" in SYSTEM_PROMPT
    assert "do not fan out across several fuzzy matches in parallel" in SYSTEM_PROMPT
    assert "inspect one best canonical\n     anchor first" in SYSTEM_PROMPT
    assert "focused" in SYSTEM_PROMPT
    assert "from that one anchor instead of launching parallel traversals" in (
        SYSTEM_PROMPT
    )


def test_system_prompt_prefers_containment_helpers_before_broad_topology() -> None:
    assert "`contains`, `contained_in`" in SYSTEM_PROMPT
    assert "`get_elements_in_storey`" in SYSTEM_PROMPT
    assert "`find_same_storey_elements`" in SYSTEM_PROMPT
    assert "`find_elements_inside_footprint`" in SYSTEM_PROMPT
    assert "`find_container_elements_excluding`" in SYSTEM_PROMPT
    assert "before\n    broad topology traversal" in SYSTEM_PROMPT


def test_system_prompt_treats_intersects_bbox_as_noisy_last_resort() -> None:
    assert "Use `intersects_bbox` only as a noisy last resort" in SYSTEM_PROMPT
    assert "Treat `intersects_bbox` as a noisy fallback" in SYSTEM_PROMPT


def test_system_prompt_covers_alignment_footprint_and_shared_boundary_tools() -> None:
    assert "`aligned_with`" in SYSTEM_PROMPT
    assert "`shares_boundary_with`" in SYSTEM_PROMPT
    assert "inside footprint" in SYSTEM_PROMPT
    assert "find_elements_inside_footprint" in SYSTEM_PROMPT
    assert "find_same_storey_elements" in SYSTEM_PROMPT


def test_schema_correction_hint_is_explicit_about_real_tool_calls() -> None:
    assert "real final_result tool call only" in _SCHEMA_CORRECTION_HINT
    assert "Do NOT print plain text, Markdown, fenced JSON" in _SCHEMA_CORRECTION_HINT
    assert (
        "Do NOT include tool_call_id, tool_name, or parameters"
        in _SCHEMA_CORRECTION_HINT
    )
