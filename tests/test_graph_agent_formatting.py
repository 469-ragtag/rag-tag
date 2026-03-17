from __future__ import annotations

from rag_tag.agent.graph_agent import SYSTEM_PROMPT, _sanitize_model_text


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
    assert "find_container_elements_excluding" in SYSTEM_PROMPT
    assert "trace_distribution_network" in SYSTEM_PROMPT
    assert "find_shortest_path" in SYSTEM_PROMPT
    assert "find_by_classification" in SYSTEM_PROMPT
    assert "find_equipment_serving_space" in SYSTEM_PROMPT
    assert "aggregate_elements" in SYSTEM_PROMPT
    assert "group_elements_by_property" in SYSTEM_PROMPT
    assert "do not count or sum mentally" in SYSTEM_PROMPT
