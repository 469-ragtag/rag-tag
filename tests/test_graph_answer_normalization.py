from __future__ import annotations

from rag_tag.agent.models import (
    GraphAnswer,
    RecoveryKind,
    recovery_kind,
    was_normalized_from_plain_text,
)


def test_graph_answer_normalizes_plain_text_paragraph_response() -> None:
    output = GraphAnswer.model_validate(
        (
            "The plumbing wall length is 3800 mm. Adjacent elements include "
            "the kitchen and the outer wall."
        )
    )

    assert output.answer.startswith("The plumbing wall length is 3800 mm")
    assert output.data is None
    assert output.warning is None
    assert was_normalized_from_plain_text(output) is True
    assert recovery_kind(output) == RecoveryKind.PLAIN_TEXT


def test_graph_answer_normalizes_bullet_list_response() -> None:
    output = GraphAnswer.model_validate(
        "- plumbing wall length: 3800 mm\n- adjacent: kitchen, outer wall"
    )

    assert "plumbing wall length" in output.answer
    assert output.data is None
    assert output.warning is None
    assert was_normalized_from_plain_text(output) is True
    assert recovery_kind(output) == RecoveryKind.PLAIN_TEXT


def test_graph_answer_regular_json_is_not_marked_plain_text() -> None:
    output = GraphAnswer.model_validate(
        {"answer": "The plumbing wall length is 3800 mm.", "data": None}
    )

    assert was_normalized_from_plain_text(output) is False
    assert recovery_kind(output) == RecoveryKind.NONE


def test_graph_answer_unwraps_single_item_tool_envelope_list() -> None:
    output = GraphAnswer.model_validate(
        [
            {
                "tool_call_id": "5",
                "tool_name": "final_result",
                "parameters": {
                    "answer": "Recovered final result.",
                    "data": {"length": 3800},
                    "warning": None,
                },
            }
        ]
    )

    assert output.answer == "Recovered final result."
    assert output.data == {"length": 3800}
    assert recovery_kind(output) == RecoveryKind.TOOL_ENVELOPE


def test_graph_answer_schema_remains_strict() -> None:
    schema = GraphAnswer.model_json_schema()

    assert schema["additionalProperties"] is False
