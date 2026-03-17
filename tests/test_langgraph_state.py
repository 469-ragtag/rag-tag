from __future__ import annotations

from rag_tag.agent.langgraph_nodes import (
    decompose_question,
    dispatch_next_subquestion,
    finalize_output,
    run_specialist,
    should_continue_specialist_loop,
    should_dispatch_next_subquestion,
    synthesize_answer,
)
from rag_tag.agent.langgraph_state import (
    build_initial_langgraph_state,
    specialist_step_budget,
)
from rag_tag.config import GraphOrchestrationConfig


def test_build_initial_langgraph_state_applies_defaults() -> None:
    state = build_initial_langgraph_state(
        "Which rooms are adjacent to the kitchen?",
        max_steps=10,
        config=GraphOrchestrationConfig(),
    )

    assert state["question"] == "Which rooms are adjacent to the kitchen?"
    assert state["decomposition_enabled"] is True
    assert state["max_subquestions"] == 3
    assert state["reserved_orchestration_steps"] == 3
    assert state["remaining_steps"] == 7
    assert state["fallback_to_graph_agent"] is True


def test_build_initial_langgraph_state_normalizes_invalid_numeric_inputs() -> None:
    state = build_initial_langgraph_state(
        "Q",
        max_steps=0,
        config=GraphOrchestrationConfig(
            max_subquestions=0,
            reserved_orchestration_steps=-2,
            specialist_step_cap=0,
        ),
    )

    assert state["max_steps"] == 1
    assert state["max_subquestions"] == 1
    assert state["reserved_orchestration_steps"] == 0
    assert state["specialist_step_cap"] == 1
    assert state["remaining_steps"] == 1


def test_decompose_question_respects_disable_flag() -> None:
    state = build_initial_langgraph_state(
        "Original question",
        max_steps=8,
        config=GraphOrchestrationConfig(enabled_subquestion_decomposition=False),
    )

    updated = decompose_question(
        state,
        decompose=lambda question: ["ignored", question],
    )

    assert updated["subquestions"] == ["Original question"]


def test_decompose_question_caps_subquestions_and_strips_blanks() -> None:
    state = build_initial_langgraph_state(
        "Original question",
        max_steps=8,
        config=GraphOrchestrationConfig(max_subquestions=2),
    )

    updated = decompose_question(
        state,
        decompose=lambda question: [" first ", "", "second", "third"],
    )

    assert updated["subquestions"] == ["first", "second"]


def test_specialist_step_budget_honors_cap() -> None:
    state = build_initial_langgraph_state(
        "Q",
        max_steps=10,
        config=GraphOrchestrationConfig(
            reserved_orchestration_steps=1,
            specialist_step_cap=4,
        ),
    )

    assert specialist_step_budget(state) == 4


def test_run_specialist_tracks_result_and_consumes_budget() -> None:
    state = build_initial_langgraph_state(
        "Original question",
        max_steps=10,
        config=GraphOrchestrationConfig(
            max_subquestions=2,
            reserved_orchestration_steps=2,
            specialist_step_cap=3,
        ),
    )
    decompose_question(state, decompose=lambda question: ["part one", "part two"])
    dispatch_next_subquestion(state)

    updated = run_specialist(
        state,
        run_specialist_call=lambda subquestion, budget: {
            "answer": f"{subquestion} with budget={budget}"
        },
    )

    assert updated["current_subquestion_index"] == 1
    assert updated["current_subquestion"] is None
    assert updated["remaining_steps"] == 5
    assert updated["specialist_results"] == [
        {
            "question": "part one",
            "result": {"answer": "part one with budget=3"},
            "budget_used": 3,
        }
    ]


def test_run_specialist_marks_fallback_when_budget_is_exhausted() -> None:
    state = build_initial_langgraph_state(
        "Original question",
        max_steps=2,
        config=GraphOrchestrationConfig(
            reserved_orchestration_steps=2,
            fallback_to_graph_agent=True,
        ),
    )
    decompose_question(state)
    dispatch_next_subquestion(state)

    updated = run_specialist(
        state,
        run_specialist_call=lambda subquestion, budget: {"answer": subquestion},
    )

    assert updated["fallback_required"] is True
    assert updated["current_subquestion_index"] == len(updated["subquestions"])
    assert "Step budget exceeded before specialist call" in updated["warnings"][0]
    assert updated["specialist_results"] == []


def test_workflow_branch_helpers_reflect_state_progress() -> None:
    state = build_initial_langgraph_state(
        "Original question",
        max_steps=8,
        config=GraphOrchestrationConfig(max_subquestions=2, specialist_step_cap=2),
    )
    decompose_question(state, decompose=lambda question: ["part one", "part two"])

    assert should_dispatch_next_subquestion(state) == "dispatch_next_subquestion"

    dispatch_next_subquestion(state)
    assert should_continue_specialist_loop(state) == "run_specialist"

    run_specialist(
        state,
        run_specialist_call=lambda subquestion, budget: {"answer": subquestion},
    )
    assert should_dispatch_next_subquestion(state) == "dispatch_next_subquestion"

    dispatch_next_subquestion(state)
    run_specialist(
        state,
        run_specialist_call=lambda subquestion, budget: {"answer": subquestion},
    )
    assert should_dispatch_next_subquestion(state) == "synthesize_answer"


def test_synthesize_and_finalize_output_include_results_and_warnings() -> None:
    state = build_initial_langgraph_state(
        "Original question",
        max_steps=6,
        config=GraphOrchestrationConfig(),
    )
    state["subquestions"] = ["part one"]
    state["specialist_results"] = [
        {
            "question": "part one",
            "result": {"answer": "Recovered answer."},
            "budget_used": 2,
        }
    ]
    state["warnings"] = ["partial evidence"]

    synthesize_answer(state)
    output = finalize_output(state)

    assert output["answer"] == "Recovered answer."
    assert output["warning"] == "partial evidence"
    assert output["data"]["subquestions"] == ["part one"]
