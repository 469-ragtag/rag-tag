"""Pure node functions for the LangGraph graph orchestrator."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Literal

from .langgraph_state import LangGraphState, specialist_step_budget

Decomposer = Callable[[str], Sequence[str]]
SpecialistRunner = Callable[[str, int], dict[str, object]]
Synthesizer = Callable[[LangGraphState], dict[str, object]]
Finalizer = Callable[[LangGraphState], dict[str, object]]


def decompose_question(
    state: LangGraphState,
    *,
    decompose: Decomposer | None = None,
) -> LangGraphState:
    """Populate subquestions for the workflow, respecting config caps."""
    try:
        if not state["decomposition_enabled"]:
            state["subquestions"] = [state["question"]]
            return state

        if decompose is None:
            candidates = [state["question"]]
        else:
            candidates = [
                item.strip() for item in decompose(state["question"]) if item.strip()
            ]

        if not candidates:
            candidates = [state["question"]]

        state["subquestions"] = list(candidates[: state["max_subquestions"]])
    except Exception as exc:
        state["warnings"].append(f"LangGraph decomposition failed: {exc}")
        state["fallback_reason"] = str(exc)
        state["fallback_required"] = state["fallback_to_graph_agent"]
        state["subquestions"] = []
    return state


def dispatch_next_subquestion(state: LangGraphState) -> LangGraphState:
    """Select the next subquestion to hand to the specialist worker."""
    index = state["current_subquestion_index"]
    subquestions = state["subquestions"]
    state["current_subquestion"] = (
        subquestions[index] if index < len(subquestions) else None
    )
    return state


def run_specialist(
    state: LangGraphState,
    *,
    run_specialist_call: SpecialistRunner,
) -> LangGraphState:
    """Run the specialist on the current subquestion and track budget/results."""
    subquestion = state["current_subquestion"]
    if subquestion is None:
        return state

    budget = specialist_step_budget(state)
    if budget <= 0:
        reason = f"Step budget exceeded before specialist call for '{subquestion}'."
        state["warnings"].append(reason)
        state["fallback_reason"] = reason
        state["fallback_required"] = state["fallback_to_graph_agent"]
        state["current_subquestion_index"] = len(state["subquestions"])
        state["current_subquestion"] = None
        return state

    try:
        result = run_specialist_call(subquestion, budget)
        state["specialist_results"].append(
            {
                "question": subquestion,
                "result": result,
                "budget_used": budget,
            }
        )
        state["remaining_steps"] = max(int(state["remaining_steps"]) - budget, 0)
        state["current_subquestion_index"] += 1
        state["current_subquestion"] = None
    except Exception as exc:
        state["warnings"].append(
            f"LangGraph specialist execution failed for '{subquestion}': {exc}"
        )
        state["fallback_reason"] = str(exc)
        state["fallback_required"] = state["fallback_to_graph_agent"]
        state["current_subquestion_index"] = len(state["subquestions"])
        state["current_subquestion"] = None
    return state


def should_continue_specialist_loop(
    state: LangGraphState,
) -> Literal["run_specialist", "synthesize_answer"]:
    """Return the next workflow edge after dispatch."""
    if state["current_subquestion"] is not None:
        return "run_specialist"
    return "synthesize_answer"


def should_dispatch_next_subquestion(
    state: LangGraphState,
) -> Literal["dispatch_next_subquestion", "synthesize_answer"]:
    """Return whether more specialist work remains."""
    if state["current_subquestion_index"] < len(state["subquestions"]):
        return "dispatch_next_subquestion"
    return "synthesize_answer"


def route_after_decomposition(
    state: LangGraphState,
) -> Literal[
    "dispatch_next_subquestion", "fallback_to_legacy_agent", "synthesize_answer"
]:
    """Route after decomposition based on failure and available work."""
    if state["fallback_required"]:
        return "fallback_to_legacy_agent"
    return should_dispatch_next_subquestion(state)


def route_after_specialist(
    state: LangGraphState,
) -> Literal[
    "dispatch_next_subquestion", "fallback_to_legacy_agent", "synthesize_answer"
]:
    """Route after each specialist step."""
    if state["fallback_required"]:
        return "fallback_to_legacy_agent"
    return should_dispatch_next_subquestion(state)


def synthesize_answer(
    state: LangGraphState,
    *,
    synthesize: Synthesizer | None = None,
) -> LangGraphState:
    """Assemble a deterministic placeholder final payload from collected results."""
    if synthesize is not None:
        try:
            state["final_output"] = synthesize(state)
            return state
        except Exception as exc:
            state["warnings"].append(f"LangGraph synthesis failed: {exc}")
            state["fallback_reason"] = str(exc)
            state["fallback_required"] = state["fallback_to_graph_agent"]
            return state

    answered = [
        item["result"].get("answer")
        for item in state["specialist_results"]
        if isinstance(item.get("result"), dict)
        and isinstance(item["result"].get("answer"), str)
        and item["result"]["answer"].strip()
    ]
    answer = " ".join(answered).strip()
    if not answer:
        answer = "I could not gather enough graph evidence to answer this question yet."
        state["warnings"].append("LangGraph synthesis ran without specialist answers.")

    payload: dict[str, object] = {
        "answer": answer,
        "data": {
            "subquestions": list(state["subquestions"]),
            "specialist_results": list(state["specialist_results"]),
        },
    }
    if state["warnings"]:
        payload["warning"] = " ".join(state["warnings"])
    state["final_output"] = payload
    return state


def route_after_synthesis(
    state: LangGraphState,
) -> Literal["fallback_to_legacy_agent", "finalize_output"]:
    """Route after synthesis based on whether fallback is required."""
    if state["fallback_required"]:
        return "fallback_to_legacy_agent"
    return "finalize_output"


def fallback_to_legacy_agent(
    state: LangGraphState,
    *,
    run_fallback_call: SpecialistRunner,
) -> LangGraphState:
    """Retry once through the legacy GraphAgent when allowed and budgeted."""
    if not state["fallback_required"]:
        return state

    budget = max(int(state["remaining_steps"]), 0)
    if budget <= 0:
        state["warnings"].append(
            "LangGraph fallback was skipped because no step budget remained."
        )
        return state

    try:
        state["fallback_result"] = run_fallback_call(state["question"], budget)
        state["remaining_steps"] = 0
    except Exception as exc:
        state["warnings"].append(f"LangGraph fallback execution failed: {exc}")
        state["fallback_reason"] = str(exc)
        state["fallback_result"] = {"error": f"Agent execution failed: {exc}"}
        state["remaining_steps"] = 0
    return state


def finalize_output(
    state: LangGraphState,
    *,
    finalize: Finalizer | None = None,
) -> dict[str, object]:
    """Return the final output payload, synthesizing if needed."""
    if finalize is not None:
        state["final_output"] = finalize(state)
        return state["final_output"]
    if state["final_output"] is None:
        synthesize_answer(state)
    return state["final_output"] or {
        "answer": "",
        "warning": "No final output produced.",
    }
