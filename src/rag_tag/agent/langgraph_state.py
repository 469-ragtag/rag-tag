"""Typed state helpers for the LangGraph graph orchestrator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict

from rag_tag.config import GraphOrchestrationConfig
from rag_tag.graph import GraphRuntime


@dataclass(frozen=True)
class LangGraphRunContext:
    """Per-run context supplied to the compiled LangGraph workflow."""

    runtime: GraphRuntime


class LangGraphState(TypedDict):
    """Mutable state shared across LangGraph-style orchestration nodes."""

    question: str
    decomposition_enabled: bool
    max_subquestions: int
    reserved_orchestration_steps: int
    specialist_step_cap: int | None
    fallback_to_graph_agent: bool
    max_steps: int
    remaining_steps: int
    subquestions: list[str]
    current_subquestion_index: int
    current_subquestion: str | None
    specialist_results: list[dict[str, object]]
    warnings: list[str]
    final_output: dict[str, object] | None
    fallback_required: bool
    fallback_reason: str | None
    fallback_result: dict[str, object] | None


def build_initial_langgraph_state(
    question: str,
    *,
    max_steps: int,
    config: GraphOrchestrationConfig,
) -> LangGraphState:
    """Construct the initial orchestration state from config defaults."""
    normalized_steps = max(max_steps, 1)
    reserved_steps = max(config.reserved_orchestration_steps, 0)
    remaining_steps = max(normalized_steps - reserved_steps, 0)

    return LangGraphState(
        question=question,
        decomposition_enabled=config.enabled_subquestion_decomposition,
        max_subquestions=max(config.max_subquestions, 1),
        reserved_orchestration_steps=reserved_steps,
        specialist_step_cap=(
            max(config.specialist_step_cap, 1)
            if config.specialist_step_cap is not None
            else None
        ),
        fallback_to_graph_agent=config.fallback_to_graph_agent,
        max_steps=normalized_steps,
        remaining_steps=remaining_steps,
        subquestions=[],
        current_subquestion_index=0,
        current_subquestion=None,
        specialist_results=[],
        warnings=[],
        final_output=None,
        fallback_required=False,
        fallback_reason=None,
        fallback_result=None,
    )


def specialist_step_budget(state: LangGraphState) -> int:
    """Return the available step budget for the current specialist call."""
    remaining = max(int(state["remaining_steps"]), 0)
    cap = state["specialist_step_cap"]
    if cap is None:
        pending = max(
            len(state["subquestions"]) - state["current_subquestion_index"],
            1,
        )
        if remaining <= 0:
            return 0
        return max(remaining // pending, 1)
    return min(remaining, max(cap, 1))
