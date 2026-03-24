"""Graph orchestrator wrapper built on a compiled LangGraph workflow."""

from __future__ import annotations

import importlib.util
import json
from collections.abc import Callable, Sequence
from pathlib import Path

from langgraph.graph import END, StateGraph
from pydantic import BaseModel, ConfigDict, Field
from pydantic_ai import Agent

from rag_tag.config import GraphOrchestrationConfig, load_project_config
from rag_tag.graph import GraphRuntime
from rag_tag.llm.pydantic_ai import get_agent_model, get_agent_model_settings
from rag_tag.usage import UsageMetrics, normalize_usage_metrics, sum_usage_metrics

from .graph_agent import (
    GraphAgent,
    _graph_answer_to_response,
    _is_pydantic_test_model,
)
from .langgraph_nodes import (
    decompose_question,
    dispatch_next_subquestion,
    fallback_to_legacy_agent,
    finalize_output,
    route_after_decomposition,
    route_after_specialist,
    route_after_synthesis,
    run_specialist,
    synthesize_answer,
)
from .langgraph_state import (
    LangGraphState,
    build_initial_langgraph_state,
)
from .models import GraphAnswer

Decomposer = Callable[[str], Sequence[str]]
Synthesizer = Callable[[LangGraphState], dict[str, object]]

_MODULE_DIR = Path(__file__).resolve().parent

_DECOMPOSITION_SYSTEM_PROMPT = """
You are decomposing an IFC graph-reasoning question into a short sequence of
focused subquestions for a specialist graph agent.

Rules:
- Preserve the user's intent exactly.
- Return 1 to 3 subquestions.
- Keep each subquestion answerable by the existing graph specialist.
- Prefer sequential dependencies when the later question relies on an earlier
  answer.
- Do not mention orchestration, budgets, or implementation details.
"""

_SYNTHESIS_SYSTEM_PROMPT = """
You are synthesizing a final IFC graph answer from grounded specialist results.

Rules:
- Use only the supplied specialist outputs.
- Return a GraphAnswer-compatible JSON object.
- Keep the answer concise and directly responsive to the original question.
- Include a warning only when the evidence is partial or contradictory.
- Preserve useful structured data when possible.
"""


class _DecompositionOutput(BaseModel):
    """Structured subquestion plan returned by the decomposition model."""

    model_config = ConfigDict(extra="forbid")

    subquestions: list[str] = Field(default_factory=list)


class LangGraphAgent:
    """Wrapper that orchestrates multiple specialist calls before synthesis."""

    def __init__(
        self,
        *,
        debug_llm_io: bool = False,
        specialist: GraphAgent | None = None,
        orchestration_config: GraphOrchestrationConfig | None = None,
        decompose: Decomposer | None = None,
        synthesize: Synthesizer | None = None,
    ) -> None:
        _ensure_langgraph_dependency()
        self._debug_llm_io = debug_llm_io
        self._specialist = specialist or GraphAgent(debug_llm_io=debug_llm_io)
        self._config = orchestration_config or _load_orchestration_config()
        self._decompose = decompose
        self._synthesize = synthesize
        self._decomposition_agent: Agent[None, _DecompositionOutput] | None = None
        self._synthesis_agent: Agent[None, GraphAnswer] | None = None
        self._active_runtime: GraphRuntime | None = None
        self._active_usage_records: list[UsageMetrics] = []
        self._workflow = self._build_workflow()

    def run(
        self,
        question: str,
        runtime: GraphRuntime,
        *,
        max_steps: int = 20,
        trace: object | None = None,
        run_id: str | None = None,
    ) -> dict[str, object]:
        del trace, run_id
        state = build_initial_langgraph_state(
            question,
            max_steps=max_steps,
            config=self._config,
        )
        final_state = state
        self._active_usage_records = []

        try:
            self._active_runtime = runtime
            final_state = self._workflow.invoke(state)
            response = self._response_from_state(final_state)
        except Exception as exc:
            response = self._outer_failure_response(state, runtime, exc)
        finally:
            self._active_runtime = None
        return self._attach_usage(response, final_state)

    def _build_workflow(self):
        builder = StateGraph(LangGraphState)

        def _decompose_node(state: LangGraphState) -> LangGraphState:
            return decompose_question(
                state,
                decompose=self._decompose or self._decompose_with_model,
            )

        def _dispatch_node(state: LangGraphState) -> LangGraphState:
            return dispatch_next_subquestion(state)

        def _run_specialist_node(state: LangGraphState) -> LangGraphState:
            return run_specialist(
                state,
                run_specialist_call=lambda subquestion, budget: self._specialist.run(
                    subquestion,
                    self._require_active_runtime(),
                    max_steps=budget,
                ),
            )

        def _synthesize_node(state: LangGraphState) -> LangGraphState:
            return synthesize_answer(
                state,
                synthesize=self._synthesize or self._synthesize_with_model,
            )

        def _fallback_node(state: LangGraphState) -> LangGraphState:
            return fallback_to_legacy_agent(
                state,
                run_fallback_call=lambda original_question, budget: (
                    self._specialist.run(
                        original_question,
                        self._require_active_runtime(),
                        max_steps=budget,
                    )
                ),
            )

        def _finalize_node(state: LangGraphState) -> LangGraphState:
            finalize_output(
                state,
                finalize=lambda current_state: self._finalize_state_output(
                    current_state,
                    self._require_active_runtime(),
                ),
            )
            return state

        builder.add_node("decompose_question", _decompose_node)
        builder.add_node("dispatch_next_subquestion", _dispatch_node)
        builder.add_node("run_specialist", _run_specialist_node)
        builder.add_node("synthesize_answer", _synthesize_node)
        builder.add_node("fallback_to_legacy_agent", _fallback_node)
        builder.add_node("finalize_output", _finalize_node)

        builder.set_entry_point("decompose_question")
        builder.add_conditional_edges(
            "decompose_question",
            route_after_decomposition,
            {
                "dispatch_next_subquestion": "dispatch_next_subquestion",
                "fallback_to_legacy_agent": "fallback_to_legacy_agent",
                "synthesize_answer": "synthesize_answer",
            },
        )
        builder.add_edge("dispatch_next_subquestion", "run_specialist")
        builder.add_conditional_edges(
            "run_specialist",
            route_after_specialist,
            {
                "dispatch_next_subquestion": "dispatch_next_subquestion",
                "fallback_to_legacy_agent": "fallback_to_legacy_agent",
                "synthesize_answer": "synthesize_answer",
            },
        )
        builder.add_conditional_edges(
            "synthesize_answer",
            route_after_synthesis,
            {
                "fallback_to_legacy_agent": "fallback_to_legacy_agent",
                "finalize_output": "finalize_output",
            },
        )
        builder.add_edge("fallback_to_legacy_agent", "finalize_output")
        builder.add_edge("finalize_output", END)
        return builder.compile()

    def _require_active_runtime(self) -> GraphRuntime:
        if self._active_runtime is None:
            raise RuntimeError("LangGraph runtime context was not initialized.")
        return self._active_runtime

    def _decompose_with_model(self, question: str) -> list[str]:
        agent = self._get_decomposition_agent()
        prompt = (
            "Original graph question:\n"
            f"{question.strip()}\n\n"
            f"Return at most {self._config.max_subquestions} focused subquestions."
        )
        result = agent.run_sync(prompt)
        self._record_usage(result)
        subquestions = [
            item.strip() for item in result.output.subquestions if item.strip()
        ]
        return subquestions[: self._config.max_subquestions] or [question]

    def _synthesize_with_model(self, state: LangGraphState) -> dict[str, object]:
        if not state["specialist_results"]:
            return _deterministic_bounded_output(state)
        agent = self._get_synthesis_agent()
        prompt = (
            "Original graph question:\n"
            f"{state['question']}\n\n"
            "Subquestions:\n"
            f"{json.dumps(state['subquestions'], indent=2, ensure_ascii=True)}\n\n"
            "Specialist results:\n"
            f"{json.dumps(state['specialist_results'], indent=2, ensure_ascii=True)}"
            "\n\n"
            "Warnings gathered so far:\n"
            f"{json.dumps(state['warnings'], indent=2, ensure_ascii=True)}"
        )
        result = agent.run_sync(prompt)
        self._record_usage(result)
        return result.output.model_dump(exclude_none=True)

    def _get_decomposition_agent(self) -> Agent[None, _DecompositionOutput]:
        if self._decomposition_agent is None:
            model, model_settings = _resolve_langgraph_model()
            self._decomposition_agent = Agent(
                model,
                output_type=_DecompositionOutput,
                system_prompt=_DECOMPOSITION_SYSTEM_PROMPT,
                model_settings=model_settings,
                retries=1,
                output_retries=1,
            )
        return self._decomposition_agent

    def _get_synthesis_agent(self) -> Agent[None, GraphAnswer]:
        if self._synthesis_agent is None:
            model, model_settings = _resolve_langgraph_model()
            self._synthesis_agent = Agent(
                model,
                output_type=GraphAnswer,
                system_prompt=_SYNTHESIS_SYSTEM_PROMPT,
                model_settings=model_settings,
                retries=1,
                output_retries=1,
            )
        return self._synthesis_agent

    def _finalize_state_output(
        self,
        state: LangGraphState,
        runtime: GraphRuntime,
    ) -> dict[str, object]:
        if state["fallback_result"] is not None:
            return self._normalize_fallback_result(state)

        payload = state["final_output"]
        if payload is None:
            payload = _deterministic_bounded_output(state)

        try:
            output = GraphAnswer.model_validate(payload)
            return _graph_answer_to_response(output)
        except Exception as exc:
            state["warnings"].append(f"LangGraph finalization failed: {exc}")
            if state["fallback_to_graph_agent"] and int(state["remaining_steps"]) > 0:
                state["fallback_result"] = self._specialist.run(
                    state["question"],
                    runtime,
                    max_steps=int(state["remaining_steps"]),
                )
                state["remaining_steps"] = 0
                return self._normalize_fallback_result(state)
            return self._normalize_invalid_payload(state, exc)

    def _normalize_fallback_result(self, state: LangGraphState) -> dict[str, object]:
        result = dict(state["fallback_result"] or {})
        reason = state["fallback_reason"] or _merge_warning_text(state["warnings"])
        if "answer" in result and isinstance(result.get("answer"), str):
            if reason:
                return _apply_fallback_warning(result, reason=reason)
            return result

        message = result.get("error")
        if not isinstance(message, str) or not message.strip():
            message = "LangGraph fallback did not return a valid answer."
        return _graph_answer_to_response(
            GraphAnswer(
                answer=(
                    "I could not complete the LangGraph orchestration, and the legacy "
                    "graph fallback did not produce a final answer."
                ),
                data={
                    "fallback_result": result,
                    "specialist_results": list(state["specialist_results"]),
                },
                warning=_merge_warning_text(
                    list(state["warnings"]) + [f"LangGraph fallback failed: {message}"]
                ),
            )
        )

    def _normalize_invalid_payload(
        self,
        state: LangGraphState,
        exc: Exception,
    ) -> dict[str, object]:
        return _graph_answer_to_response(
            GraphAnswer(
                answer=(
                    "I could not gather enough graph evidence to answer this question."
                ),
                data={
                    "subquestions": list(state["subquestions"]),
                    "specialist_results": list(state["specialist_results"]),
                    "raw_final_output": state["final_output"],
                },
                warning=_merge_warning_text(
                    list(state["warnings"])
                    + [f"LangGraph final output normalization failed: {exc}"]
                ),
            )
        )

    def _response_from_state(self, state: LangGraphState) -> dict[str, object]:
        payload = state.get("final_output")
        if payload is None:
            return self._normalize_invalid_payload(
                state,
                RuntimeError("Missing GraphAnswer-compatible final output."),
            )
        try:
            output = GraphAnswer.model_validate(payload)
        except Exception:
            if _looks_like_public_response(payload):
                return payload
            return self._normalize_invalid_payload(
                state,
                RuntimeError("Missing GraphAnswer-compatible final output."),
            )
        return _graph_answer_to_response(output)

    def _outer_failure_response(
        self,
        state: LangGraphState,
        runtime: GraphRuntime,
        exc: Exception,
    ) -> dict[str, object]:
        if state["fallback_to_graph_agent"] and int(state["remaining_steps"]) > 0:
            fallback_result = self._specialist.run(
                state["question"],
                runtime,
                max_steps=int(state["remaining_steps"]),
            )
            return _normalize_outer_fallback_result(
                fallback_result,
                state=state,
                reason=str(exc),
            )
        state["warnings"].append(f"LangGraph orchestration failed: {exc}")
        return _graph_answer_to_response(
            GraphAnswer(
                answer=(
                    "I could not gather enough graph evidence to answer this question."
                ),
                data={
                    "subquestions": list(state["subquestions"]),
                    "specialist_results": list(state["specialist_results"]),
                },
                warning=_merge_warning_text(state["warnings"]),
            )
        )

    def _record_usage(self, payload: object) -> None:
        normalized = normalize_usage_metrics(payload)
        if normalized.usage_available:
            self._active_usage_records.append(normalized)

    def _attach_usage(
        self,
        response: dict[str, object],
        state: LangGraphState,
    ) -> dict[str, object]:
        aggregate = sum_usage_metrics(
            *self._active_usage_records,
            *(
                item.get("result", {}).get("usage")
                for item in state["specialist_results"]
                if isinstance(item.get("result"), dict)
            ),
            state["fallback_result"].get("usage")
            if isinstance(state["fallback_result"], dict)
            else None,
        )
        if not aggregate.usage_available:
            return response
        enriched = dict(response)
        enriched["usage"] = aggregate.as_dict()
        return enriched


def _load_orchestration_config() -> GraphOrchestrationConfig:
    loaded = load_project_config(_MODULE_DIR)
    return loaded.config.graph_orchestration


def _ensure_langgraph_dependency() -> None:
    if importlib.util.find_spec("langgraph") is None:
        raise RuntimeError(
            "LangGraph orchestrator requires the 'langgraph' package to be "
            "installed. Run the project dependency sync to install it."
        )


def _resolve_langgraph_model():
    model = get_agent_model()
    try:
        model_settings = get_agent_model_settings()
    except RuntimeError:
        if _is_pydantic_test_model(model):
            model_settings = None
        else:
            raise
    return model, model_settings


def _deterministic_bounded_output(state: LangGraphState) -> dict[str, object]:
    answers: list[str] = []
    warnings = list(state["warnings"])
    errors: list[str] = []

    for item in state["specialist_results"]:
        result = item.get("result")
        if not isinstance(result, dict):
            continue
        answer = result.get("answer")
        if isinstance(answer, str) and answer.strip():
            answers.append(answer.strip())
        warning = result.get("warning")
        if isinstance(warning, str) and warning.strip():
            warnings.append(warning.strip())
        error = result.get("error")
        if isinstance(error, str) and error.strip():
            errors.append(error.strip())

    if answers:
        answer = " ".join(answers)
    elif errors:
        answer = "I could not gather enough graph evidence to answer this question."
        warnings.append("LangGraph synthesis completed with only specialist errors.")
    else:
        answer = "I could not gather enough graph evidence to answer this question yet."
        warnings.append("LangGraph synthesis ran without specialist answers.")

    data: dict[str, object] = {
        "subquestions": list(state["subquestions"]),
        "specialist_results": list(state["specialist_results"]),
    }
    if errors:
        data["specialist_errors"] = errors

    payload: dict[str, object] = {"answer": answer, "data": data}
    warning_text = _merge_warning_text(warnings)
    if warning_text is not None:
        payload["warning"] = warning_text
    return payload


def _apply_fallback_warning(
    fallback_result: dict[str, object],
    *,
    reason: str,
) -> dict[str, object]:
    warning = f"LangGraph orchestration failed; fell back to GraphAgent: {reason}"
    response = dict(fallback_result)
    existing_warning = response.get("warning")
    if isinstance(existing_warning, str) and existing_warning.strip():
        response["warning"] = f"{warning} {existing_warning}"
    elif "error" not in response:
        response["warning"] = warning
    return response


def _normalize_outer_fallback_result(
    fallback_result: dict[str, object],
    *,
    state: LangGraphState,
    reason: str,
) -> dict[str, object]:
    if "answer" in fallback_result and isinstance(fallback_result.get("answer"), str):
        return _apply_fallback_warning(fallback_result, reason=reason)
    return _graph_answer_to_response(
        GraphAnswer(
            answer=(
                "I could not complete the LangGraph orchestration, and the legacy "
                "graph fallback did not produce a final answer."
            ),
            data={
                "fallback_result": fallback_result,
                "specialist_results": list(state["specialist_results"]),
            },
            warning=_merge_warning_text(
                list(state["warnings"]) + [f"LangGraph fallback failed: {reason}"]
            ),
        )
    )


def _looks_like_public_response(payload: object) -> bool:
    return (
        isinstance(payload, dict)
        and isinstance(payload.get("answer"), str)
        and (
            "data" not in payload
            or payload.get("data") is None
            or isinstance(payload.get("data"), dict)
        )
        and (
            "warning" not in payload
            or payload.get("warning") is None
            or isinstance(payload.get("warning"), str)
        )
    )


def _merge_warning_text(warnings: Sequence[str]) -> str | None:
    deduped: list[str] = []
    for warning in warnings:
        if warning and warning not in deduped:
            deduped.append(warning)
    return " ".join(deduped) if deduped else None
