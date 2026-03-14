"""Graph orchestrator wrapper built on the LangGraph-style state/node layer."""

from __future__ import annotations

import importlib.util
from collections.abc import Callable, Sequence
from pathlib import Path

from rag_tag.config import GraphOrchestrationConfig, load_project_config
from rag_tag.graph import GraphRuntime

from .graph_agent import GraphAgent
from .langgraph_nodes import (
    decompose_question,
    dispatch_next_subquestion,
    finalize_output,
    run_specialist,
    should_dispatch_next_subquestion,
    synthesize_answer,
)
from .langgraph_state import LangGraphState, build_initial_langgraph_state
from .models import GraphAnswer

Decomposer = Callable[[str], Sequence[str]]
Synthesizer = Callable[[LangGraphState], dict[str, object]]

_MODULE_DIR = Path(__file__).resolve().parent


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

        try:
            decompose_question(state, decompose=self._decompose)

            while (
                should_dispatch_next_subquestion(state) == "dispatch_next_subquestion"
            ):
                dispatch_next_subquestion(state)
                run_specialist(
                    state,
                    run_specialist_call=lambda subquestion, budget: (
                        self._specialist.run(
                            subquestion,
                            runtime,
                            max_steps=budget,
                        )
                    ),
                )

            synthesize_answer(
                state,
                synthesize=self._synthesize or _default_synthesizer,
            )
            output = GraphAnswer.model_validate(finalize_output(state))
            return _graph_answer_to_response(output)
        except Exception as exc:
            if not state["fallback_to_graph_agent"]:
                return {"error": f"LangGraph orchestration failed: {exc}"}
            return _apply_fallback_warning(
                self._specialist.run(question, runtime, max_steps=max_steps),
                reason=str(exc),
            )


def _load_orchestration_config() -> GraphOrchestrationConfig:
    loaded = load_project_config(_MODULE_DIR)
    return loaded.config.graph_orchestration


def _ensure_langgraph_dependency() -> None:
    if importlib.util.find_spec("langgraph") is None:
        raise RuntimeError(
            "LangGraph orchestrator requires the 'langgraph' package to be "
            "installed. Run the project dependency sync to install it."
        )


def _default_synthesizer(state: LangGraphState) -> dict[str, object]:
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


def _graph_answer_to_response(output: GraphAnswer) -> dict[str, object]:
    response: dict[str, object] = {"answer": output.answer}
    if output.data:
        response["data"] = output.data
    if output.warning:
        response["warning"] = output.warning
    return response


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


def _merge_warning_text(warnings: list[str]) -> str | None:
    deduped: list[str] = []
    for warning in warnings:
        if warning and warning not in deduped:
            deduped.append(warning)
    return " ".join(deduped) if deduped else None
