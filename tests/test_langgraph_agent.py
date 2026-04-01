from __future__ import annotations

import networkx as nx
import pytest

from rag_tag.agent import langgraph_agent as langgraph_agent_module
from rag_tag.agent.langgraph_agent import LangGraphAgent
from rag_tag.config import GraphOrchestrationConfig
from rag_tag.evals.runtime import BENCHMARK_GRAPH_PROMPT_APPEND_ENV_VAR
from rag_tag.graph import wrap_networkx_graph


class FakeSpecialist:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    def run(
        self,
        question: str,
        runtime: object,
        *,
        max_steps: int = 20,
        trace: object | None = None,
        run_id: str | None = None,
    ) -> dict[str, object]:
        del runtime, trace, run_id
        self.calls.append((question, max_steps))
        return {"answer": f"answer:{question}", "data": {"max_steps": max_steps}}


def _joined_answer_synthesizer(state: object) -> dict[str, object]:
    specialist_results = state["specialist_results"]
    return {
        "answer": " ".join(
            item["result"]["answer"] for item in specialist_results if "result" in item
        ),
        "data": {
            "subquestions": list(state["subquestions"]),
            "specialist_results": list(specialist_results),
        },
    }


@pytest.fixture(autouse=True)
def _allow_langgraph_agent_without_installed_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        langgraph_agent_module,
        "_ensure_langgraph_dependency",
        lambda: None,
    )


def test_langgraph_agent_runs_sequential_specialist_calls_and_synthesizes() -> None:
    specialist = FakeSpecialist()
    agent = LangGraphAgent(
        specialist=specialist,
        orchestration_config=GraphOrchestrationConfig(
            max_subquestions=2,
            reserved_orchestration_steps=2,
            specialist_step_cap=2,
        ),
        decompose=lambda question: ["part one", "part two"],
        synthesize=_joined_answer_synthesizer,
    )

    result = agent.run(
        "original question",
        wrap_networkx_graph(nx.MultiDiGraph()),
        max_steps=8,
    )

    assert specialist.calls == [("part one", 2), ("part two", 2)]
    assert result["answer"] == "answer:part one answer:part two"
    assert result["data"]["subquestions"] == ["part one", "part two"]


def test_langgraph_agent_derives_budget_when_specialist_cap_is_unset() -> None:
    specialist = FakeSpecialist()
    agent = LangGraphAgent(
        specialist=specialist,
        orchestration_config=GraphOrchestrationConfig(
            max_subquestions=2,
            reserved_orchestration_steps=2,
            specialist_step_cap=None,
        ),
        decompose=lambda question: ["part one", "part two"],
        synthesize=_joined_answer_synthesizer,
    )

    agent.run(
        "original question",
        wrap_networkx_graph(nx.MultiDiGraph()),
        max_steps=8,
    )

    assert specialist.calls == [("part one", 3), ("part two", 3)]


def test_langgraph_agent_falls_back_to_graph_agent_on_invalid_final_output() -> None:
    specialist = FakeSpecialist()

    def invalid_synthesizer(state: object) -> dict[str, object]:
        del state
        return {"data": {"ok": True}}

    agent = LangGraphAgent(
        specialist=specialist,
        orchestration_config=GraphOrchestrationConfig(
            max_subquestions=1,
            reserved_orchestration_steps=2,
            specialist_step_cap=2,
            fallback_to_graph_agent=True,
        ),
        decompose=lambda question: ["focused subquestion"],
        synthesize=invalid_synthesizer,
    )

    result = agent.run(
        "original question",
        wrap_networkx_graph(nx.MultiDiGraph()),
        max_steps=6,
    )

    assert specialist.calls == [("focused subquestion", 2), ("original question", 2)]
    assert result["answer"] == "answer:original question"
    assert "LangGraph orchestration failed; fell back to GraphAgent" in str(
        result["warning"]
    )


def test_langgraph_agent_returns_bounded_answer_when_fallback_is_disabled() -> None:
    specialist = FakeSpecialist()

    def failing_synthesizer(state: object) -> dict[str, object]:
        del state
        raise RuntimeError("synthesis failed")

    agent = LangGraphAgent(
        specialist=specialist,
        orchestration_config=GraphOrchestrationConfig(
            fallback_to_graph_agent=False,
            reserved_orchestration_steps=1,
        ),
        decompose=lambda question: [question],
        synthesize=failing_synthesizer,
    )

    result = agent.run(
        "original question",
        wrap_networkx_graph(nx.MultiDiGraph()),
        max_steps=5,
    )

    assert result["answer"] == "answer:original question"
    assert "LangGraph synthesis failed: synthesis failed" in str(result["warning"])


def test_langgraph_agent_budget_exhaustion_skips_fallback_when_no_budget_remains() -> (
    None
):
    specialist = FakeSpecialist()
    agent = LangGraphAgent(
        specialist=specialist,
        orchestration_config=GraphOrchestrationConfig(
            max_subquestions=1,
            reserved_orchestration_steps=2,
            fallback_to_graph_agent=True,
        ),
        decompose=lambda question: ["focused subquestion"],
    )

    result = agent.run(
        "original question",
        wrap_networkx_graph(nx.MultiDiGraph()),
        max_steps=2,
    )

    assert specialist.calls == []
    assert "I could not gather enough graph evidence" in str(result["answer"])
    assert "Step budget exceeded before specialist call" in str(result["warning"])
    assert "fallback was skipped because no step budget remained" in str(
        result["warning"]
    )


def test_langgraph_agent_budget_exhaustion_without_fallback_stays_bounded() -> None:
    specialist = FakeSpecialist()
    agent = LangGraphAgent(
        specialist=specialist,
        orchestration_config=GraphOrchestrationConfig(
            max_subquestions=1,
            reserved_orchestration_steps=2,
            fallback_to_graph_agent=False,
        ),
        decompose=lambda question: ["focused subquestion"],
    )

    result = agent.run(
        "original question",
        wrap_networkx_graph(nx.MultiDiGraph()),
        max_steps=2,
    )

    assert specialist.calls == []
    assert "I could not gather enough graph evidence" in str(result["answer"])
    assert "Step budget exceeded before specialist call" in str(result["warning"])


def test_langgraph_agent_uses_model_backed_hooks_in_production_path() -> None:
    specialist = FakeSpecialist()
    agent = LangGraphAgent(
        specialist=specialist,
        orchestration_config=GraphOrchestrationConfig(
            max_subquestions=2,
            reserved_orchestration_steps=2,
            specialist_step_cap=2,
        ),
    )
    agent._decompose_with_model = lambda question: ["first", "second"]
    agent._synthesize_with_model = _joined_answer_synthesizer

    result = agent.run(
        "original question",
        wrap_networkx_graph(nx.MultiDiGraph()),
        max_steps=8,
    )

    assert specialist.calls == [("first", 2), ("second", 2)]
    assert result["answer"] == "answer:first answer:second"


def test_langgraph_agent_applies_benchmark_prompt_append_to_model_prompts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_prompts: list[str] = []

    class FakePydanticAgent:
        def __init__(self, model: object, **kwargs: object) -> None:
            del model
            system_prompt = kwargs.get("system_prompt")
            assert isinstance(system_prompt, str)
            captured_prompts.append(system_prompt)

    monkeypatch.setenv(BENCHMARK_GRAPH_PROMPT_APPEND_ENV_VAR, "benchmark appendix")
    monkeypatch.setattr(langgraph_agent_module, "Agent", FakePydanticAgent)
    monkeypatch.setattr(
        langgraph_agent_module,
        "_resolve_langgraph_model",
        lambda: (object(), None),
    )

    agent = LangGraphAgent(
        specialist=FakeSpecialist(),
        orchestration_config=GraphOrchestrationConfig(),
    )

    agent._get_decomposition_agent()
    agent._get_synthesis_agent()

    assert len(captured_prompts) == 2
    assert all("benchmark appendix" in prompt for prompt in captured_prompts)
    assert "focused subquestions" in captured_prompts[0]
    assert "grounded specialist results" in captured_prompts[1]


def test_langgraph_agent_init_requires_langgraph_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        langgraph_agent_module,
        "_ensure_langgraph_dependency",
        lambda: (_ for _ in ()).throw(RuntimeError("langgraph missing")),
    )

    with pytest.raises(RuntimeError, match="langgraph missing"):
        LangGraphAgent()
