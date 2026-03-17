from __future__ import annotations

import networkx as nx
import pytest

from rag_tag.agent import langgraph_agent as langgraph_agent_module
from rag_tag.agent.langgraph_agent import LangGraphAgent
from rag_tag.config import GraphOrchestrationConfig
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

    assert specialist.calls == [("focused subquestion", 4), ("original question", 6)]
    assert result["answer"] == "answer:original question"
    assert "LangGraph orchestration failed; fell back to GraphAgent" in str(
        result["warning"]
    )


def test_langgraph_agent_returns_error_when_fallback_is_disabled() -> None:
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
        synthesize=failing_synthesizer,
    )

    result = agent.run(
        "original question",
        wrap_networkx_graph(nx.MultiDiGraph()),
        max_steps=5,
    )

    assert result == {"error": "LangGraph orchestration failed: synthesis failed"}


def test_langgraph_agent_budget_exhaustion_falls_back_once_and_returns() -> None:
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

    assert specialist.calls == [("original question", 2)]
    assert result["answer"] == "answer:original question"
    assert "LangGraph orchestration failed; fell back to GraphAgent" in str(
        result["warning"]
    )
    assert "Step budget exceeded before specialist call" in str(result["warning"])


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
