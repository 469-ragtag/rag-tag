from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import networkx as nx
import pytest
from pydantic_ai import ModelRetry
from pydantic_ai.models.test import TestModel

from rag_tag.agent.graph_agent import GraphAgent
from rag_tag.agent.models import GraphAnswer
from rag_tag.graph import GraphRuntime, wrap_networkx_graph


def _validator(agent: GraphAgent):
    return agent._agent._output_validators[0].function


def test_output_validator_retries_plain_text_recovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("rag_tag.agent.graph_agent.get_agent_model", TestModel)
    agent = GraphAgent()
    output = GraphAnswer.model_validate("The plumbing wall is 3800 mm long.")

    with pytest.raises(ModelRetry, match="plain assistant text"):
        cast(Any, _validator(agent))(SimpleNamespace(partial_output=False), output)


def test_output_validator_retries_wrapped_tool_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("rag_tag.agent.graph_agent.get_agent_model", TestModel)
    agent = GraphAgent()
    output = GraphAnswer.model_validate(
        [
            {
                "tool_call_id": "5",
                "tool_name": "final_result",
                "parameters": {"answer": "Recovered.", "data": None},
            }
        ]
    )

    with pytest.raises(ModelRetry, match="wrapped tool payload"):
        cast(Any, _validator(agent))(SimpleNamespace(partial_output=False), output)


def test_unexpected_model_behavior_salvages_nested_tool_envelope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("rag_tag.agent.graph_agent.get_agent_model", TestModel)
    agent = GraphAgent()

    class FakeUnexpectedModelBehavior(Exception):
        def __init__(self, body: object) -> None:
            super().__init__("bad output")
            self.body = body

    def fake_run_sync(
        question: str, *, deps: GraphRuntime, usage_limits: object
    ) -> object:
        raise FakeUnexpectedModelBehavior(
            {
                "input": [
                    {
                        "tool_call_id": "5",
                        "tool_name": "final_result",
                        "parameters": {
                            "answer": "Recovered final answer.",
                            "data": {"length": 3800},
                            "warning": None,
                        },
                    }
                ]
            }
        )

    monkeypatch.setattr(agent._agent, "run_sync", fake_run_sync)
    monkeypatch.setattr(
        "rag_tag.agent.graph_agent.UnexpectedModelBehavior",
        FakeUnexpectedModelBehavior,
    )

    result = agent.run("question", wrap_networkx_graph(nx.MultiDiGraph()))

    assert result["answer"] == "Recovered final answer."
    assert result["data"] == {"length": 3800}
    assert "Recovered answer from malformed final_result output." in str(
        result["warning"]
    )


def test_unexpected_model_behavior_runs_targeted_output_tool_repair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("rag_tag.agent.graph_agent.get_agent_model", TestModel)
    agent = GraphAgent()

    class FakeUnexpectedModelBehavior(Exception):
        def __init__(self, body: object) -> None:
            super().__init__("Please include your response in a tool call.")
            self.body = body

    message_history = [{"role": "assistant", "parts": ["bad output"]}]
    captured: dict[str, object] = {}
    call_count = {"main": 0, "repair": 0}

    class _CapturedMessages:
        def __enter__(self) -> list[object]:
            return cast(list[object], message_history)

        def __exit__(self, exc_type, exc, tb) -> bool:
            return False

    def fake_main_run_sync(
        question: str,
        *,
        deps: GraphRuntime,
        usage_limits: object,
        message_history: list[object] | None = None,
    ) -> object:
        call_count["main"] += 1
        raise FakeUnexpectedModelBehavior(
            {
                "message": "Please include your response in a tool call.",
                "input": [
                    {
                        "tool_call_id": "5",
                        "tool_name": "final_result",
                        "parameters": {
                            "answer": "The geo-reference element is black.",
                            "data": {"element_id": "Element::abc"},
                            "warning": None,
                        },
                    }
                ],
            }
        )

    def fake_repair_run_sync(
        question: str,
        *,
        deps: GraphRuntime,
        usage_limits: object,
        message_history: list[object] | None = None,
    ) -> object:
        call_count["repair"] += 1
        captured["question"] = question
        captured["message_history"] = message_history
        captured["usage_limits"] = usage_limits
        return SimpleNamespace(
            output=GraphAnswer(
                answer="The geo-reference element is black.",
                data={"element_id": "Element::abc"},
                warning=None,
            )
        )

    monkeypatch.setattr(agent._agent, "run_sync", fake_main_run_sync)
    monkeypatch.setattr(agent._repair_agent, "run_sync", fake_repair_run_sync)
    monkeypatch.setattr(
        "rag_tag.agent.graph_agent.UnexpectedModelBehavior",
        FakeUnexpectedModelBehavior,
    )
    monkeypatch.setattr(
        "rag_tag.agent.graph_agent.capture_run_messages",
        lambda: _CapturedMessages(),
    )

    result = agent.run("question", wrap_networkx_graph(nx.MultiDiGraph()))

    assert result["answer"] == "The geo-reference element is black."
    assert result["data"] == {"element_id": "Element::abc"}
    assert call_count == {"main": 1, "repair": 1}
    assert captured["message_history"] == message_history
    assert "Repair the previous failed output formatting only." in str(
        captured["question"]
    )
    assert "Return exactly one JSON object" in str(captured["question"])


def test_output_tool_repair_detects_nested_tool_retry_cause(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("rag_tag.agent.graph_agent.get_agent_model", TestModel)
    agent = GraphAgent()

    class FakeToolRetryError(Exception):
        pass

    class FakeUnexpectedModelBehavior(Exception):
        def __init__(self) -> None:
            super().__init__("Exceeded maximum retries (5) for output validation")

    message_history = [{"role": "assistant", "parts": ["bad output"]}]
    call_count = {"main": 0, "repair": 0}

    class _CapturedMessages:
        def __enter__(self) -> list[object]:
            return cast(list[object], message_history)

        def __exit__(self, exc_type, exc, tb) -> bool:
            return False

    def fake_main_run_sync(
        question: str,
        *,
        deps: GraphRuntime,
        usage_limits: object,
        message_history: list[object] | None = None,
    ) -> object:
        call_count["main"] += 1
        outer = FakeUnexpectedModelBehavior()
        outer.__cause__ = FakeToolRetryError(
            "Please include your response in a tool call."
        )
        raise outer

    def fake_repair_run_sync(
        question: str,
        *,
        deps: GraphRuntime,
        usage_limits: object,
        message_history: list[object] | None = None,
    ) -> object:
        call_count["repair"] += 1
        return SimpleNamespace(
            output=GraphAnswer(
                answer="Recovered after nested tool retry failure.",
                data={"element_id": "Element::abc"},
                warning=None,
            )
        )

    monkeypatch.setattr(agent._agent, "run_sync", fake_main_run_sync)
    monkeypatch.setattr(agent._repair_agent, "run_sync", fake_repair_run_sync)
    monkeypatch.setattr(
        "rag_tag.agent.graph_agent.UnexpectedModelBehavior",
        FakeUnexpectedModelBehavior,
    )
    monkeypatch.setattr(
        "rag_tag.agent.graph_agent.capture_run_messages",
        lambda: _CapturedMessages(),
    )

    result = agent.run("question", wrap_networkx_graph(nx.MultiDiGraph()))

    assert result["answer"] == "Recovered after nested tool retry failure."
    assert result["data"] == {"element_id": "Element::abc"}
    assert call_count == {"main": 1, "repair": 1}
