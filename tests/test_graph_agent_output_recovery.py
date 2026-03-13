from __future__ import annotations

from types import SimpleNamespace

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
        _validator(agent)(SimpleNamespace(partial_output=False), output)


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
        _validator(agent)(SimpleNamespace(partial_output=False), output)


def test_unexpected_model_behavior_salvages_nested_tool_envelope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("rag_tag.agent.graph_agent.get_agent_model", TestModel)
    agent = GraphAgent()

    class FakeUnexpectedModelBehavior(Exception):
        def __init__(self, body: object) -> None:
            super().__init__("bad output")
            self.body = body

    def fake_run_sync(question: str, *, deps: GraphRuntime) -> object:
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
