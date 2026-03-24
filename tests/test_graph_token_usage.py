from __future__ import annotations

from rag_tag.agent.graph_agent import GraphAgent
from rag_tag.agent.langgraph_agent import LangGraphAgent
from rag_tag.agent.models import GraphAnswer


class _RunResult:
    def __init__(self, output: object, usage: dict[str, int]) -> None:
        self.output = output
        self._usage = usage

    def usage(self) -> dict[str, int]:
        return self._usage


def test_graph_agent_attaches_usage_to_success_response() -> None:
    agent = GraphAgent.__new__(GraphAgent)
    agent._agent = type(
        "FakeRunner",
        (),
        {
            "run_sync": staticmethod(
                lambda *args, **kwargs: _RunResult(
                    GraphAnswer(answer="Level 2", data=None, warning=None),
                    {"input_tokens": 18, "output_tokens": 4},
                )
            )
        },
    )()
    agent._repair_agent = None
    agent._output_retries = 1

    runtime = object()
    response = agent.run("What storey is the chimney on?", runtime, max_steps=5)

    assert response["answer"] == "Level 2"
    assert response["usage"] == {
        "input_tokens": 18,
        "output_tokens": 4,
        "total_tokens": 22,
        "reasoning_tokens": None,
        "usage_available": True,
    }


def test_langgraph_agent_aggregates_orchestration_and_specialist_usage() -> None:
    agent = LangGraphAgent.__new__(LangGraphAgent)
    agent._active_runtime = None
    agent._active_usage_records = []
    agent._specialist = None
    agent._config = type(
        "FakeConfig",
        (),
        {
            "enabled_subquestion_decomposition": True,
            "max_subquestions": 3,
            "reserved_orchestration_steps": 1,
            "specialist_step_cap": None,
            "fallback_to_graph_agent": False,
        },
    )()
    def invoke(state: dict[str, object]) -> dict[str, object]:
        agent._record_usage({"input_tokens": 9, "output_tokens": 2})
        return {
            **state,
            "final_output": {
                "answer": "Kitchen and dining room.",
                "data": None,
            },
            "specialist_results": [
                {
                    "question": "Which rooms touch the kitchen?",
                    "result": {
                        "answer": "Kitchen and dining room.",
                        "usage": {
                            "input_tokens": 12,
                            "output_tokens": 3,
                        },
                    },
                    "budget_used": 5,
                }
            ],
        }

    agent._workflow = type("FakeWorkflow", (), {"invoke": staticmethod(invoke)})()
    agent._response_from_state = LangGraphAgent._response_from_state.__get__(agent)
    agent._outer_failure_response = LangGraphAgent._outer_failure_response.__get__(
        agent
    )
    agent._attach_usage = LangGraphAgent._attach_usage.__get__(agent)
    agent._record_usage = LangGraphAgent._record_usage.__get__(agent)
    response = agent.run("Which rooms are adjacent to the kitchen?", object())

    assert response["usage"] == {
        "input_tokens": 21,
        "output_tokens": 5,
        "total_tokens": 26,
        "reasoning_tokens": None,
        "usage_available": True,
    }
