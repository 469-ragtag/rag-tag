"""Graph agent with explicit state machine workflow."""

from __future__ import annotations

import json

import networkx as nx

from rag_tag.ifc_graph_tool import query_ifc_graph
from rag_tag.llm.models import AgentStep
from rag_tag.llm.providers.base import BaseProvider

SYSTEM_PROMPT = """
You are a graph-reasoning agent for an IFC knowledge graph.

Schema (no direct graph access):
- Nodes have attributes: label, class_, properties, geometry
- Edges have attributes: relation, distance
- Hierarchy can include: Project ? Site ? Building ? Storey ? Space ? Elements
- Some levels may be missing; use labels and class_ to identify nodes
- Spatial adjacency edges exist with relation = "adjacent_to"

You must NOT access the graph directly. You can only request tools by returning JSON.
Allowed tool actions and params:
- find_nodes: {"class": "<IfcClassName or class name without Ifc prefix>",
  "property_filters": {...}}
- traverse: {"start": "<node id>", "relation": "<edge relation>", "depth": 1}
- spatial_query: {"class": "<IfcClassName or class name without Ifc prefix>",
  "near": "<Element::<GlobalId> or GlobalId>", "max_distance": 2.0}
- get_elements_in_storey: {"storey": "<storey name>"}
- find_elements_by_class: {"class": "<IfcClassName or class name without Ifc prefix>"}
- get_adjacent_elements: {"element_id": "<Element::<GlobalId> or GlobalId>"}

Use a ReAct-style loop. On each step, return a JSON object matching this schema:

Tool step: {"type": "tool", "action": "<action>", "params": { ... }}
Final step: {"type": "final", "answer": "<concise answer>"}

Required fields:
- type: must be "tool" or "final"
- For tool steps: action (string) and params (object) are required
- For final steps: answer (string) is required

No extra text, no code fences, just valid JSON.
""".strip()


class GraphAgent:
    """Graph agent with explicit workflow state machine."""

    def __init__(
        self,
        provider: BaseProvider,
        *,
        debug_llm_io: bool = False,
    ) -> None:
        """Initialize graph agent with LLM provider.

        Args:
            provider: LLM provider instance
            debug_llm_io: Enable debug printing of LLM I/O
        """
        self._provider = provider
        self._debug_llm_io = debug_llm_io

    def plan(
        self,
        question: str,
        history: list[dict[str, object]],
    ) -> AgentStep:
        """Generate next agent step using LLM.

        Args:
            question: User question
            history: Tool call history

        Returns:
            Validated AgentStep (tool or final)

        Raises:
            ValidationError: If LLM output doesn't match schema
        """
        payload = {"question": question, "history": history}
        prompt = json.dumps(payload)

        step = self._provider.generate_structured(
            prompt,
            schema=AgentStep,
            system_prompt=SYSTEM_PROMPT,
        )

        return step

    def run(
        self,
        question: str,
        graph: nx.DiGraph,
        *,
        max_steps: int = 6,
    ) -> dict[str, object]:
        """Execute agent workflow loop.

        Args:
            question: User question
            graph: NetworkX graph to query
            max_steps: Maximum reasoning steps

        Returns:
            Result dict with 'answer' or 'error' key, optionally 'history'
        """
        history: list[dict[str, object]] = []

        for _ in range(max_steps):
            try:
                step = self.plan(question, history)
            except Exception as exc:
                return {
                    "error": f"Planning failed: {exc}",
                    "history": history,
                }

            if step.type == "final":
                return {"answer": step.answer}

            if step.type == "tool":
                if not step.action:
                    return {
                        "error": "Tool step missing action",
                        "history": history,
                    }

                try:
                    tool_result = query_ifc_graph(
                        graph,
                        step.action,
                        step.params or {},
                    )
                except Exception as exc:
                    return {
                        "error": f"Tool execution failed: {exc}",
                        "history": history,
                    }

                history.append(
                    {
                        "tool": {
                            "action": step.action,
                            "params": step.params or {},
                        },
                        "result": tool_result,
                    }
                )
                continue

            # Unknown step type (should not happen with Pydantic validation)
            return {
                "error": f"Invalid step type: {step.type}",
                "history": history,
            }

        # Max steps exceeded
        summary = _summarize_graph_history(history)
        result: dict[str, object] = {
            "error": "Max steps exceeded",
            "history": history,
        }
        if summary:
            result.update(summary)
        return result


def _summarize_graph_history(
    history: list[dict[str, object]],
) -> dict[str, object] | None:
    """Extract summary from graph history when max steps exceeded.

    Args:
        history: Tool call history

    Returns:
        Summary dict with 'answer' and 'data', or None
    """
    for entry in reversed(history):
        result = entry.get("result")
        if not isinstance(result, dict):
            continue
        elements = result.get("elements")
        if isinstance(elements, list):
            labels = [
                e.get("label") or e.get("id") for e in elements if isinstance(e, dict)
            ]
            sample = [label for label in labels if label]
            return {
                "answer": f"Found {len(elements)} elements.",
                "data": {"sample": sample[:5]},
            }
        adjacent = result.get("adjacent")
        if isinstance(adjacent, list):
            labels = [
                e.get("label") or e.get("id") for e in adjacent if isinstance(e, dict)
            ]
            sample = [label for label in labels if label]
            return {
                "answer": f"Found {len(adjacent)} adjacent elements.",
                "data": {"sample": sample[:5]},
            }
        results = result.get("results")
        if isinstance(results, list):
            return {
                "answer": f"Found {len(results)} traversal results.",
                "data": {"sample": results[:5]},
            }
    return None
