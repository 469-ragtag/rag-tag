"""Graph agent using PydanticAI for tool-based reasoning."""

from __future__ import annotations

import networkx as nx
from pydantic_ai import Agent

from rag_tag.llm.pydantic_ai import get_agent_model

from .graph_tools import register_graph_tools
from .models import GraphAnswer

# System prompt emphasizes tool usage and structured reasoning
SYSTEM_PROMPT = """
You are a graph-reasoning agent for an IFC knowledge graph.

Schema (no direct graph access):
- Nodes have attributes: label, class_, properties, geometry
- Edges have attributes: relation, distance
- Hierarchy can include: Project > Site > Building > Storey > Space > Elements
- Some levels may be missing; use labels and class_ to identify nodes
- Spatial adjacency edges exist with relation = "adjacent_to"
- Topology edges can include: "above", "below", "overlaps_xy", "intersects_bbox"

Available tools:
- find_nodes: Search by IFC class and/or properties
- traverse: Follow edges from a starting node
- spatial_query: Find elements near a reference element
- get_elements_in_storey: Get all elements in a level/storey
- find_elements_by_class: Get all elements of an IFC class
- get_adjacent_elements: Get spatially adjacent elements
- get_topology_neighbors: Get neighbors via topology relation
- find_elements_above: Find elements above a reference element
- find_elements_below: Find elements below a reference element

Tool results are wrapped in an envelope:
- status: "ok" or "error"
- data: payload on success, null on error
- error: object with message/code/details on error

Use only the data field for reasoning. Call tools to gather information,
then synthesize a clear answer. For list results, provide a count and sample.
Prefer topology tools (above/below/overlaps/intersects) when questions ask
about vertical relations, overlap, or intersection. Use spatial distance
tools as fallback when topology facts are not sufficient.

Output format (REQUIRED SCHEMA):
You must return a JSON object with exactly these fields:
- "answer" (string, required): Natural language answer to the question
- "data" (object, optional): Structured data like counts, IDs, samples
- "warning" (string, optional): Warning message if applicable

Do NOT include any extra keys. Only use these three fields.
""".strip()


class GraphAgent:
    """Graph agent using PydanticAI with tool calling."""

    def __init__(self, *, debug_llm_io: bool = False) -> None:
        """Initialize graph agent with PydanticAI.

        Args:
            debug_llm_io: Enable debug printing (not implemented for PydanticAI)
        """
        self._debug_llm_io = debug_llm_io

        # Create PydanticAI agent with tools and structured output
        model = get_agent_model()
        self._agent: Agent[nx.DiGraph, GraphAnswer] = Agent(
            model,
            deps_type=nx.DiGraph,
            output_type=GraphAnswer,
            system_prompt=SYSTEM_PROMPT,
            retries=2,  # Tool call retries
            output_retries=3,  # Output validation retries
        )

        # Register graph query tools
        register_graph_tools(self._agent)

    def run(
        self,
        question: str,
        graph: nx.DiGraph,
        *,
        max_steps: int = 6,
        trace: object | None = None,
        run_id: str | None = None,
    ) -> dict[str, object]:
        """Execute agent workflow with tool calls.

        Args:
            question: User question
            graph: NetworkX graph to query (passed as dependency)
            max_steps: Maximum reasoning steps (NOTE: PydanticAI manages
                this internally)
            trace: Ignored (legacy parameter, Logfire used instead)
            run_id: Ignored (legacy parameter, Logfire used instead)

        Returns:
            Result dict with 'answer' or 'error' key, compatible with tui.py

        Note:
            max_steps parameter is kept for API compatibility but PydanticAI
            doesn't expose direct control over iteration count. The model will
            continue calling tools until it produces a final answer or hits
            internal limits.
        """
        # NOTE: PydanticAI doesn't expose max_steps control directly.
        # Tool calling continues until the model produces final output.
        # If we need hard limits, we could:
        # 1. Implement a custom result validator that counts tool calls
        # 2. Use a timeout mechanism
        # For now, rely on PydanticAI's internal management.

        try:
            result = self._agent.run_sync(question, deps=graph)

            # Extract structured output
            output = result.output

            # Build response dict compatible with tui.py
            response: dict[str, object] = {"answer": output.answer}

            if output.data:
                response["data"] = output.data

            if output.warning:
                response["warning"] = output.warning

            return response

        except Exception as exc:
            error_msg = f"Agent execution failed: {exc}"
            return {"error": error_msg}
