"""Graph agent using PydanticAI with typed tools."""

from __future__ import annotations

import networkx as nx
from pydantic_ai import Agent

from rag_tag.agent.graph_tools import (
    find_elements_by_class,
    find_nodes,
    get_adjacent_elements,
    get_elements_in_storey,
    spatial_query,
    traverse,
)
from rag_tag.agent.models import GraphAnswer
from rag_tag.llm.pydantic_ai import resolve_model

SYSTEM_PROMPT = """
You are a graph-reasoning agent for an IFC knowledge graph.

Schema:
- Nodes have attributes: label, class_, properties, geometry
- Edges have attributes: relation, distance
- Hierarchy can include: Project > Site > Building > Storey > Space > Elements
- Some levels may be missing; use labels and class_ to identify nodes
- Spatial adjacency edges exist with relation = "adjacent_to"

Available tools:
- find_nodes: Find nodes by class and optional property filters
- traverse: Traverse graph from a start node following edges
- spatial_query: Find elements within distance of a reference element
- get_elements_in_storey: Get all elements in a building storey
- find_elements_by_class: Find all elements of a specific IFC class
- get_adjacent_elements: Get elements adjacent to a given element

Tool results are wrapped in an envelope:
- status: "ok" or "error"
- data: payload on success, null on error
- error: object with message/code/details on error

Use the data field for reasoning. If status is "error", explain the issue to the user.

Return a concise, accurate answer based on the tool results.
""".strip()


class GraphAgent:
    """Graph agent using PydanticAI with typed tools."""

    def __init__(
        self,
        model_name: str | None = None,
        provider_name: str | None = None,
    ) -> None:
        """Initialize graph agent with PydanticAI.

        Args:
            model_name: Model name override (e.g., 'command-a-03-2025')
            provider_name: Provider name (cohere, gemini, openai)
        """
        model = resolve_model(provider_name, model_name=model_name, purpose="agent")

        self._agent = Agent(
            model,
            output_type=GraphAnswer,
            instructions=SYSTEM_PROMPT,
            deps_type=nx.DiGraph,
        )

        self._agent.tool(find_nodes)
        self._agent.tool(traverse)
        self._agent.tool(spatial_query)
        self._agent.tool(get_elements_in_storey)
        self._agent.tool(find_elements_by_class)
        self._agent.tool(get_adjacent_elements)

    async def run_async(
        self,
        question: str,
        graph: nx.DiGraph,
        *,
        max_steps: int = 6,
    ) -> dict[str, object]:
        """Execute agent workflow loop (async).

        Args:
            question: User question
            graph: NetworkX graph to query
            max_steps: Maximum reasoning steps (not directly enforced by PydanticAI)

        Returns:
            Result dict with 'answer' or 'error' key
        """
        try:
            result = await self._agent.run(question, deps=graph)
            return {"answer": result.output.answer}
        except Exception as exc:
            return {"error": str(exc)}

    def run(
        self,
        question: str,
        graph: nx.DiGraph,
        *,
        max_steps: int = 6,
        trace: object | None = None,
        run_id: str | None = None,
    ) -> dict[str, object]:
        """Execute agent workflow loop (sync wrapper).

        Args:
            question: User question
            graph: NetworkX graph to query
            max_steps: Maximum reasoning steps
            trace: Legacy trace parameter (ignored, Logfire used instead)
            run_id: Legacy run_id parameter (ignored, Logfire used instead)

        Returns:
            Result dict with 'answer' or 'error' key
        """
        import asyncio

        def _run_in_new_loop() -> dict[str, object]:
            loop = asyncio.new_event_loop()
            try:
                asyncio.set_event_loop(loop)
                return loop.run_until_complete(
                    self.run_async(question, graph, max_steps=max_steps)
                )
            finally:
                try:
                    loop.run_until_complete(loop.shutdown_asyncgens())
                finally:
                    asyncio.set_event_loop(None)
                    loop.close()

        try:
            return asyncio.run(self.run_async(question, graph, max_steps=max_steps))
        except RuntimeError as exc:
            message = str(exc)
            if "cannot be called from a running event loop" in message:
                # Already inside a running loop (e.g. Jupyter) -- patch and retry
                import nest_asyncio

                nest_asyncio.apply()
                return asyncio.get_event_loop().run_until_complete(
                    self.run_async(question, graph, max_steps=max_steps)
                )
            if "no current event loop" in message.lower():
                return _run_in_new_loop()
            return {"error": f"Agent execution failed: {exc}"}
        except Exception as exc:
            return {"error": f"Agent execution failed: {exc}"}
