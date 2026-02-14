"""Graph agent using PydanticAI with typed tools."""

from __future__ import annotations

import json
from typing import Any

import networkx as nx
from pydantic_ai import Agent, ModelRetry

from rag_tag.agent.graph_tools import (
    find_elements_by_class,
    find_nodes,
    get_adjacent_elements,
    get_elements_in_storey,
    resolve_entity_reference,
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
- resolve_entity_reference: Resolve vague names/aliases (name/id/guid)
    to likely graph nodes

Tool results are wrapped in an envelope:
- status: "ok" or "error"
- data: payload on success, null on error
- error: object with message/code/details on error

Use the data field for reasoning. If status is "error", explain the issue to the user.

Entity resolution policy:
- Always try resolving user-mentioned entities with tools before asking clarification.
- Start reference resolution with `resolve_entity_reference` for vague terms.
- For terms like "house", first search likely containers
    (`IfcBuilding`, `IfcProject`, `IfcSite`) and use best match.
- If resolver returns candidates, pick the top candidate and continue.
- If an explicit ID/GlobalId is missing, attempt lookup by class/name
    synonyms via `find_nodes`.
- Only ask the user for clarification after at least one concrete lookup attempt fails.
- For vague language (e.g., "near", "crowded"), make reasonable defaults
    and state assumptions.
- Defaults when unspecified: near=5.0m, crowded=3+ adjacent elements.
- Never ask the user to provide ID/GlobalId/GUID if tools can resolve candidates.

Return a concise, accurate answer based on the tool results.
Output format:
- Return only one final answer field.
- Prefer JSON object: {"answer": "..."}.
- If you return plain text, make it just the final answer sentence.
""".strip()

FALLBACK_PROMPT = """
You are a graph-reasoning agent for an IFC knowledge graph.

Use the available tools to answer the user's question accurately.
Return plain text only (no JSON, no markdown), as one concise final answer.
If tool status is "error", explain the issue briefly.
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
        self._provider_name = provider_name or "auto"
        self._model_label = self._describe_model(model)

        self._agent = Agent(
            model,
            output_type=GraphAnswer,
            instructions=SYSTEM_PROMPT,
            deps_type=nx.DiGraph,
            output_retries=3,
        )

        self._fallback_agent = Agent(
            model,
            output_type=str,
            instructions=FALLBACK_PROMPT,
            deps_type=nx.DiGraph,
        )

        self._register_tools(self._agent)
        self._register_tools(self._fallback_agent)

        @self._agent.output_validator
        def _no_id_guid_request(data: GraphAnswer) -> GraphAnswer:
            text = data.answer.lower()
            asks_for_ids = "provide" in text and (
                " id" in text
                or "guid" in text
                or "globalid" in text
                or "global id" in text
            )
            if asks_for_ids:
                raise ModelRetry(
                    "Resolve entity references using tools and continue with the "
                    "best candidate. Do not ask for ID/GUID."
                )
            return data

        @self._fallback_agent.output_validator
        def _no_id_guid_request_fallback(answer: str) -> str:
            text = answer.lower()
            asks_for_ids = "provide" in text and (
                " id" in text
                or "guid" in text
                or "globalid" in text
                or "global id" in text
            )
            if asks_for_ids:
                raise ModelRetry(
                    "Resolve entity references using tools and continue with the "
                    "best candidate. Do not ask for ID/GUID."
                )
            return answer

    @staticmethod
    def _register_tools(agent: Agent) -> None:
        """Register graph tools on a PydanticAI agent."""
        agent.tool(find_nodes, name="find_nodes")
        agent.tool(traverse, name="traverse")
        agent.tool(spatial_query, name="spatial_query")
        agent.tool(get_elements_in_storey, name="get_elements_in_storey")
        agent.tool(find_elements_by_class, name="find_elements_by_class")
        agent.tool(get_adjacent_elements, name="get_adjacent_elements")
        agent.tool(resolve_entity_reference, name="resolve_entity_reference")

    @staticmethod
    def _describe_model(model: object) -> str:
        """Best-effort model description for debugging."""
        for attr in ("model_name", "name"):
            value = getattr(model, attr, None)
            if isinstance(value, str) and value.strip():
                return value
        return repr(model)

    @staticmethod
    def _extract_messages(result: Any) -> Any:
        """Extract serialized model messages when available."""
        try:
            raw = result.new_messages_json()
            if isinstance(raw, bytes):
                return json.loads(raw.decode("utf-8"))
        except Exception:
            return None
        return None

    def _build_debug_payload(
        self,
        *,
        question: str,
        result: Any,
        answer: str,
        mode: str,
        fallback_used: bool,
        structured_error: str | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "component": "agent",
            "provider": self._provider_name,
            "model": self._model_label,
            "mode": mode,
            "fallback_used": fallback_used,
            "input": {
                "question": question,
                "instructions": (
                    FALLBACK_PROMPT if mode == "fallback" else SYSTEM_PROMPT
                ),
            },
            "output": {"answer": answer},
            "messages": self._extract_messages(result),
        }
        if structured_error:
            payload["structured_error"] = structured_error
        return payload

    @staticmethod
    def _is_output_validation_failure(exc: Exception) -> bool:
        """Detect output-validation failures that warrant plaintext fallback."""
        message = str(exc).lower()
        return "output validation" in message or "validation" in message

    async def run_async(
        self,
        question: str,
        graph: nx.DiGraph,
        *,
        max_steps: int = 6,
        debug_llm_io: bool = False,
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
            payload: dict[str, object] = {"answer": result.output.answer}
            if debug_llm_io:
                payload["llm_debug"] = self._build_debug_payload(
                    question=question,
                    result=result,
                    answer=result.output.answer,
                    mode="structured",
                    fallback_used=False,
                )
            return payload
        except Exception as exc:
            if self._is_output_validation_failure(exc):
                try:
                    fallback_result = await self._fallback_agent.run(
                        question, deps=graph
                    )
                    answer = fallback_result.output.strip()
                    if not answer:
                        return {"error": "Fallback agent returned empty answer."}
                    payload: dict[str, object] = {"answer": answer}
                    if debug_llm_io:
                        payload["llm_debug"] = self._build_debug_payload(
                            question=question,
                            result=fallback_result,
                            answer=answer,
                            mode="fallback",
                            fallback_used=True,
                            structured_error=f"{exc.__class__.__name__}: {exc}",
                        )
                    return payload
                except Exception as fallback_exc:
                    fallback_name = fallback_exc.__class__.__name__
                    return {
                        "error": (
                            "Structured output failed and fallback failed "
                            f"({fallback_name}): {fallback_exc}"
                        )
                    }
            error_name = exc.__class__.__name__
            return {"error": f"{error_name}: {exc}"}

    def run(
        self,
        question: str,
        graph: nx.DiGraph,
        *,
        max_steps: int = 6,
        trace: object | None = None,
        run_id: str | None = None,
        debug_llm_io: bool = False,
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

        def _ensure_open_loop() -> None:
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                return
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            running_loop = None

        if running_loop is not None and running_loop.is_running():
            import nest_asyncio

            nest_asyncio.apply()
            try:
                return asyncio.get_event_loop().run_until_complete(
                    self.run_async(
                        question,
                        graph,
                        max_steps=max_steps,
                        debug_llm_io=debug_llm_io,
                    )
                )
            except Exception as exc:
                return {"error": f"Agent execution failed: {exc}"}

        _ensure_open_loop()
        try:
            result = self._agent.run_sync(question, deps=graph)
            payload: dict[str, object] = {"answer": result.output.answer}
            if debug_llm_io:
                payload["llm_debug"] = self._build_debug_payload(
                    question=question,
                    result=result,
                    answer=result.output.answer,
                    mode="structured",
                    fallback_used=False,
                )
            return payload
        except Exception as exc:
            if self._is_output_validation_failure(exc):
                try:
                    fallback_result = self._fallback_agent.run_sync(
                        question, deps=graph
                    )
                    answer = fallback_result.output.strip()
                    if answer:
                        payload = {"answer": answer}
                        if debug_llm_io:
                            payload["llm_debug"] = self._build_debug_payload(
                                question=question,
                                result=fallback_result,
                                answer=answer,
                                mode="fallback",
                                fallback_used=True,
                                structured_error=f"{exc.__class__.__name__}: {exc}",
                            )
                        return payload
                    return {"error": "Fallback agent returned empty answer."}
                except Exception as fallback_exc:
                    fallback_name = fallback_exc.__class__.__name__
                    return {
                        "error": (
                            "Agent failed: structured output and fallback failed "
                            f"({fallback_name}): {fallback_exc}"
                        )
                    }
            error_name = exc.__class__.__name__
            return {"error": f"Agent execution failed ({error_name}): {exc}"}
