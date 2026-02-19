"""Graph agent using PydanticAI for tool-based reasoning."""

from __future__ import annotations

import networkx as nx
from pydantic_ai import Agent, UnexpectedModelBehavior

from rag_tag.llm.pydantic_ai import get_agent_model

from .graph_tools import register_graph_tools
from .models import GraphAnswer

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """
You are a graph-reasoning agent for an IFC (Industry Foundation Classes)
knowledge graph. You answer natural-language questions by calling tools and
synthesising results into the required JSON schema.

IFC ontology rules:
1. IFC class names are CamelCase with no spaces (e.g. IfcWall, IfcDoor,
   IfcSpace, IfcBuildingStorey).
2. Multi-word phrases (e.g. "plumbing wall", "structural column") are names
   or descriptions, not class names. Use fuzzy_find_nodes for those.
3. PredefinedType is an enum stored in node.properties (e.g. WALL, DOOR,
   BASESLAB). It is not a separate IFC class.
4. ObjectType and Description are free-text fields in node.properties.

Schema:
- Nodes: label, class_, properties, geometry
- Edges: relation, distance
- Hierarchy may include Project > Site > Building > Storey > Space > Elements
- Some levels may be missing; use labels and class_ to identify nodes
- Spatial adjacency uses relation="adjacent_to"
- Topology relations can include above, below, overlaps_xy, intersects_bbox,
  intersects_3d, touches_surface

Location/storey queries:
- Location is represented by edges, not a node property.
- Use traverse(start=<element_id>, relation="contained_in", depth=3)
  and inspect IfcBuildingStorey nodes.

Available tools:
- find_nodes
- fuzzy_find_nodes
- traverse
- spatial_query
- get_elements_in_storey
- find_elements_by_class
- get_adjacent_elements
- get_topology_neighbors
- get_intersections_3d
- find_elements_above
- find_elements_below
- list_property_keys

Tool results use this envelope:
{ "status": "ok"|"error", "data": <payload or null>, "error": <null or obj> }
Use only the data field for reasoning. If status is error, try another tool
path or report the limitation in warning.

Reasoning guidance:
- For vertical relation / overlap / intersection / contact questions, prefer
  topology tools first.
- Use spatial distance tools as fallback when topology facts are insufficient.
- For list outputs, include a count and a representative sample.

Fallback chain:
1. Try find_nodes with a normalised IFC class name.
2. If empty, call fuzzy_find_nodes with the original phrase.
3. If still empty, relax optional filters and retry.
4. If still empty, return best partial answer and set warning.
5. Never refuse to answer; always return the required schema.

Output format (required, valid JSON, no markdown fences):
{
  "answer": "<natural language answer>",
  "data": { ... } or null,
  "warning": "<message if applicable>" or null
}

- "answer" is required and non-empty.
- "data" is optional structured payload (counts, IDs, sample elements).
- "warning" is optional and used for partial results or fallback notices.
- Do not add extra keys.
""".strip()


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class GraphAgent:
    """Graph agent using PydanticAI with tool calling."""

    def __init__(self, *, debug_llm_io: bool = False) -> None:
        """Initialise graph agent with PydanticAI.

        Args:
            debug_llm_io: Enable debug printing (unused for PydanticAI;
                          kept for API compatibility).
        """
        self._debug_llm_io = debug_llm_io

        model = get_agent_model()
        self._agent: Agent[nx.DiGraph, GraphAnswer] = Agent(
            model,
            deps_type=nx.DiGraph,
            output_type=GraphAnswer,
            system_prompt=SYSTEM_PROMPT,
            retries=2,
            # Extra retries specifically for output schema validation.
            # Increased from 3 to 5 to give the model more chances to
            # produce valid JSON when it initially returns prose or extra keys.
            output_retries=5,
        )

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
            question: User question.
            graph: NetworkX graph to query (passed as dependency).
            max_steps: Kept for API compatibility; PydanticAI manages
                iteration count internally.
            trace: Ignored (legacy; Logfire used instead).
            run_id: Ignored (legacy; Logfire used instead).

        Returns:
            Result dict with 'answer' / 'data' / 'warning' keys, or an
            'error' key if the agent run fails entirely.
        """
        try:
            result = self._agent.run_sync(question, deps=graph)
            output = result.output

            response: dict[str, object] = {"answer": output.answer}
            if output.data:
                response["data"] = output.data
            if output.warning:
                response["warning"] = output.warning
            return response

        except UnexpectedModelBehavior as exc:
            # The model failed to produce a valid structured output even after
            # all output_retries. Surface a safe fallback so the CLI always
            # has something to display.
            raw = getattr(exc, "body", None) or str(exc)
            raw_snippet = str(raw)[:400] if raw else ""
            return {
                "answer": (
                    "I was unable to produce a well-structured answer for this "
                    "question. Please try rephrasing or ask a simpler question."
                ),
                "warning": (f"Output validation failed after all retries: {exc}"),
                "data": {"raw_response_snippet": raw_snippet} if raw_snippet else None,
            }

        except Exception as exc:
            return {"error": f"Agent execution failed: {exc}"}
