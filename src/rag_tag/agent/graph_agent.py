"""Graph agent using PydanticAI for tool-based reasoning."""

from __future__ import annotations

import re

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
knowledge graph.  You answer natural-language questions by calling tools and
synthesising results into the required JSON schema.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
IFC ONTOLOGY RULES (read carefully)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. IFC CLASS names are always CamelCase with NO spaces:
   IfcWall  IfcDoor  IfcColumn  IfcSlab  IfcBeam  IfcWindow
   IfcSpace  IfcBuildingStorey  IfcBuilding  IfcSite  IfcProject
   IfcRoof  IfcStair  IfcRailing  IfcPlate  IfcMember  IfcFurnishingElement

2. Multi-word phrases (e.g. "plumbing wall", "structural column") are
   NAMES or DESCRIPTIONS, NOT class names.  When the user asks about
   "plumbing walls" use fuzzy_find_nodes with that phrase as the query.
   NEVER pass a phrase containing spaces to find_nodes as class_.

3. PredefinedType is an ENUM stored in node.properties (e.g. "WALL",
   "DOOR", "BASESLAB").  It is NOT a separate IFC class.

4. ObjectType and Description are free-text fields inside node.properties.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
NODE SCHEMA
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

- id        : graph node key (e.g. "Element::abc123", "Storey::<GlobalId>")
- label     : human-readable element name
- class_    : IFC class (CamelCase, no spaces)
- properties: dict — GlobalId, ObjectType, Description, PredefinedType, …

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EDGE RELATIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

- "contains"      → container to child  (storey → wall, space → door …)
- "contained_in"  → child to its spatial container  (wall → storey)
- "adjacent_to"   → spatial proximity between elements
- "typed_by"      → element to its IFC type object

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LOCATION / FLOOR / STOREY QUERIES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Location is an EDGE, not a property.  To find which storey element X is on:

  traverse(start=<element_id>, relation="contained_in", depth=3)

Look for IfcBuildingStorey nodes in the results.
Do NOT search for a "storey" property — it does not exist.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
AVAILABLE TOOLS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

find_nodes           Search by IFC class and/or property filters.
                     Auto-fuzzy-normalises class_; falls back to fuzzy search
                     when exact query is empty.

fuzzy_find_nodes     Score-ranked text search over Name/ObjectType/Description.
                     Use for descriptive phrases or when find_nodes returns nothing.

traverse             Follow graph edges from a node.
                     Use relation='contained_in' for location/floor queries.

spatial_query        Elements within a given distance of a reference element.

get_elements_in_storey  All non-container elements in a named storey.

find_elements_by_class  All elements of a given IFC class.

get_adjacent_elements   Directly adjacent elements of one element.

get_topology_neighbors  Topology neighbors by one relation
                        (above, below, overlaps_xy, intersects_bbox,
                        intersects_3d, touches_surface).

get_intersections_3d    Exact 3D mesh-informed intersection neighbors.

find_elements_above     Elements above a reference element.

find_elements_below     Elements below a reference element.

list_property_keys   Discover valid property key names (use before
                     setting property_filters in find_nodes).

Tool results use a standard envelope:
  { "status": "ok"|"error", "data": <payload or null>, "error": <null or obj> }
Use only the data field for reasoning.  Treat status="error" as a signal to
try an alternative approach or report the issue in warning.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FALLBACK CHAIN  (follow in order)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. Try find_nodes with a normalised IFC class name.
2. If result is empty, call fuzzy_find_nodes with the original phrase.
3. If fuzzy also returns nothing, drop optional filters and retry.
4. If still nothing, state the limitation in the "warning" field and give
   the best partial answer you can.
5. NEVER refuse to answer.  Always return the required JSON schema.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT FORMAT  ← REQUIRED — always valid JSON, no markdown fences
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Return EXACTLY this JSON object (no extra keys):

{
  "answer":  "<natural language answer>",
  "data":    { ... } or null,
  "warning": "<message if applicable>" or null
}

- "answer" is REQUIRED and must be a non-empty string.
- "data" is optional structured payload (counts, IDs, sample elements …).
- "warning" is optional; use it for partial results or fallback notices.
- Do NOT wrap in markdown fences.  Do NOT add extra keys.
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
            answer = _sanitize_model_text(output.answer) or ""
            warning = _sanitize_model_text(output.warning)

            response: dict[str, object] = {"answer": answer}
            if output.data:
                response["data"] = output.data
            if warning:
                response["warning"] = warning
            response["answer"] = _polish_answer_with_data(
                str(response.get("answer", "")),
                response.get("data"),
            )
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


def _polish_answer_with_data(answer: str, data: object | None) -> str:
    """Improve terse list-preface answers using available structured data."""
    if not answer or not isinstance(data, dict):
        return answer

    values = _extract_primary_list(data)
    if not values:
        return answer

    trimmed = answer.rstrip()
    if not trimmed.endswith(":"):
        return answer

    shown = ", ".join(values[:3])
    return f"{trimmed[:-1]} ({len(values)}): {shown}"


def _sanitize_model_text(value: object | None) -> str | None:
    """Strip provider annotation tags and normalize whitespace."""
    if value is None:
        return None
    text = str(value)
    text = re.sub(r"</?co:[^>]+>", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text or None


def _extract_primary_list(data: dict[str, object]) -> list[str]:
    keys = (
        "overlapping_elements",
        "elements",
        "neighbors",
        "results",
        "matches",
        "sample",
        "adjacent",
        "intersections_3d",
    )
    for key in keys:
        raw = data.get(key)
        if not isinstance(raw, list) or not raw:
            continue
        values: list[str] = []
        for item in raw:
            if isinstance(item, str):
                values.append(item)
            elif isinstance(item, dict):
                label = item.get("label")
                node_id = item.get("id")
                if isinstance(label, str) and label.strip():
                    values.append(label.strip())
                elif isinstance(node_id, str) and node_id.strip():
                    values.append(node_id.strip())
        if values:
            return values
    return []
