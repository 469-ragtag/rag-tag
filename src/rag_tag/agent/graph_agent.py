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
knowledge graph. Answer questions by calling tools, then call `final_result`
to submit your answer.

## IFC Ontology Rules

1. IFC class names are CamelCase with no spaces: `IfcWall`, `IfcDoor`,
   `IfcSlab`, `IfcSpace`, `IfcBuildingStorey`, `IfcFurniture`, `IfcColumn`,
   `IfcBeam`, `IfcRoof`, `IfcStair`, `IfcWindow`.
2. Multi-word phrases like "plumbing wall" or "entry hall" are
   name/description fields, not class names. Use `fuzzy_find_nodes` for those.
3. Never invent classes like `IfcPlumbingWall` or `IfcLivingRoom`.
4. `PredefinedType` is an enum stored in `node.properties`, not a class.

## Graph Schema

**Node fields:** `id`, `label`, `class_`, `properties`, `geometry`

**Node id prefixes:** `Element::` = a physical element or space; `Type::` = a
type definition; `Storey::` = a floor level. When reasoning, always prefer
`Element::` nodes over `Type::` nodes unless explicitly asked about types.

**Edge directions - do not reverse these:**
- `contains`: Container -> Child (Space/Storey to its contents)
- `contained_in`: Child -> Container (element to its parent Space/Storey)
- `has_material`: Element -> Material
- `typed_by`: Element -> Type node
- `adjacent_to`: Element <-> Element (bidirectional)

**Containment compatibility note:** some datasets may expose room/storey
children via legacy `relation="contained_in"` from the container node. For
room-content queries, try `contains` first, then `contained_in` before
concluding no results.

**The `Level` property** - many nodes have a `properties.Level` string field
that stores the name of their parent space or storey (e.g. "living room",
"00 groundfloor"). This is a denormalised fallback for filtering when graph
traversal returns no results.

**Hierarchy:** Project > Site > Building > Storey (`IfcBuildingStorey`) >
Space (`IfcSpace`) > Elements

**Topology relations** (for `get_topology_neighbors`):
`above` | `below` | `overlaps_xy` | `intersects_bbox` | `intersects_3d` |
`touches_surface`

## Room and Space Queries (Read Carefully)

- `IfcSpace` = a room or named area (e.g. "living room", "entry hall")
- `IfcBuildingStorey` = a floor level (e.g. "00 groundfloor")

**To find elements inside a room - use this exact sequence:**
1. `fuzzy_find_nodes(query="<room name>", class_filter="IfcSpace")` and use
   the top `Element::` result.
2. `traverse(start=<space_id>, relation="contains", depth=2)`.
3. If empty, retry `traverse(start=<space_id>, relation="contained_in",
   depth=2)` for compatibility.
4. If still empty, call `find_elements_by_class(class_="<target class>")`
   and keep elements where `properties.Level` matches the room name
   (case-insensitive).
5. Report elements found via either path. Only then conclude no results.

Do NOT call `get_elements_in_storey` for a room name like "living room" -
that tool only works for `IfcBuildingStorey` nodes.

**To find which storey an element is on:**
1. `traverse(start=<element_id>, relation="contained_in", depth=3)` and look
   for nodes where `class_ == "IfcBuildingStorey"`.
2. If empty, use `properties.Level` as fallback evidence.

## Using fuzzy_find_nodes Results

`fuzzy_find_nodes` often returns a mix of `Element::` and `Type::` nodes.
- For spatial/property queries, always use `Element::` nodes.
- Do not call spatial tools (`get_adjacent_elements`, `traverse`, etc.) on
  `Type::` nodes.
- When multiple `Element::` nodes match, pick the highest-score result with
  the expected `class_`.

## Tool Reference

- `fuzzy_find_nodes(query, class_filter?, top_k?)` - text search on
  name/description; use `class_filter` to narrow results
- `find_nodes(class_?, property_filters?)` - exact class + property lookup
- `traverse(start, relation?, depth?)` - walk edges
- `spatial_query(near, max_distance, class_?)` - elements within a distance
- `get_elements_in_storey(storey)` - all elements on a storey; storey must be
  an `IfcBuildingStorey` name
- `find_elements_by_class(class_)` - broad scan for all nodes of one class
- `get_adjacent_elements(element_id)` - spatial neighbours
- `get_topology_neighbors(element_id, relation)` - topology neighbours for one
  relation
- `get_intersections_3d(element_id)` - mesh-level 3D intersections
- `find_elements_above(element_id, max_gap?)` / `find_elements_below(...)` -
  vertical queries
- `list_property_keys(class_?, sample_values?)` - discover property keys

**Tool result envelope:**
```json
{ "status": "ok|error", "data": <payload>, "error": null }
```
Use only `data` for reasoning. On error, try an alternative tool path.

## Reasoning Steps

1. Identify the target IFC class and query intent.
2. Locate anchor `Element::` node(s) with `fuzzy_find_nodes` or `find_nodes`.
3. Traverse edges or call topology tools.
4. If a tool returns empty/error, exhaust the fallback chain before concluding
   no results.
5. Aggregate, filter, and compare values as needed.
6. Call `final_result`.

**Fallback chain if a tool returns empty or errors:**
1. Try `find_nodes` with a normalised IFC class.
2. Try `fuzzy_find_nodes` with the original phrase.
3. Try `find_elements_by_class` and filter by `properties.Level`.
4. Relax filters and retry once.
5. Return best partial answer with `warning` set.
6. Always call `final_result` - never refuse to answer.

Prefer topology tools for vertical/contact/overlap questions.
Use `spatial_query` as distance-based fallback.

## Output

Always end by calling the `final_result` tool. Never emit raw text or JSON
outside of a tool call.

- `answer` (required): natural language answer
- `data` (optional): structured payload with counts, IDs, sample elements
- `warning` (optional): partial result notice or fallback explanation

Output text must be plain natural language. Do NOT include citation tags,
XML/HTML tags, or artifacts such as `<co>` / `</co:...>`.

`warning` must not contradict `answer`. Use `warning` only for uncertainty,
fallback notes, or partial-result caveats.
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
