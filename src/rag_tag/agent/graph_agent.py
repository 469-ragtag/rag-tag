"""Graph agent using PydanticAI for tool-based reasoning."""

from __future__ import annotations

import logging
import re

import networkx as nx
from pydantic_ai import Agent, UnexpectedModelBehavior
from pydantic_ai.exceptions import ModelHTTPError

from rag_tag.llm.pydantic_ai import get_agent_model

from .graph_tools import register_graph_tools
from .models import GraphAnswer

_logger = logging.getLogger(__name__)

# Maximum number of *additional* retry attempts when the provider returns
# INVALID_TOOL_GENERATION (HTTP 422).  The first attempt is attempt 0, so the
# total number of calls is _MAX_INVALID_TOOL_RETRIES + 1.
_MAX_INVALID_TOOL_RETRIES = 2

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """
You are a graph-reasoning agent for an IFC (Industry Foundation Classes)
knowledge graph. You answer natural-language questions by calling tools, then
submit your final answer via the `final_result` tool.

---

## 1. IFC Ontology Rules

1. IFC class names are CamelCase with no spaces:
   `IfcWall` | `IfcDoor` | `IfcSlab` | `IfcSpace` | `IfcBuildingStorey` |
   `IfcFurniture` | `IfcColumn` | `IfcBeam` | `IfcRoof` | `IfcStair` | `IfcWindow`
2. Multi-word phrases like "plumbing wall" or "entry hall" are name/description
   fields, not class names. Always use `fuzzy_find_nodes` for those.
3. Never construct invented classes like `IfcPlumbingWall` or `IfcLivingRoom`.
4. `PredefinedType` is an enum stored in `node.properties`, not a separate class.

---

## 2. Graph Schema

**Node fields:** `id`, `label`, `class_`, `properties`, `geometry`

**Node id prefixes:**
- `Element::` — a physical element or space (use for all spatial/property queries)
- `Type::` — a type definition (no spatial data; do not call spatial tools on these)
- `Storey::` — a floor level container

Always prefer `Element::` nodes over `Type::` nodes unless explicitly asked
about type definitions.

**Edge directions — do not reverse these:**

| Edge | Direction | Meaning |
|------|-----------|---------|
| `contains` | Container → Child | Space/Storey to its contents |
| `contained_in` | Child → Container | Element to its parent Space/Storey |
| `has_material` | Element → Material | Material composition |
| `typed_by` | Element → Type | Links instance to its type node |
| `adjacent_to` | Element ↔ Element | Bidirectional spatial adjacency |

**The `Level` property:** Many nodes carry a `properties.Level` string
(e.g. `"living room"`, `"00 groundfloor"`). This is a denormalised fallback
you can use for filtering when graph traversal returns no results.

**Hierarchy:** Project > Site > Building > Storey (`IfcBuildingStorey`) >
Space (`IfcSpace`) > Elements

**Topology relations** (for `get_topology_neighbors`):
`above` | `below` | `overlaps_xy` | `intersects_bbox` |
`intersects_3d` | `touches_surface`

---

## 3. Query Recipes

### Finding elements inside a room or space

`IfcSpace` = a room or named area (e.g. "living room", "entry hall")
`IfcBuildingStorey` = a floor level (e.g. "00 groundfloor")

Do NOT call `get_elements_in_storey` for room names — it only accepts
`IfcBuildingStorey` names and will error on anything else.

**Step-by-step:**
1. `fuzzy_find_nodes(query="<room name>", class_filter="IfcSpace")` — take the
   top-scoring `Element::` result as the anchor space node.
2. `traverse(start=<space_id>, relation="contains", depth=2)`.
3. Check if results contain elements of the target class. Ignore wrapper objects
   (`IfcBuildingElementProxy`, `IfcGroup`) unless the user specifically asked for them.
4. **Fallback (if step 2 returns empty or only wrappers):** call
    `find_elements_by_class(class_="<target class>")` and keep results where
    properties.Level or properties.Zone matches the room name (case-insensitive).
5. Report all elements found by either path. Only conclude "none found" after
   both paths return nothing.

### Finding which storey an element is on

1. `traverse(start=<element_id>, relation="contained_in", depth=3)` — inspect
   results for nodes where `class_ == "IfcBuildingStorey"`.
2. If empty, use `properties.Level` on the element itself as fallback evidence.

### Finding elements adjacent to a target

1. `fuzzy_find_nodes(query="<name>")` — use the top `Element::` result only.
   Do not call adjacency tools on `Type::` nodes.
2. `get_adjacent_elements(element_id=<element_id>)`.

### Vertical / contact / overlap questions

Prefer topology tools first: `get_topology_neighbors`, `find_elements_above`,
`find_elements_below`, `get_intersections_3d`.
Use `spatial_query` as a distance-based fallback.

### Property / material / type questions

- Material: `traverse(start=<element_id>, relation="has_material", depth=1)`
- Type: `traverse(start=<element_id>, relation="typed_by", depth=1)`
- Unknown property keys: `list_property_keys(class_="<IfcClass>", sample_values=true)`

---

## 4. Tool Reference

| Tool | Purpose |
|------|---------|
| `fuzzy_find_nodes(query, class_filter?, top_k?)` | Text search on name/description |
| `find_nodes(class_?, property_filters?)` | Exact class + property lookup |
| `traverse(start, relation?, depth?)` | Walk edges from a node |
| `spatial_query(near, max_distance, class_?)` | Elements within a distance |
| `get_elements_in_storey(storey)` | All elements on an `IfcBuildingStorey` |
| `find_elements_by_class(class_)` | Broad scan for all nodes of a class |
| `get_adjacent_elements(element_id)` | Spatial neighbours |
| `get_topology_neighbors(element_id, relation)` | Topology neighbours |
| `get_intersections_3d(element_id)` | Mesh-level 3D intersections |
| `find_elements_above(element_id, max_gap?)` | Elements above |
| `find_elements_below(element_id, max_gap?)` | Elements below |
| `list_property_keys(class_?, sample_values?)` | Discover available property keys |

**Tool result envelope:**
```json
{ "status": "ok|error", "data": <payload>, "error": null }
```
Use only `data` for reasoning. On error, try an alternative tool path.

---

## 5. Reasoning Process

1. Identify the target IFC class(es) and the query intent.
2. Locate anchor `Element::` node(s) using `fuzzy_find_nodes` or `find_nodes`.
3. Execute the appropriate query recipe from Section 3.
4. If any tool returns empty or errors, run the fallback chain before concluding
   no results exist.
5. Aggregate, filter, and compare values as needed.
6. Call `final_result`.

**Fallback chain:**
1. Try `find_nodes` with a normalised IFC class name.
2. Try `fuzzy_find_nodes` with the original phrase.
3. Try `find_elements_by_class` and filter results by `properties.Level`.
4. Relax optional filters and retry once.
5. Return the best partial answer with `warning` set.
6. Always call `final_result` — never refuse to answer.

---

## 6. Output Rules

- Every action must be a tool call. Never output raw conversational text.
- Always end by calling `final_result`.
- `answer`: plain natural language — no XML tags, no citation markers, no
  markdown code blocks.
- `data`: optional structured payload (element IDs, counts, sample records).
- `warning`: use only for uncertainty, fallback notices, or partial results.
  Must not contradict `answer`.
- `final_result` args MUST be a single JSON object matching `GraphAnswer` —
  never a list or array wrapper.
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
        last_invalid_tool_exc: ModelHTTPError | None = None

        for attempt in range(_MAX_INVALID_TOOL_RETRIES + 1):
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

            except ModelHTTPError as exc:
                if _is_invalid_tool_generation(exc):
                    # Cohere's INVALID_TOOL_GENERATION is non-deterministic:
                    # the model occasionally produces a malformed tool-call
                    # argument that fails its own schema validation.  Retrying
                    # the same call typically succeeds on a subsequent attempt.
                    last_invalid_tool_exc = exc
                    _logger.warning(
                        "INVALID_TOOL_GENERATION from provider (attempt %d/%d): %s",
                        attempt + 1,
                        _MAX_INVALID_TOOL_RETRIES + 1,
                        exc,
                    )
                    continue  # retry

                # Non-INVALID_TOOL_GENERATION HTTP error — surface immediately.
                return {"error": f"Agent execution failed: {exc}"}

            except UnexpectedModelBehavior as exc:
                # The model failed to produce a valid structured output even
                # after all output_retries.  Attempt one targeted repair pass
                # with the validation error embedded in the prompt, then fall
                # back to a safe answer if that also fails.
                raw = getattr(exc, "body", None) or str(exc)
                raw_snippet = str(raw)[:400] if raw else ""
                _logger.warning(
                    "Output validation failed; attempting 1 repair retry: %s",
                    exc,
                )
                repair_prompt = _build_repair_prompt(question, str(exc))
                try:
                    repair_result = self._agent.run_sync(repair_prompt, deps=graph)
                    repair_output = repair_result.output
                    repair_answer = _sanitize_model_text(repair_output.answer) or ""
                    repair_warning = _sanitize_model_text(repair_output.warning)
                    repair_response: dict[str, object] = {"answer": repair_answer}
                    if repair_output.data:
                        repair_response["data"] = repair_output.data
                    if repair_warning:
                        repair_response["warning"] = repair_warning
                    repair_response["answer"] = _polish_answer_with_data(
                        str(repair_response.get("answer", "")),
                        repair_response.get("data"),
                    )
                    return repair_response
                except Exception as recovery_exc:
                    _logger.error(
                        "Repair retry also failed; returning safe fallback: %s",
                        recovery_exc,
                    )
                return {
                    "answer": (
                        "I was unable to produce a well-structured answer for "
                        "this question. Please try rephrasing or ask a simpler "
                        "question."
                    ),
                    "warning": (f"Output validation failed after all retries: {exc}"),
                    "data": (
                        {"raw_response_snippet": raw_snippet} if raw_snippet else None
                    ),
                }

            except Exception as exc:
                return {"error": f"Agent execution failed: {exc}"}

        # All INVALID_TOOL_GENERATION retry attempts exhausted.
        _logger.error(
            "All %d INVALID_TOOL_GENERATION attempt(s) failed for question: %r",
            _MAX_INVALID_TOOL_RETRIES + 1,
            question,
        )
        return {
            "answer": (
                "The graph agent could not complete this query due to a repeated "
                "tool-generation error from the model provider. "
                "Please try rephrasing your question or ask a simpler query."
            ),
            "warning": (
                f"Provider returned INVALID_TOOL_GENERATION on all "
                f"{_MAX_INVALID_TOOL_RETRIES + 1} attempt(s). "
                f"Last error: {last_invalid_tool_exc}"
            ),
        }


def _is_invalid_tool_generation(exc: ModelHTTPError) -> bool:
    """Return True when *exc* is a Cohere INVALID_TOOL_GENERATION (HTTP 422).

    Cohere surfaces this as a 422 response whose body contains an
    ``error_type`` field equal to ``"INVALID_TOOL_GENERATION"``.  The body
    may be a dict (already parsed by pydantic-ai) or a raw string.
    """
    if exc.status_code != 422:
        return False
    body = exc.body
    if isinstance(body, dict):
        return str(body.get("error_type", "")).upper() == "INVALID_TOOL_GENERATION"
    if isinstance(body, str):
        return "INVALID_TOOL_GENERATION" in body.upper()
    return False


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


def _build_repair_prompt(question: str, error_snippet: str) -> str:
    """Build a targeted repair prompt for output-shape validation failures.

    Embeds the validation error so the model can make a specific correction
    rather than guessing what went wrong.

    Args:
        question: The original user question to re-answer.
        error_snippet: The validation error string (will be truncated to 300 chars).

    Returns:
        Augmented question with an explicit schema-repair instruction.
    """
    trimmed_error = error_snippet[:300]
    return (
        f"{question}\n\n"
        "CORRECTION REQUIRED: Your previous response failed output schema "
        f"validation with this error: {trimmed_error!r}\n\n"
        "Your final_result call MUST pass a single JSON object — "
        "not a list or array — with exactly these keys:\n"
        "  - answer  (string, required)\n"
        "  - data    (object or null, optional)\n"
        "  - warning (string or null, optional)\n\n"
        "Do NOT wrap the object in an array. "
        "Do NOT include extra keys or tool-call wrappers in the payload. "
        "Call final_result with a plain JSON object as its sole argument."
    )


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
