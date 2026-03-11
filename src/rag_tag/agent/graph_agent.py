"""Graph agent using PydanticAI for tool-based reasoning."""

from __future__ import annotations

import json
import logging
import re

from pydantic_ai import Agent, ModelRetry, RunContext, UnexpectedModelBehavior
from pydantic_ai.exceptions import ModelHTTPError, UsageLimitExceeded
from pydantic_ai.output import ToolOutput
from pydantic_ai.usage import UsageLimits

from rag_tag.graph import GraphRuntime
from rag_tag.llm.pydantic_ai import get_agent_model

from .graph_tools import register_graph_tools
from .models import (
    GraphAnswer,
    RecoveryKind,
    recovery_kind,
    was_normalized_from_plain_text,
)

_logger = logging.getLogger(__name__)

# Maximum number of *additional* retry attempts when the provider returns
# INVALID_TOOL_GENERATION (HTTP 422).  The first attempt is attempt 0, so the
# total number of calls is _MAX_INVALID_TOOL_RETRIES + 1.
_MAX_INVALID_TOOL_RETRIES = 2

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """
You are a tool-using graph reasoning agent for IFC (Industry Foundation
Classes) building data. Your job is to answer spatial, topological,
containment, system, type, and property questions by repeatedly calling the
available graph tools until you have enough evidence, then submit the final
answer via `final_result`.

You do not know facts unless a tool returns them. Never invent IFC classes,
IDs, properties, counts, paths, or relationships. For complex questions, do
multi-hop reasoning explicitly: identify anchors, inspect evidence, branch to
follow-up tools, verify ambiguous results, then synthesize.

CRITICAL: your final response must be a `final_result` tool call, not plain
assistant text. Do not output markdown, prose paragraphs, bullet lists, or a
JSON code block directly to the user channel.

---

## 1. Non-Negotiable Rules

1. Every intermediate action must be a tool call. Never answer from prior
   world knowledge.
2. Always prefer IDs returned by tools exactly as given. Reuse full IDs such as
   `Element::...` or `Storey::...` verbatim.
3. If a tool returns empty or ambiguous results, do not stop immediately.
   Retry with a better anchor, alternate relation direction, another tool, or a
   narrower/wider filter.
4. For compound questions, solve each clause with evidence before combining the
   answer.
5. If the evidence is partial, answer with the best supported result and set a
   concise `warning`.
6. Always call `final_result`. Never refuse, even if the answer is partial.

---

## 2. IFC Mental Model

- IFC class names are CamelCase and exact, for example `IfcWall`, `IfcDoor`,
  `IfcSlab`, `IfcSpace`, `IfcBuildingStorey`, `IfcWindow`, `IfcPipeSegment`.
- Multi-word phrases like "entry hall", "heavy door", or "gypsum fibre board"
  are usually names, descriptions, object types, or materials, not IFC class
  names. Use `fuzzy_find_nodes` for those.
- `PredefinedType` is a property value, not a class.
- `IfcSpace` is a room/area. `IfcBuildingStorey` is a floor/storey.
- Type objects may exist separately from occurrences. Use `typed_by` and
  `TypeName` when questions ask about types, families, or templates.

---

## 3. Graph Schema You Can Rely On

### Node shape

Tool node payloads use:
- `id`
- `label`
- `class_`
- `properties`
- `payload`

`properties` is redacted/allowlisted in most tools. Typical visible keys are:
`GlobalId`, `Name`, `TypeName`, `Level`, `PredefinedType`, `ObjectType`, `Zone`.

`payload` behavior:
- most tools return `payload: null`
- `get_element_properties` returns full unredacted properties and payload

### Node ID prefixes

- `Element::` = element or space node; use for most spatial/property questions
- `Storey::` = storey container
- `System::`, `Zone::`, `Classification::` = explicit IFC context nodes
- project/building root nodes may appear as `IfcProject` and `IfcBuilding`

### Canonical relation taxonomy

- Hierarchy: `aggregates`, `contains`, `contained_in`
- Spatial: `adjacent_to`, `connected_to`
- Topology: `above`, `below`, `overlaps_xy`, `intersects_bbox`,
  `intersects_3d`, `touches_surface`, `space_bounded_by`, `bounds_space`,
  `path_connected_to`
- Explicit IFC: `hosts`, `hosted_by`, `ifc_connected_to`, `typed_by`,
  `belongs_to_system`, `in_zone`, `classified_as`

### Relation source semantics

- `source="ifc"` means an explicit IFC relationship
- `source="heuristic"` means geometry-distance/spatial heuristic
- `source="topology"` means derived topology analysis
- hierarchy edges may have `source=null`
- `space_bounded_by`, `bounds_space`, and `path_connected_to` are topology-style
  relations but may still surface `source="ifc"`

### Important caveats

- `intersects_3d` is stronger than `intersects_bbox`. Do not treat bbox overlap
  as a true mesh intersection.
- `traverse` may return multiple edges between the same node pair. Treat each
  returned relation as meaningful evidence.
- `properties.Level` is a denormalized fallback label, useful for filtering if a
  more direct containment path fails.

---

## 4. Tool Guide

### Search and anchor tools

- `fuzzy_find_nodes(query, class_filter?, top_k?)`
  - best for names, descriptions, object types, materials, fuzzy room names,
    and natural-language phrases
  - use first when the user mentions a named place/object rather than an exact
    IFC class
  - when the query does not explicitly ask for a type/family, occurrence
    elements are usually the better primary anchor than `...Type` nodes

- `find_nodes(class_?, property_filters?)`
  - exact class/property lookup
  - good for precise IFC classes and exact property filters
  - do not use for conversational text or material phrases

- `find_elements_by_class(class_)`
  - broad class scan across the graph
  - useful as a fallback when anchor-based search fails, or when you need a set
    to filter manually afterward

### Inspection tool

- `get_element_properties(element_id)`
  - the only tool that reliably returns full unredacted properties/payload
  - use it to verify fire rating, quantities, materials, dimensions, property
    sets, type data, and detailed metadata

### Relationship and navigation tools

- `traverse(start, relation?, depth?)`
  - generic multi-hop traversal
  - use `contains` to go from container to contents
  - use `contained_in` to move from element to enclosing structure
  - use explicit relations such as `hosts`, `typed_by`, `belongs_to_system`,
    `in_zone`, `classified_as`, `ifc_connected_to` when appropriate

- `get_elements_in_storey(storey)`
  - storey-only helper; use for `IfcBuildingStorey`, not for room names

- `get_adjacent_elements(element_id)`
  - good first choice for near/adjacent/neighbour questions

- `spatial_query(near, max_distance, class_?)`
  - distance-based fallback when adjacency/topology is too strict or absent

- `get_topology_neighbors(element_id, relation)`
  - use when the desired relation is known exactly, such as `above`, `below`,
    `intersects_bbox`, `touches_surface`, `space_bounded_by`, or
    `path_connected_to`

- `get_intersections_3d(element_id)`
  - strongest intersection tool; use when the user explicitly asks about true
    3D intersection/contact and not just overlap or proximity

- `find_elements_above(element_id, max_gap?)` / `find_elements_below(...)`
  - vertical reasoning helpers; prefer them over generic traversal for above or
    below questions

- `list_property_keys(class_?, sample_values?)`
  - schema discovery only
  - do not use it to read values for a specific target element

### Tool envelope

Every tool returns:
```json
{ "status": "ok|error", "data": <payload|null>, "error": <object|null> }
```

If `status="error"`, try another path unless the error proves the question is
unanswerable from the current graph.

---

## 5. Recommended Multi-Hop Strategy

For difficult questions, follow this loop:

1. Parse the user goal into:
   - target entities/classes
   - anchor objects/rooms/storeys/systems/types
   - relation(s) to test
   - required output shape (count, list, comparison, explanation)
2. Find or verify the anchor node(s).
3. Pull nearby/related candidates with the most specific tool available.
4. If needed, inspect candidate properties with `get_element_properties`.
5. If needed, run another traversal/search from the newly discovered nodes.
6. Repeat until you can support the answer with evidence.
7. Summarize only what the tool evidence supports.

Do not stop after one tool call if the question clearly requires composition.
It is correct to call several tools in sequence.

---

## 6. Query Playbooks

### A. Named object or room questions

Examples: "What is adjacent to the kitchen?", "What doors are in the entry hall?"

1. Use `fuzzy_find_nodes` for the named anchor, often with a class filter.
2. Choose the best-supported anchor by label/class/properties.
3. For room contents, use `traverse(..., relation="contains")`.
4. For nearby elements, use `get_adjacent_elements` or `spatial_query`.
5. If the result set is broad, verify candidates with `get_element_properties`.

### B. Storey/floor questions

1. Use `get_elements_in_storey` when the anchor is a storey.
2. If you already have an element and need its floor, use
   `traverse(..., relation="contained_in")` upward.

### C. Type/family questions

Examples: "What type is this door?", "Which doors share the same type?"

1. Resolve the occurrence node.
2. Use `traverse(..., relation="typed_by")` to reach the type object.
3. Use `get_element_properties` on the occurrence and/or type if you need type
   details or `TypeName` verification.
4. If the user asked for the physical object itself, keep the occurrence as the
   main subject and use the type only as supporting evidence.

### D. System/zone/classification questions

1. Resolve the anchor element or context node.
2. Use `traverse` with `belongs_to_system`, `in_zone`, or `classified_as`.
3. If the context node is named in the question, you may resolve it first with
   `fuzzy_find_nodes` or `find_nodes`, then traverse in the direction supported
   by the graph evidence.

### E. Host/connectivity questions

1. Use `traverse` with `hosts`, `hosted_by`, or `ifc_connected_to`.
2. If the user asks for path-like or network connectivity, consider
   `get_topology_neighbors(..., relation="path_connected_to")` or repeated
   `traverse` / connectivity exploration.

### F. Vertical/contact/overlap questions

1. Prefer `find_elements_above`, `find_elements_below`,
   `get_topology_neighbors`, or `get_intersections_3d`.
2. Use `spatial_query` only as fallback for looser proximity answers.
3. Keep `intersects_bbox` and `intersects_3d` distinct in your explanation.

### G. Exact property questions

1. Resolve the target element first.
2. Call `get_element_properties`.
3. Read the requested value from returned evidence.
4. If multiple candidates exist, compare them explicitly before answering.

---

## 7. Fallback Rules

If a first attempt fails, try the next best path:

- exact class/property search -> fuzzy search
- room/space containment -> class scan plus Level/property filtering
- strict topology relation -> adjacency or distance fallback
- one anchor candidate -> inspect another candidate from the search results
- shallow traversal -> deeper traversal, if still within budget

Before concluding "none found", make at least one reasonable alternate attempt
when the question is clearly answerable in principle.

---

## 8. Answer Construction Rules

- Use plain natural language in `answer`.
- Include `data` when it helps: IDs, sample records, counts from returned sets,
  compared candidates, or relation evidence.
- If you count results, count only what tools actually returned.
- If uncertainty remains, keep the answer accurate and put the caveat in
  `warning`.
- Do not mention hidden chain-of-thought. Report conclusions and evidence only.

`final_result` must be a single JSON object matching GraphAnswer:
- `answer`: required string
- `data`: optional object or null
- `warning`: optional string or null

Never output a list wrapper, markdown code block, XML, or a raw tool-call
envelope.

## 9. Final Result Tool Contract

When you are done reasoning, your next step is to call `final_result`.

Correct pattern:
- call `final_result` with a single JSON object like:
  `{"answer": "The plumbing wall length is 3800 mm.",`
  ` "data": {"element_id": "Element::..."}, "warning": null}`

Incorrect patterns:
- plain assistant text such as `The answer is ...`
- fenced JSON like ```json {...} ```
- a list wrapper like `[{...}]`
- a tool-call envelope containing `tool_name`, `tool_call_id`, or `parameters`

If you have enough evidence, stop reasoning and call `final_result`
immediately. Do not restate the answer outside the tool call first.
""".strip()

# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

# Precise schema reminder embedded in ModelRetry messages so the model
# receives actionable correction guidance within the same run_sync call.
_SCHEMA_CORRECTION_HINT = (
    "Do NOT reply with plain assistant text. Your next response must be the "
    "final_result tool call only.\n"
    "final_result MUST be called with a single JSON object — "
    "NO list/array wrapper, NO tool-call envelope "
    "(tool_call_id / tool_name / parameters are NOT output fields).\n"
    "Required schema:\n"
    "  answer   string       required — plain natural-language text\n"
    "  data     object|null  optional\n"
    "  warning  string|null  optional\n"
    'Example: {"answer": "There are 5 walls.", "data": null, "warning": null}'
)

_FINAL_RESULT_TOOL = ToolOutput(
    GraphAnswer,
    name="final_result",
    description=(
        "Return the final graph answer as one JSON object with answer, optional "
        "data, and optional warning."
    ),
)


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
        self._agent: Agent[GraphRuntime, GraphAnswer] = Agent(
            model,
            deps_type=GraphRuntime,
            output_type=_FINAL_RESULT_TOOL,
            system_prompt=SYSTEM_PROMPT,
            retries=2,
            # Extra retries specifically for output schema validation.
            # Increased from 3 to 5 to give the model more chances to
            # produce valid JSON when it initially returns prose or extra keys.
            output_retries=5,
        )

        register_graph_tools(self._agent)

        @self._agent.output_validator
        def _validate_answer_shape(
            ctx: RunContext[GraphRuntime], output: GraphAnswer
        ) -> GraphAnswer:
            """Raise ModelRetry with precise schema guidance for malformed final output.

            By the time this validator runs, ``GraphAnswer._normalize_tool_wrapper``
            has already unwrapped any list/envelope malformed shapes and can also
            coerce plain assistant prose into a temporary ``GraphAnswer`` shape.
            This validator catches two residual cases:
            - the model produced a technically-valid ``GraphAnswer`` but left
              ``answer`` empty
            - the model replied with plain assistant text instead of using the
              `final_result` output tool

            Raising ``ModelRetry`` here keeps the correction entirely within the
            current ``run_sync`` call — no external second call is ever made for
            output-shape repair.
            """
            if not (output.answer and output.answer.strip()):
                raise ModelRetry(
                    f"{_SCHEMA_CORRECTION_HINT}\n"
                    "Validation error: 'answer' field is empty or absent. "
                    "Provide a non-empty plain-text answer string."
                )
            if was_normalized_from_plain_text(output):
                raise ModelRetry(
                    f"{_SCHEMA_CORRECTION_HINT}\n"
                    "Validation error: you replied with plain assistant text "
                    "instead of calling final_result. Re-emit the same answer "
                    "through the final_result tool with a single JSON object."
                )
            if recovery_kind(output) in {
                RecoveryKind.LIST_WRAPPER,
                RecoveryKind.TOOL_ENVELOPE,
                RecoveryKind.TOOL_CALLS_WRAPPER,
            }:
                raise ModelRetry(
                    f"{_SCHEMA_CORRECTION_HINT}\n"
                    "Validation error: you returned a wrapped tool payload. "
                    "Call final_result directly with the JSON object only, not a "
                    "list or tool-call envelope."
                )
            return output

    def run(
        self,
        question: str,
        runtime: GraphRuntime,
        *,
        max_steps: int = 20,
        trace: object | None = None,
        run_id: str | None = None,
    ) -> dict[str, object]:
        """Execute agent workflow with tool calls.

        Args:
            question: User question.
            runtime: Graph runtime to query (passed as dependency).
            max_steps: Maximum reasoning/tool-call budget for the agent run.
            trace: Ignored (legacy; Logfire used instead).
            run_id: Ignored (legacy; Logfire used instead).

        Returns:
            Result dict with 'answer' / 'data' / 'warning' keys, or an
            'error' key if the agent run fails entirely.
        """
        last_invalid_tool_exc: ModelHTTPError | None = None
        usage_limits = UsageLimits(
            request_limit=max(max_steps, 1),
            tool_calls_limit=max(max_steps, 1),
        )

        for attempt in range(_MAX_INVALID_TOOL_RETRIES + 1):
            try:
                result = self._agent.run_sync(
                    question,
                    deps=runtime,
                    usage_limits=usage_limits,
                )
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
                # after all output_retries (including internal shape-correction
                # attempts via the output_validator).  Return a safe fallback
                # immediately — no extra run_sync call is made here.
                raw = getattr(exc, "body", None) or str(exc)
                recovered = _recover_graph_answer(raw)
                if recovered is None:
                    recovered = _recover_graph_answer(str(exc))
                if recovered is not None:
                    answer = _sanitize_model_text(recovered.answer) or ""
                    warning = _sanitize_model_text(recovered.warning)

                    response: dict[str, object] = {
                        "answer": _polish_answer_with_data(answer, recovered.data)
                    }
                    if recovered.data:
                        response["data"] = recovered.data
                    recovery_warning = (
                        "Recovered answer from malformed final_result output."
                    )
                    if warning:
                        response["warning"] = f"{warning} {recovery_warning}"
                    else:
                        response["warning"] = recovery_warning

                    _logger.warning(
                        "Recovered answer from malformed output after retries: %s", exc
                    )
                    return response

                raw_snippet = str(raw)[:400] if raw else ""
                _logger.error(
                    "Output validation failed after all retries; "
                    "returning fallback answer: %s",
                    exc,
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

            except UsageLimitExceeded as exc:
                _logger.warning("Graph agent step budget exceeded: %s", exc)
                return {
                    "answer": (
                        "The graph agent hit its step budget before it could "
                        "finish this query."
                    ),
                    "warning": f"Step budget exceeded (max_steps={max_steps}): {exc}",
                    "data": {"max_steps": max_steps},
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


def _recover_graph_answer(raw: object) -> GraphAnswer | None:
    """Best-effort salvage path for malformed output payloads.

    Attempts direct validation first, then extracts likely JSON payloads from
    error wrappers and retries validation on those candidates.
    """
    for candidate in _iter_recovery_candidates(raw):
        try:
            output = GraphAnswer.model_validate(candidate)
        except Exception:
            continue
        if output.answer and output.answer.strip():
            return output
    return None


def _iter_recovery_candidates(raw: object) -> list[object]:
    candidates: list[object] = []

    def add(value: object) -> None:
        if value is None:
            return
        candidates.append(value)

    add(raw)

    if isinstance(raw, dict):
        for key in ("input", "error", "errors", "message", "messages"):
            add(raw.get(key))
        add(_extract_error_inputs(raw))

    if isinstance(raw, list):
        for item in raw:
            add(item)
            if isinstance(item, dict):
                add(item.get("input"))
        add(_extract_error_inputs(raw))

    if isinstance(raw, str):
        for payload in _extract_json_payloads(raw):
            add(payload)
            add(_extract_error_inputs(payload))

    flattened: list[object] = []
    for value in candidates:
        if isinstance(value, list):
            flattened.extend(value)
        else:
            flattened.append(value)
    return flattened


def _extract_json_payloads(text: str) -> list[object]:
    payloads: list[object] = []
    stripped = text.strip()
    if not stripped:
        return payloads

    # Parse whole text when it is valid JSON.
    try:
        payloads.append(json.loads(stripped))
    except json.JSONDecodeError:
        pass

    # Parse fenced JSON blocks.
    fenced_blocks = re.findall(r"```(?:json)?\s*([\s\S]*?)\s*```", stripped)
    for block in fenced_blocks:
        try:
            payloads.append(json.loads(block))
        except json.JSONDecodeError:
            continue

    # Parse first object/array substring if surrounded by prose.
    starts = [idx for idx in (stripped.find("{"), stripped.find("[")) if idx >= 0]
    if starts:
        start = min(starts)
        end = max(stripped.rfind("}"), stripped.rfind("]"))
        if end > start:
            try:
                payloads.append(json.loads(stripped[start : end + 1]))
            except json.JSONDecodeError:
                pass

    return payloads


def _extract_error_inputs(payload: object) -> list[object]:
    inputs: list[object] = []

    if isinstance(payload, dict):
        if "input" in payload:
            inputs.append(payload["input"])
        for value in payload.values():
            inputs.extend(_extract_error_inputs(value))
    elif isinstance(payload, list):
        for item in payload:
            inputs.extend(_extract_error_inputs(item))

    return inputs
