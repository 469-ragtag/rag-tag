"""Graph agent using PydanticAI for tool-based reasoning."""

from __future__ import annotations

import json
import logging
import os
import re
from collections.abc import Sequence
from pathlib import Path

from pydantic_ai import (
    Agent,
    ModelMessage,
    ModelRetry,
    RunContext,
    UnexpectedModelBehavior,
    capture_run_messages,
)
from pydantic_ai.exceptions import ModelHTTPError, UsageLimitExceeded
from pydantic_ai.output import ToolOutput
from pydantic_ai.usage import UsageLimits

from rag_tag.config import get_default_graph_output_retries
from rag_tag.graph import GraphRuntime
from rag_tag.llm.pydantic_ai import get_agent_model, get_agent_model_settings
from rag_tag.usage import (
    normalize_usage_metrics,
    sum_usage_metrics,
    usage_metrics_from_messages,
)

from .graph_tools import register_graph_tools
from .models import (
    GraphAnswer,
    RecoveryKind,
    recovery_kind,
    was_normalized_from_plain_text,
)

_logger = logging.getLogger(__name__)
_MODULE_DIR = Path(__file__).resolve().parent
_BENCHMARK_GRAPH_PROMPT_APPEND_ENV_VAR = "RAG_TAG_BENCHMARK_GRAPH_PROMPT_APPEND"

# Maximum number of *additional* retry attempts when the provider returns
# INVALID_TOOL_GENERATION (HTTP 422).  The first attempt is attempt 0, so the
# total number of calls is _MAX_INVALID_TOOL_RETRIES + 1.
_MAX_INVALID_TOOL_RETRIES = 2
_OUTPUT_REPAIR_REQUEST_LIMIT = 2
_OUTPUT_REPAIR_TOOL_CALL_LIMIT = 1


def _is_pydantic_test_model(model: object) -> bool:
    """Return True when *model* is PydanticAI's lightweight test model."""
    model_type = model if isinstance(model, type) else type(model)
    return getattr(model_type, "__name__", "") == "TestModel" and getattr(
        model_type, "__module__", ""
    ).startswith("pydantic_ai.models.test")


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
assistant text. Do not output prose, Markdown, or a JSON code block directly
to the user channel outside the `final_result` tool call. Lightweight Markdown
is allowed inside the `answer` field of `final_result`.

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
  - use this as a fallback when no more specific macro tool fits
  - use `contains` to go from container to contents
  - use `contained_in` to move from element to enclosing structure
  - use explicit relations such as `hosts`, `typed_by`, `belongs_to_system`,
    `in_zone`, `classified_as`, `ifc_connected_to` when appropriate

- `trace_distribution_network(start, max_depth?, relations?, max_results?)`
  - preferred macro tool for bounded network/system tracing
  - use this instead of repeated `traverse(..., relation="ifc_connected_to")`
    when the user wants a connected branch, network summary, or downstream/upstream
    connectivity set

- `find_shortest_path(start, end, max_path_length?, relations?)`
  - preferred tool when the user explicitly asks for the path/connection between
    two anchors
  - use this instead of manually chaining `traverse` calls hop by hop

- `find_by_classification(classification, max_results?)`
  - preferred tool for classification/code/reference label questions
  - use this instead of raw `traverse(..., relation="classified_as")` unless you
    already have the exact classification node and only need one tiny follow-up

- `find_equipment_serving_space(space, max_depth?, max_results?)`
  - preferred macro tool for "what serves this room/space" questions
  - use this before composing room boundaries, terminals, systems, and network
    traversal by hand

- `get_elements_in_storey(storey)`
  - storey-only helper; use for `IfcBuildingStorey`, not for room names

- `find_container_elements_excluding(container_id, exclude_container_ids?, depth?)`
  - best for set-difference questions such as "elements in the building but not
    on the ground floor"
  - prefer this over unconstrained `traverse(..., depth=5)` once you know the
    main container and the container(s) to exclude
  - returns non-container members from `contains` / `aggregates`, and for
    `IfcZone` / `IfcSpatialZone` also incoming `in_zone` members, minus
    excluded container members

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

### Graph-to-SQL bridge tools

- `aggregate_elements(element_ids, metric, field?)`
  - use after graph discovery when the user asks for an exact count, sum,
    average, minimum, or maximum over a returned element set
  - pass the exact tool-returned IDs/GlobalIds; do not count or sum mentally in
    the prompt
  - use `metric="count"` with no field for exact set size; use `field` for core
    columns, dotted property keys, or dotted quantity keys

- `group_elements_by_property(element_ids, property_key, max_groups?)`
  - use after graph discovery when the user asks for a breakdown by level,
    type, property, quantity, or other DB-backed field
  - do not bucket/group results manually in context when this tool fits

### Macro-first defaults

When one of these fits, use it before generic `traverse` or manual multi-step
composition:

1. `trace_distribution_network` for connected branch/network tracing; do not
   emulate it with repeated `traverse(..., relation="ifc_connected_to")`.
2. `find_shortest_path` for a path or connection between two anchors; do not
   chain hop-by-hop traversals.
3. `find_by_classification` for classification/reference/code lookups; do not
   start with raw `classified_as` traversal.
4. `find_equipment_serving_space` for "what serves this room/space" questions;
   do not manually compose room -> terminal -> system -> equipment unless the
   macro tool fails.
5. `aggregate_elements` / `group_elements_by_property` for exact counts, math,
   or grouped breakdowns over discovered graph sets; do not count, sum,
   average, min/max, or group in-context.

### Tool envelope

Every tool returns:
```json
{ "status": "ok|error", "data": <payload|null>, "error": <object|null> }
```

Many tool payloads also include `data.evidence`: a compact grounding list with
`global_id` when available, an internal `id` fallback, plus `label`, `class_`,
and sometimes `relation`, `source_tool`, or `match_reason`.

If `status="error"`, try another path unless the error proves the question is
unanswerable from the current graph.

---

## 5. Recommended Multi-Hop Strategy

For difficult questions, follow this loop:

Prefer macro/helper tools first; only drop to generic `traverse` when no more
specific tool fits or the macro tool returns weak evidence.

1. Parse the user goal into:
   - target entities/classes
   - anchor objects/rooms/storeys/systems/types
   - relation(s) to test
   - required output shape (count, list, comparison, explanation)
2. Find or verify the anchor node(s).
3. Pull nearby/related candidates with the most specific tool available.
4. If needed, inspect candidate properties with `get_element_properties`.
5. If the question asks for exact set math or breakdowns over discovered
   elements, call `aggregate_elements` or `group_elements_by_property`.
6. If needed, run another traversal/search from the newly discovered nodes.
7. Repeat until you can support the answer with evidence.
8. Summarize only what the tool evidence supports.

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
2. Use `find_by_classification` when the question is driven by a
   classification label/reference.
3. Otherwise use `traverse` with `belongs_to_system`, `in_zone`, or
   `classified_as`.
4. If the context node is named in the question, you may resolve it first with
   `fuzzy_find_nodes` or `find_nodes`, then traverse in the direction supported
   by the graph evidence.

### E. Host/connectivity questions

1. Use `trace_distribution_network` for bounded network tracing from one anchor.
2. Use `find_shortest_path` when the user asks for the path between two anchors.
3. Use `traverse` with `hosts`, `hosted_by`, or `ifc_connected_to` only for
   small targeted follow-up inspection.
4. If the user asks for path-like topology specifically, consider
   `get_topology_neighbors(..., relation="path_connected_to")`.

### F. Space served-by questions

Examples: "What equipment serves Room 101?", "Which unit supplies the kitchen?"

1. Resolve the space anchor, usually with `fuzzy_find_nodes` if the room is named.
2. Use `find_equipment_serving_space`.
3. Only fall back to manual composition if the macro tool returns weak/empty
   evidence and you need to inspect one specific candidate.

### G. Vertical/contact/overlap questions

1. Prefer `find_elements_above`, `find_elements_below`,
   `get_topology_neighbors`, or `get_intersections_3d`.
2. Use `spatial_query` only as fallback for looser proximity answers.
3. Keep `intersects_bbox` and `intersects_3d` distinct in your explanation.

### H. Exact property questions

1. Resolve the target element first.
2. Call `get_element_properties`.
3. Read the requested value from returned evidence.
4. If multiple candidates exist, compare them explicitly before answering.

### I. Negative location / exclusion questions

Examples: "What is in the building but not on the ground floor?", "Which
elements belong to this zone but not this room?"

1. Resolve the main container and the container(s) to exclude.
2. Use `find_container_elements_excluding`.
3. Do not fall back to broad unconstrained traversal unless the helper fails.
4. Once the helper returns the needed set, stop and answer; do not keep
   exploring unrelated edges.

### J. Aggregation / grouping over discovered graph sets

Examples: "How many of these are fire-rated?", "Sum the net volume of the
walls around this room.", "Group the found doors by level."

1. First discover the exact element set with graph/search tools.
2. Reuse the exact returned IDs or GlobalIds.
3. Call `aggregate_elements` for count/sum/avg/min/max.
4. Call `group_elements_by_property` for deterministic grouped breakdowns.
5. Do not do set math, counting, summing, averaging, or grouping in the prompt
   when one of these bridge tools applies.

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

Avoid unconstrained `traverse(..., depth>3)` unless you still lack the basic
container anchors. Broad traversal is a last resort because it wastes tool
budget and floods the context window.

---

## 8. Answer Construction Rules

- Use lightweight Markdown in `answer` when it improves readability.
- Good formats: short paragraphs, `##`/`###` headings, bullet lists, and valid
  Markdown tables.
- Do not use ASCII-art tables, inline pipe-delimited rows, or malformed
  pseudo-Markdown.
- If you present multiple entities, prefer short Markdown sections over one long
  paragraph.
- Ground claims with tool-returned IDs. Prefer `data.evidence[].global_id` when
  present, otherwise use `data.evidence[].id` or other exact tool-returned IDs.
- Include `data` when it helps: `evidence`, IDs, sample records, counts from
  returned sets, compared candidates, or relation evidence.
- If you count, sum, average, min/max, or group results, use the dedicated tool
  outputs rather than doing the math in-context.
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


def build_system_prompt() -> str:
    """Return the graph-agent system prompt plus any benchmark-only appendix."""

    return append_benchmark_prompt(SYSTEM_PROMPT)


def get_benchmark_graph_prompt_append() -> str | None:
    """Return the active benchmark-only graph prompt appendix, if any."""

    appendix = os.getenv(_BENCHMARK_GRAPH_PROMPT_APPEND_ENV_VAR)
    if appendix is None:
        return None
    cleaned = appendix.strip()
    return cleaned or None


def append_benchmark_prompt(
    prompt: str,
    *,
    prompt_append: str | None = None,
) -> str:
    """Append the active benchmark-only prompt appendix when configured."""

    appendix = prompt_append
    if appendix is None:
        appendix = get_benchmark_graph_prompt_append()
    if appendix is None:
        return prompt
    return f"{prompt}\n\n---\n\n{appendix}"


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

# Precise schema reminder embedded in ModelRetry messages so the model
# receives actionable correction guidance within the same run_sync call.
_SCHEMA_CORRECTION_HINT = (
    "Your previous response was invalid. Your next response must be a real "
    "final_result tool call only.\n"
    "Do NOT print plain text, Markdown, fenced JSON, a list/array wrapper, or "
    "a tool-call envelope as text.\n"
    "Call final_result with one JSON object only. Do NOT include tool_call_id, "
    "tool_name, or parameters.\n"
    "Required schema:\n"
    "  answer   string       required — lightweight Markdown allowed\n"
    "  data     object|null  optional\n"
    "  warning  string|null  optional\n"
    'Example: {"answer": "There are 5 walls.", "data": null, "warning": null}'
)

_OUTPUT_TOOL_REPAIR_PROMPT = (
    "Repair the previous failed output formatting only. Use the tool evidence "
    "already present in the message history and do not call any more graph tools.\n"
    f"{_SCHEMA_CORRECTION_HINT}"
)

_OUTPUT_JSON_REPAIR_PROMPT = (
    "Repair the previous failed output formatting only. Use the tool evidence "
    "already present in the message history and do not call any more graph tools.\n"
    "Return exactly one JSON object with keys `answer`, `data`, and `warning`.\n"
    "Do NOT emit a tool-call wrapper, `tool_call_id`, `tool_name`, or `parameters`.\n"
    "Do NOT emit Markdown fences or extra prose outside the JSON object."
)

_OUTPUT_TOOL_CALL_FAILURE_MARKERS = (
    "please include your response in a tool call",
    "tool call",
    "output tool",
    "final_result",
)

_FINAL_RESULT_TOOL = ToolOutput(
    GraphAnswer,
    name="final_result",
    description=(
        "Return the final graph answer as one JSON object with answer, optional "
        "grounded data/evidence, and optional warning."
    ),
)


class GraphAgent:
    """Graph agent using PydanticAI with tool calling."""

    def __init__(
        self,
        *,
        debug_llm_io: bool = False,
        output_retries: int | None = None,
    ) -> None:
        """Initialise graph agent with PydanticAI.

        Args:
            debug_llm_io: Enable debug printing (unused for PydanticAI;
                          kept for API compatibility).
            output_retries: Structured-output validation retries. When None,
                uses the checked-in config default or the built-in fallback.
        """
        self._debug_llm_io = debug_llm_io
        resolved_output_retries = output_retries
        if resolved_output_retries is None:
            resolved_output_retries = get_default_graph_output_retries(_MODULE_DIR)
        self._output_retries = resolved_output_retries

        model = get_agent_model()
        try:
            model_settings = get_agent_model_settings()
        except RuntimeError:
            # Unit tests monkeypatch get_agent_model to TestModel and should not
            # require provider-specific runtime config just to build the agent.
            if _is_pydantic_test_model(model):
                model_settings = None
            else:
                raise
        self._agent: Agent[GraphRuntime, GraphAnswer] = Agent(
            model,
            deps_type=GraphRuntime,
            output_type=_FINAL_RESULT_TOOL,
            system_prompt=build_system_prompt(),
            model_settings=model_settings,
            retries=2,
            output_retries=resolved_output_retries,
        )
        self._repair_agent: Agent[GraphRuntime, GraphAnswer] = Agent(
            model,
            deps_type=GraphRuntime,
            output_type=GraphAnswer,
            system_prompt=(
                "You are a formatting-repair assistant. Repair only the final "
                "output shape from the prior conversation. Do not call tools."
            ),
            model_settings=model_settings,
            retries=1,
            output_retries=resolved_output_retries,
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
        repair_attempted = False
        usage_limits = UsageLimits(
            request_limit=max(max_steps, 1),
            tool_calls_limit=max(max_steps, 1),
        )

        for attempt in range(_MAX_INVALID_TOOL_RETRIES + 1):
            captured_messages: Sequence[ModelMessage] | None = None
            try:
                with capture_run_messages() as captured_messages:
                    result = self._agent.run_sync(
                        question,
                        deps=runtime,
                        usage_limits=usage_limits,
                    )
                output = result.output
                return _attach_usage(
                    _graph_answer_to_response(output),
                    normalize_usage_metrics(result),
                )

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
                if (
                    not repair_attempted
                    and captured_messages
                    and _should_attempt_output_tool_repair(exc)
                ):
                    repair_attempted = True
                    repaired_response = self._attempt_output_tool_repair(
                        runtime=runtime,
                        message_history=captured_messages,
                    )
                    if repaired_response is not None:
                        return _attach_usage(
                            repaired_response,
                            sum_usage_metrics(
                                usage_metrics_from_messages(captured_messages),
                                repaired_response.get("usage"),
                            ),
                        )

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
                    return _attach_usage(
                        response,
                        usage_metrics_from_messages(captured_messages),
                    )

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

    def _attempt_output_tool_repair(
        self,
        *,
        runtime: GraphRuntime,
        message_history: Sequence[ModelMessage],
    ) -> dict[str, object] | None:
        """Run one targeted repair pass for missing real output-tool calls."""
        try:
            repair_result = self._repair_agent.run_sync(
                _OUTPUT_JSON_REPAIR_PROMPT,
                deps=runtime,
                usage_limits=UsageLimits(
                    request_limit=max(
                        self._output_retries + 1, _OUTPUT_REPAIR_REQUEST_LIMIT
                    ),
                    tool_calls_limit=_OUTPUT_REPAIR_TOOL_CALL_LIMIT,
                ),
                message_history=message_history,
            )
        except UnexpectedModelBehavior as exc:
            _logger.warning("Output-tool repair pass failed: %s", exc)
            return None
        except Exception as exc:
            _logger.warning("Output-tool repair pass raised unexpected error: %s", exc)
            return None

        _logger.warning("Recovered answer via targeted output-tool repair pass.")
        return _attach_usage(
            _graph_answer_to_response(repair_result.output),
            normalize_usage_metrics(repair_result),
        )


def _graph_answer_to_response(output: GraphAnswer) -> dict[str, object]:
    """Convert validated graph output into the public response payload."""
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


def _attach_usage(
    response: dict[str, object],
    usage: object,
) -> dict[str, object]:
    normalized = normalize_usage_metrics(usage)
    if not normalized.usage_available:
        return response
    enriched = dict(response)
    enriched["usage"] = normalized.as_dict()
    return enriched


def _should_attempt_output_tool_repair(exc: UnexpectedModelBehavior) -> bool:
    """Return True when the failure looks like a missing real output tool call."""
    for candidate in _iter_output_tool_repair_candidates(exc):
        text = _flatten_output_error_text(candidate).lower()
        if not text:
            continue
        if any(marker in text for marker in _OUTPUT_TOOL_CALL_FAILURE_MARKERS):
            return True
    return False


def _iter_output_tool_repair_candidates(
    exc: UnexpectedModelBehavior,
) -> list[object]:
    """Collect nested exception payloads that may mention tool-call failures."""
    candidates: list[object] = [getattr(exc, "body", None), str(exc)]

    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        candidates.append(str(current))
        candidates.append(getattr(current, "body", None))
        next_exc = current.__cause__ or current.__context__
        current = next_exc if isinstance(next_exc, BaseException) else None

    return candidates


def _flatten_output_error_text(payload: object) -> str:
    """Collapse nested output-validation payloads into searchable text."""
    if payload is None:
        return ""
    if isinstance(payload, str):
        return payload
    if isinstance(payload, dict):
        parts = [
            _flatten_output_error_text(value)
            for key, value in payload.items()
            if key in {"message", "error", "errors", "body", "input"}
        ]
        return " ".join(part for part in parts if part)
    if isinstance(payload, list):
        return " ".join(_flatten_output_error_text(item) for item in payload)
    return str(payload)


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
    """Strip provider annotation tags while preserving meaningful line breaks."""
    if value is None:
        return None
    text = str(value)
    text = re.sub(r"</?co:[^>]+>", "", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[^\S\n]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = "\n".join(line.rstrip() for line in text.split("\n")).strip()
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
