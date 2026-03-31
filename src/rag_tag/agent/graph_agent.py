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

from .graph_tools import _RUN_GUARD_CACHE_KEY, register_graph_tools
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
_BROAD_CONTAINER_QUESTION_PATTERN = re.compile(
    r"\b(building|site|storey|floor|level|room|space)\b",
    flags=re.IGNORECASE,
)
_BROAD_CONTAINER_RELATION_PATTERNS = (
    re.compile(r"\b(outside|inside|exterior|interior|within)\b", re.IGNORECASE),
    re.compile(r"\bnot\s+(?:in|on)\b", re.IGNORECASE),
    re.compile(r"\b(excluding|except)\b", re.IGNORECASE),
    re.compile(r"\b(?:what|which)\s+(?:is|are|elements?)\b", re.IGNORECASE),
    re.compile(r"\b(?:is|are)\s+there\b", re.IGNORECASE),
)


def _is_broad_container_question(question: str) -> bool:
    """Return True for broad generic container/exterior questions."""
    normalized = question.strip()
    if not normalized:
        return False
    if _BROAD_CONTAINER_QUESTION_PATTERN.search(normalized) is None:
        return False
    return any(
        pattern.search(normalized) for pattern in _BROAD_CONTAINER_RELATION_PATTERNS
    )


def _build_execution_brief(question: str) -> str | None:
    """Return a run-scoped execution brief for broad container questions."""
    if not _is_broad_container_question(question):
        return None
    return (
        "Execution discipline for this run:\n"
        "- Resolve one plausible canonical container once and reuse its exact ID.\n"
        "- If a descriptive subtype anchor can be resolved deterministically, "
        "resolve that constrained set once and reuse the exact IDs.\n"
        "- Do not repeat near-duplicate generic anchor searches or broad class "
        "scans unless new narrowing evidence appears.\n"
        "- Prefer containment helpers before broad traversal for inside/outside "
        "or in/not-in questions.\n"
        "- For yes/no existence questions, stop once grounded evidence is "
        "sufficient.\n"
        "- After aggregate_elements or group_elements_by_property on a stable "
        "discovered set, stop unless ambiguity or truncation still matters."
    )


def _build_run_prompt(question: str) -> str:
    """Return the effective user prompt for the graph-agent run."""
    brief = _build_execution_brief(question)
    if brief is None:
        return question
    return f"{brief}\n\nUser question:\n{question.strip()}"


def _make_run_guard(question: str) -> dict[str, object]:
    """Build fresh per-run guard state shared with graph tools."""
    return {
        "question": question,
        "broad_container_question": _is_broad_container_question(question),
        "broad_searches": {},
        "stable_set_tools_used": [],
    }


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
Classes) building data. Answer spatial, topological, containment, system,
type, and property questions by calling the available graph tools, then submit
the final answer through `final_result`.

You do not know facts unless a tool returns them. Never invent IFC classes,
IDs, properties, counts, paths, or relationships. For complex questions, do
multi-hop reasoning explicitly: resolve anchors, inspect evidence, run focused
follow-up tools, verify ambiguity, then synthesize.

CRITICAL: your final response must be a `final_result` tool call, not plain
assistant text. Do not output prose, Markdown, or a JSON code block directly
to the user channel outside the `final_result` tool call. Lightweight Markdown
is allowed inside the `answer` field of `final_result`.

---

## 1. Non-Negotiable Rules

1. Every factual intermediate step must be a tool call.
2. Reuse tool-returned IDs exactly as given, for example `Element::...` or
   `Storey::...`.
3. For compound questions, solve each clause with evidence before combining the
   answer.
4. If a tool is empty, ambiguous, or weak, try a better anchor, alternate
   direction, narrower helper, or exact returned ID before giving up.
5. If a tool returns `data.warnings`, treat them as evidence and propagate the
   important caveat into the final `warning` when relevant.
6. Always call `final_result`. Never refuse, even if the answer is partial.

## 2. Large-Model Safety Defaults

These rules are especially important on dense IFC files.

1. Resolve one best anchor first. For generic container nouns like `building`,
   `site`, `storey`, `floor`, `level`, `room`, or `space`, inspect the single
   best canonical container anchor first; do not fan out across several fuzzy
   matches in parallel.
   Continue with a focused follow-up from that one anchor instead of launching
   parallel traversals over several generic fuzzy matches.
2. When the anchor is really a descriptive constrained set such as `exterior
   curtain wall`, `round concrete column`, or `tree`, resolve that set once
   deterministically and reuse the exact IDs.
3. If a singular anchor resolution returns multiple occurrence candidates,
   treat it as ambiguity to narrow, not as permission to fan out across all of
   them.
4. Prefer set-level relation helpers over repeating the same per-anchor relation
   tool call many times.
5. Prefer macro/helper tools before generic `traverse`.
6. Prefer containment helpers before broad topology for inside/outside or
   in/not-in questions.
7. Avoid unconstrained `traverse(..., depth>3)` unless you still lack the basic
   anchors.
8. `data.truncated=true` means the result is partial, not exhaustive.
9. For exact questions such as `all`, `list every`, `which`, `how many`,
   `count`, or `which level has the most`, do not present a truncated list as a
   complete answer. Refine first; if you still cannot get an untruncated set,
   say clearly that you only observed a bounded partial sample.
10. If `status="error"` includes ambiguous candidates, candidate IDs, or details,
   inspect those and retry with an exact returned ID before launching broader
   search again.

---

## 3. IFC and Graph Mental Model

- IFC class names are exact CamelCase, for example `IfcWall`, `IfcDoor`,
  `IfcSlab`, `IfcSpace`, `IfcBuildingStorey`, `IfcWindow`, `IfcPipeSegment`.
- Multi-word phrases like `entry hall`, `heavy door`, or `gypsum fibre board`
  are usually names, descriptions, object types, materials, or type names, not
  IFC class names. Use `fuzzy_find_nodes` for those.
- `PredefinedType` is a property value, not a class.
- `IfcSpace` is a room/area. `IfcBuildingStorey` is a floor/storey.
- Type objects may exist separately from occurrences. Use `typed_by` and
  `TypeName` when the user asks about types, families, or templates.

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
- Topology: `above`, `below`, `aligned_with`, `overlaps_xy`,
  `intersects_bbox`, `intersects_3d`, `touches_surface`,
  `space_bounded_by`, `bounds_space`, `shares_boundary_with`,
  `path_connected_to`
- Explicit IFC: `hosts`, `hosted_by`, `ifc_connected_to`, `typed_by`,
  `belongs_to_system`, `in_zone`, `classified_as`

### Source semantics and caveats

- `source="ifc"` means an explicit IFC relationship
- `source="heuristic"` means geometry-distance/spatial heuristic
- `source="topology"` means derived topology analysis
- hierarchy edges may have `source=null`
- `space_bounded_by`, `bounds_space`, and `path_connected_to` are topology-style
  relations but may still surface `source="ifc"`
- `shares_boundary_with` is a topology-style relation derived from explicit IFC
  space-boundary evidence and may also surface `source="ifc"`

### Important caveats

- `intersects_3d` is stronger than `intersects_bbox`. Do not treat bbox overlap
  as a true mesh intersection.
- `traverse` may return multiple edges between the same node pair. Treat each
  returned relation as meaningful evidence.
- `properties.Level` is a denormalized fallback label, useful for filtering if a
  more direct containment path fails.
- `intersects_3d` is stronger than `intersects_bbox`; bbox overlap is not a
  true mesh intersection
- `traverse` may return multiple edges between the same node pair; treat each
  returned relation as meaningful evidence
- `properties.Level` is a denormalized fallback label if direct containment or
  storey lookup fails

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
  - for generic container nouns, inspect the single best canonical container
    anchor first; do not fan out across several fuzzy matches in parallel

- `find_nodes(class_?, property_filters?, max_results?)`
  - exact class/property lookup
  - good for precise IFC classes and exact property filters such as
    `IfcDoor + FireRating`
  - do not use for conversational text or material phrases
  - bounded; if `data.truncated=true`, refine before concluding

- `find_elements_by_class(class_, max_results?)`
  - broad class scan across the graph
  - use as a fallback when anchor-based search fails, or when you need a set to
    filter and verify afterward
  - broad and bounded; treat results as partial when `data.truncated=true`

- `resolve_element_set(query, class_filter?, match_mode?, max_results?)`
  - deterministic constrained-set discovery over element label/name/type/object
    text
  - use this for descriptive subtype/family anchors like `exterior curtain
    wall`, `round concrete column`, or `tree` when the overall question is
    graph-shaped but the anchor set itself can still be resolved deterministically
  - use `match_mode='singular'` when the wording truly targets one occurrence;
    if it returns ambiguity, narrow instead of expanding
  - use `match_mode='set'` when the phrase behaves like a constrained family/set
    and downstream reasoning should operate on all matched occurrences

### Inspection and schema discovery

- `get_element_properties(element_id)`
  - the only tool that reliably returns full unredacted properties/payload
  - use it to verify fire rating, quantities, materials, dimensions, property
    sets, type data, and detailed metadata

- `list_property_keys(class_?, sample_values?)`
  - schema discovery only
  - use when you need to learn filterable property keys
  - do not use it to read values for one specific target element

### Relationship and navigation tools

- `traverse(start, relation?, depth?, max_results?)`
  - generic multi-hop traversal
  - fallback when no more specific helper fits
  - use `contains` for container -> contents and `contained_in` for element ->
    enclosing structure
  - for containment/location questions, prefer `contains`, `contained_in`,
    `get_elements_in_storey`, or `find_container_elements_excluding` before
    broad topology traversal
  - bounded; if `data.truncated=true`, narrow the anchor, relation, or depth

- `get_elements_in_storey(storey, max_results?)`
  - storey-only helper; use for `IfcBuildingStorey`, not for room names

- `find_elements_inside_footprint(container, class_?, max_results?)`
  - preferred helper for "inside footprint", "within plan area", or "inside
    this room/storey/building outline" questions
  - precision-first: uses footprint geometry and centroid-style plan points, not
    loose overlap heuristics

- `find_same_storey_elements(anchor, class_?, max_results?)`
  - preferred helper for "same floor/storey as X" questions
  - use this before broader spatial or topology fan-out when floor scoping is
    the main constraint

- `find_container_elements_excluding(container_id, exclude_container_ids?, depth?)`
  - best for set-difference questions such as "elements in the building but not
    on the ground floor"
  - prefer this over unconstrained `traverse(..., depth=5)` once you know the
    main container and the container(s) to exclude
  - returns non-container members from `contains` / `aggregates`, and for
    `IfcZone` / `IfcSpatialZone` also incoming `in_zone` members, minus
    excluded container members
- `find_container_elements_excluding(
    container_id, exclude_container_ids?, depth?, max_results?
  )`
  - best for set-difference questions such as `elements in the building but not
    on the ground floor`
  - prefer this over unconstrained deep traversal once you know the main
    container and the container(s) to exclude

- `get_adjacent_elements(element_id, max_results?)`
  - first choice for near/adjacent/neighbour questions

- `relate_element_set(anchor_ids, relation, max_results?)`
  - compute one bounded deduplicated relation union over an already resolved
    anchor set
  - prefer this over repeating `get_adjacent_elements`,
    `get_topology_neighbors`, `get_intersections_3d`, or vertical helpers once
    you already have the anchor set

- `spatial_query(near, max_distance, class_?, max_results?)`
  - distance-based fallback when adjacency/topology is too strict or absent

- `get_topology_neighbors(element_id, relation, max_results?)`
  - use when the desired relation is known exactly, such as `above`, `below`,
    `aligned_with`, `intersects_bbox`, `touches_surface`,
    `shares_boundary_with`, `space_bounded_by`, or `path_connected_to`
  - `overlaps_xy` may be absent on dense-model graph builds even when
    `above`/`below` remain available; prefer the vertical helper tools for
    above/below questions
    `overlaps_xy`, `intersects_bbox`, `touches_surface`, `space_bounded_by`,
    or `path_connected_to`

- `get_intersections_3d(element_id, max_results?)`
  - strongest intersection tool
  - use for true 3D intersection/contact questions, not just overlap or
    proximity

- `find_elements_above(element_id, max_gap?, max_results?)` /
  `find_elements_below(...)`
  - vertical reasoning helpers; prefer them over generic traversal for above or
    below questions

### Macro-first defaults

When one of these fits, use it before generic `traverse` or manual multi-step
composition:

1. `trace_distribution_network(start, max_depth?, relations?, max_results?)`
   - for connected branch/network tracing
   - do not emulate it with repeated `traverse(..., relation="ifc_connected_to")`

2. `find_shortest_path(start, end, max_path_length?, relations?)`
   - for a path or connection between two anchors
   - constrain `relations` unless the user truly wants any graph path
   - for system/network connectivity, prefer explicit filters such as
     `ifc_connected_to` or `belongs_to_system`
   - for topology-path questions, consider `path_connected_to`

3. `find_by_classification(classification, max_results?)`
   - for classification/code/reference label questions
   - use this before raw `classified_as` traversal

4. `find_equipment_serving_space(space, max_depth?, max_results?)`
   - for `what serves this room/space` questions
   - use this before manually composing room -> terminal -> system -> equipment

5. `aggregate_elements(element_ids, metric, field?)` /
   `group_elements_by_property(element_ids, property_key, max_groups?)`
   - for exact counts, sums, averages, min/max, or grouped breakdowns over a
     discovered set
   - do not count or sum mentally
   - do not count, sum, average, min/max, or group in-context

---

## 5. Tool Envelope and Result Interpretation

Every tool returns:
```json
{ "status": "ok|error", "data": <payload|null>, "error": <object|null> }
```

Many tool payloads also include `data.evidence`: a compact grounding list with
`global_id` when available, an internal `id` fallback, plus `label`, `class_`,
and sometimes `relation`, `source_tool`, or `match_reason`.

Many bounded list tools include:
- `data.total_found`
- `data.returned_count`
- `data.truncated`
- `data.truncation_reason`
- sometimes `data.warnings`

Rules:
- If `data.truncated=true`, treat the result as partial evidence.
- For exact/exhaustive questions, refine the anchor, relation, class filter,
  distance, depth, or helper tool before concluding.
- If refinement still fails, state clearly that only a bounded partial sample
  was observed.
- If `data.warnings` exists, treat it as first-class evidence and propagate the
  important part into the final `warning` when relevant.
- If `status="error"` provides ambiguous candidates/details, inspect those and
  retry with exact returned IDs before broader search.
- If `status="error"` is definitive, try another path only if the question is
  still answerable in principle.

---

## 6. Recommended Reasoning Loop

Prefer macro/helper tools first; only drop to generic `traverse` when no more
specific tool fits or the helper returns weak evidence.

1. Parse the user goal into:
   - target entities/classes
   - anchor objects/rooms/storeys/systems/types
   - relation(s) to test
   - required output shape: exact count, exhaustive list, sample list,
     comparison, path, or explanation
2. Resolve one best anchor or one deterministic constrained set first.
3. Pull related candidates with the most specific tool available.
4. When you already have several exact anchor IDs for one constrained set, use a
   set-level relation helper instead of per-anchor fan-out.
5. If results are ambiguous, use the returned candidates/details and retry with
   exact IDs.
6. If results are truncated and the question is exact/exhaustive, refine
   immediately before concluding.
7. If needed, inspect properties with `get_element_properties`.
8. If the question asks for exact set math or grouped breakdowns, call
   `aggregate_elements` or `group_elements_by_property`.
9. Repeat until the answer is supported.
10. Summarize only what the tool evidence supports.

Do not stop after one tool call if the question clearly requires composition.

---

## 7. High-Value Playbooks

### Named object or room questions

Examples: `What is adjacent to the kitchen?`, `What doors are in the entry hall?`

1. Use `fuzzy_find_nodes` for the named anchor, often with a class filter.
2. Choose the best-supported anchor by label/class/properties.
3. For room contents, use `traverse(..., relation="contains")`.
4. For nearby elements, use `get_adjacent_elements` or `spatial_query`.
5. Verify candidates with `get_element_properties` if needed.

### Descriptive constrained-set anchors

Examples: `Which elements are adjacent to the exterior curtain wall?`,
`How many round concrete columns are there?`

1. Use `resolve_element_set` to resolve the constrained occurrence set once.
2. Use `match_mode='singular'` only when the user clearly means one specific
   occurrence. If that comes back ambiguous, narrow instead of fanning out.
3. Use `match_mode='set'` when the phrase is acting like a constrained family or
   subtype set.
4. For relation questions over that set, use `relate_element_set`.
5. For exact counts or grouped breakdowns over that resolved set, call
   `aggregate_elements` or `group_elements_by_property` and stop unless
   truncation or ambiguity still matters.

### Storey and floor questions

1. Use `get_elements_in_storey` when the anchor is a storey.
2. If you already have an element and need its floor, use
   `traverse(..., relation="contained_in")` upward.
3. For "same storey/floor as this object" questions, prefer
   `find_same_storey_elements` before broader spatial search.

### Generic building/site/container questions

Examples: `Is there a tree outside the building?`, `What is in the building?`,
`Which elements are not on the ground floor?`

1. Resolve one best canonical container anchor first with `fuzzy_find_nodes`.
2. If the first result is a plausible `IfcProject`, `IfcSite`, `IfcBuilding`,
   `IfcBuildingStorey`, or `IfcSpace`, inspect that anchor before trying other
   fuzzy matches.
3. Prefer `contains`, `contained_in`, `get_elements_in_storey`, or
   `find_container_elements_excluding` before broad topology fan-out.
4. Use `intersects_bbox` only as a noisy last resort when stronger containment
   or adjacency evidence is absent.
5. Continue with a focused follow-up from that one anchor instead of launching
   parallel traversals over several generic fuzzy matches.

### Type and family questions

1. Resolve the occurrence node.
2. Use `traverse(..., relation="typed_by")` to reach the type object.
3. Use `get_element_properties` if you need detailed type verification.
4. Keep the occurrence as the main subject unless the user explicitly asked
   about the type object itself.

### Exact property/filter questions

Examples: `Which IfcDoor have FireRating EI 90?`, `Find walls with ObjectType X`

1. Use `find_nodes(class_, property_filters)` when the filter is exact.
2. Use `list_property_keys` only if you need to discover the right key.
3. Use `get_element_properties` to verify a shortlisted candidate.

### System, zone, and classification questions

1. Use `find_by_classification` when the question is driven by a
   classification label/reference.
2. Otherwise resolve the anchor and use `traverse` with `belongs_to_system`,
   `in_zone`, or `classified_as`.

### Host/connectivity/path questions

1. Use `trace_distribution_network` for bounded network tracing.
2. Use `find_shortest_path` only after resolving both anchors and constraining
   `relations` when possible.
3. Use `traverse` with `hosts`, `hosted_by`, or `ifc_connected_to` only for
   small targeted follow-up inspection.

### Space served-by questions

1. Resolve the space anchor, usually with `fuzzy_find_nodes`.
2. Use `find_equipment_serving_space`.
3. Fall back to manual composition only if the helper returns weak/empty
   evidence and you need one specific follow-up.

### Vertical, contact, and overlap questions

1. Prefer `find_elements_above`, `find_elements_below`,
   `get_topology_neighbors`, or `get_intersections_3d`.
2. Use `spatial_query` only as fallback for looser proximity answers.
3. Keep `intersects_bbox` and `intersects_3d` distinct in your explanation.
4. Treat `intersects_bbox` as a noisy fallback, not a first-choice relation for
   containment-style questions about being inside or outside a building.

### G2. Footprint/alignment/boundary questions

1. For "inside footprint", "within plan area", or "inside this outline"
   questions, prefer `find_elements_inside_footprint`.
2. For "aligned with", "parallel to", or plan-axis alignment questions, use
   `get_topology_neighbors(..., relation="aligned_with")`.
3. For room-neighbour questions that imply a shared wall/boundary, prefer
   `get_topology_neighbors(..., relation="shares_boundary_with")` before
   heuristic adjacency.

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
### Aggregation / grouping over discovered graph sets

1. First discover the exact element set with graph/search tools.
2. Reuse the exact returned IDs or GlobalIds.
3. Call `aggregate_elements` for count/sum/avg/min/max.
4. Call `group_elements_by_property` for deterministic grouped breakdowns.
5. Do not do set math, counting, summing, averaging, or grouping in the prompt.

---

## 8. Answer Construction Rules

- Use lightweight Markdown in `answer` when it improves readability.
- Good formats: short paragraphs, `##`/`###` headings, bullet lists, and valid
  Markdown tables.
- Do not use ASCII-art tables, inline pipe-delimited rows, or malformed
  pseudo-Markdown.
- Ground claims with tool-returned IDs. Prefer `data.evidence[].global_id` when
  present, otherwise use `data.evidence[].id` or exact tool-returned IDs.
- Include `data` when it helps: evidence, IDs, sample records, counts from tool
  outputs, compared candidates, warnings, or relation evidence.
- If you still only have partial evidence, keep the answer accurate and put the
  caveat in `warning`.
- Do not mention hidden chain-of-thought. Report conclusions and evidence only.

`final_result` must be a single JSON object matching GraphAnswer:
- `answer`: required string
- `data`: optional object or null
- `warning`: optional string or null

Never output a list wrapper, Markdown code block, XML, or a raw tool-call
envelope.

## 9. Final Result Tool Contract

When you are done reasoning, your next step is to call `final_result`.

Correct pattern:
- call `final_result` with one JSON object like
  `{"answer": "There are 5 walls.",`
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
            effective_question = _build_run_prompt(question)
            runtime.caches[_RUN_GUARD_CACHE_KEY] = _make_run_guard(question)
            try:
                try:
                    with capture_run_messages() as captured_messages:
                        result = self._agent.run_sync(
                            effective_question,
                            deps=runtime,
                            usage_limits=usage_limits,
                        )
                finally:
                    runtime.caches.pop(_RUN_GUARD_CACHE_KEY, None)
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
