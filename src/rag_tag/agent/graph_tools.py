"""PydanticAI tools for graph queries.

Each tool wraps the existing graph query interface from ifc_graph_tool.py,
preserving the envelope structure (status/data/error) for compatibility.

Fuzzy normalisation is handled here (in the tool layer) so that ifc_graph_tool.py
remains untouched. The key improvements are:

- fuzzy_find_nodes: score-ranked text search across Name/ObjectType/Description.
- find_nodes: normalises class_ via rapidfuzz before querying; treats multi-word
  inputs as name searches; falls back to fuzzy_find_nodes when exact query is empty.
- list_property_keys: discovers available property keys to aid filter selection.
- traverse: docstring clarifies contains/contained_in semantics for location queries.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Any

import networkx as nx
from pydantic_ai import RunContext
from rapidfuzz import fuzz, process

from rag_tag.ifc_graph_tool import query_ifc_graph, sanitize_properties_for_llm
from rag_tag.sql_element_lookup import decode_db_value

# Minimum rapidfuzz WRatio score (0-100) to accept a fuzzy class normalisation.
_CLASS_FUZZY_THRESHOLD = 72
_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module-level helpers (not registered as tools)
# ---------------------------------------------------------------------------


def _all_class_values(G: nx.DiGraph) -> list[str]:
    """Return sorted unique IFC class_ values present in the graph."""
    return sorted({str(d["class_"]) for _, d in G.nodes(data=True) if d.get("class_")})


def _normalize_class_fuzzy(class_input: str, G: nx.DiGraph) -> tuple[str | None, float]:
    """Match a user-supplied class name against actual class_ values in the graph.

    Returns:
        (best_match, score) where best_match is None if no match exceeds the
        threshold.
    """
    known = _all_class_values(G)
    if not known:
        return None, 0.0
    result = process.extractOne(class_input, known, scorer=fuzz.WRatio)
    if result is None:
        return None, 0.0
    best, score, _ = result
    return (
        (best, float(score))
        if score >= _CLASS_FUZZY_THRESHOLD
        else (None, float(score))
    )


def _fuzzy_find_nodes_impl(
    G: nx.DiGraph,
    query: str,
    class_filter: str | None = None,
    top_k: int = 10,
    min_score: float = 50.0,
) -> dict[str, Any]:
    """Score nodes by fuzzy-matching query against label, ObjectType, Description.

    Returns a standard envelope dict (status/data/error).
    """
    results: list[dict[str, Any]] = []
    for node_id, data in G.nodes(data=True):
        if class_filter is not None:
            if str(data.get("class_", "")).lower() != class_filter.lower():
                continue

        props: dict[str, Any] = data.get("properties", {}) or {}
        payload: dict[str, Any] = data.get("payload") or {}
        candidates = [
            str(data.get("label", "")),
            str(props.get("ObjectType", "")),
            str(props.get("Description", "")),
            str(payload.get("Name", "")),
            str(payload.get("IfcType", "")),
            str(payload.get("ClassRaw", "")),
        ]

        # Include individual material names so queries like
        # "made of gypsum fiber-board" can surface the right element.
        for mat_name in props.get("Materials") or payload.get("Materials") or []:
            if mat_name:
                candidates.append(str(mat_name))

        best_score = max(
            (fuzz.WRatio(query, c) for c in candidates if c and c != "None"),
            default=0.0,
        )

        if best_score >= min_score:
            results.append(
                {
                    "id": node_id,
                    "label": data.get("label"),
                    "class_": data.get("class_"),
                    "score": round(best_score, 1),
                    "properties": sanitize_properties_for_llm(props),
                }
            )

    results.sort(key=lambda x: x["score"], reverse=True)
    return {
        "status": "ok",
        "data": {
            "query": query,
            "class_filter": class_filter,
            "matches": results[:top_k],
            "total": len(results),
        },
        "error": None,
    }


def _decode_typed_db_value(value: Any) -> Any:
    """Decode DB value using the shared sql_element_lookup decoder."""
    return decode_db_value(value)


def _collect_dotted_keys_from_sqlite(
    db_path: Path,
    class_filter: str | None,
) -> dict[str, list[Any]]:
    """Collect dotted PropertySet/Quantity keys from SQLite.

    Returns a mapping of ``dotted_key -> up to 3 sample values``.
    Any DB/schema/runtime error is handled gracefully by returning an empty map.
    """
    if not db_path.exists():
        return {}

    where_sql = ""
    params: tuple[Any, ...] = ()
    if class_filter:
        where_sql = " WHERE LOWER(e.ifc_class) = LOWER(?)"
        params = (class_filter,)

    key_samples: dict[str, list[Any]] = {}

    def _record_sample(key: str, value: Any) -> None:
        if key not in key_samples:
            key_samples[key] = []
        if len(key_samples[key]) < 3:
            key_samples[key].append(value)

    try:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        try:
            prop_rows = conn.execute(
                "SELECT p.pset_name, p.property_name, p.value "
                "FROM properties p "
                "JOIN elements e ON e.express_id = p.element_id"
                f"{where_sql}",
                params,
            ).fetchall()
            for row in prop_rows:
                pset_name = str(row["pset_name"])
                prop_name = str(row["property_name"])
                _record_sample(
                    f"{pset_name}.{prop_name}",
                    _decode_typed_db_value(row["value"]),
                )

            qty_rows = conn.execute(
                "SELECT q.qto_name, q.quantity_name, q.value "
                "FROM quantities q "
                "JOIN elements e ON e.express_id = q.element_id"
                f"{where_sql}",
                params,
            ).fetchall()
            for row in qty_rows:
                qto_name = str(row["qto_name"])
                qty_name = str(row["quantity_name"])
                _record_sample(
                    f"{qto_name}.{qty_name}",
                    _decode_typed_db_value(row["value"]),
                )
        finally:
            conn.close()
    except Exception as exc:  # noqa: BLE001
        _logger.debug("list_property_keys DB fallback failed (%s): %s", db_path, exc)
        return {}

    return key_samples


# ---------------------------------------------------------------------------
# Tool registration
# ---------------------------------------------------------------------------


def register_graph_tools(agent: Any) -> None:
    """Register all graph query tools on the given PydanticAI agent.

    Args:
        agent: PydanticAI Agent instance to register tools on.
    """

    @agent.tool
    def fuzzy_find_nodes(
        ctx: RunContext[nx.DiGraph],
        query: str,
        class_filter: str | None = None,
        top_k: int = 10,
    ) -> dict[str, Any]:
        """Fuzzy-search for graph nodes by matching query against common text fields."""
        return _fuzzy_find_nodes_impl(
            ctx.deps, query, class_filter=class_filter, top_k=top_k
        )

    @agent.tool
    def find_nodes(
        ctx: RunContext[nx.DiGraph],
        class_: str | None = None,
        property_filters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Find nodes in the IFC graph by class and/or property filters.

        Multi-word class_ input is treated as a descriptive query and routed to
        fuzzy_find_nodes. class_ is fuzzy-normalised against classes present in
        the graph before querying.

        ``property_filters`` supports two key formats:
        - Flat key (e.g. ``{"Name": "Wall A"}``): matched against the node's
          direct ``properties`` dict first, then searched across all psets.
        - Dotted key (e.g. ``{"Pset_WallCommon.FireRating": "EI 90"}``): targets
          a specific named PropertySet.  Use ``list_property_keys`` to discover
          valid keys.

        A missing key never matches an expected value of ``None``; filters only
        pass when the key is explicitly present with the expected value.
        """
        G: nx.DiGraph = ctx.deps

        # Multi-word input means descriptive name, not IFC class.
        if class_ and " " in class_.strip():
            return _fuzzy_find_nodes_impl(G, class_, top_k=20)

        normalized_class = class_
        if class_:
            best, _score = _normalize_class_fuzzy(class_, G)
            if best is not None:
                normalized_class = best

        params: dict[str, Any] = {}
        if normalized_class:
            params["class"] = normalized_class
        if property_filters:
            params["property_filters"] = property_filters

        result = query_ifc_graph(G, "find_nodes", params)

        # Exact match empty -> fuzzy fallback over names/descriptions.
        if (
            result.get("status") == "ok"
            and len((result.get("data") or {}).get("elements", [])) == 0
            and class_
        ):
            if property_filters:
                return {
                    "status": "error",
                    "error": (
                        f"Exact match for properties {property_filters} failed. "
                        "The value might be formatted differently in the raw "
                        "data. Try using 'fuzzy_find_nodes' instead."
                    ),
                }

        return result

    @agent.tool
    def traverse(
        ctx: RunContext[nx.DiGraph],
        start: str,
        relation: str | None = None,
        depth: int = 1,
    ) -> dict[str, Any]:
        """Traverse the graph from a starting node following edges.

        For location/storey lookup:
        - relation='contains' to move from storey/space -> contained elements
        - relation='contained_in' to move from element -> containing structure
        """
        params: dict[str, Any] = {"start": start, "depth": depth}
        if relation:
            params["relation"] = relation
        return query_ifc_graph(ctx.deps, "traverse", params)

    @agent.tool
    def spatial_query(
        ctx: RunContext[nx.DiGraph],
        near: str,
        max_distance: float,
        class_: str | None = None,
    ) -> dict[str, Any]:
        """Find elements within a spatial distance of a reference element."""
        params: dict[str, Any] = {"near": near, "max_distance": max_distance}
        if class_:
            params["class"] = class_
        return query_ifc_graph(ctx.deps, "spatial_query", params)

    @agent.tool
    def get_elements_in_storey(
        ctx: RunContext[nx.DiGraph],
        storey: str,
    ) -> dict[str, Any]:
        """Get all non-container elements in a specific storey/level."""
        return query_ifc_graph(ctx.deps, "get_elements_in_storey", {"storey": storey})

    @agent.tool
    def find_elements_by_class(
        ctx: RunContext[nx.DiGraph],
        class_: str,
    ) -> dict[str, Any]:
        """Find all elements of a specific IFC class."""
        return query_ifc_graph(ctx.deps, "find_elements_by_class", {"class": class_})

    @agent.tool
    def get_adjacent_elements(
        ctx: RunContext[nx.DiGraph],
        element_id: str,
    ) -> dict[str, Any]:
        """Get elements spatially adjacent to a given element."""
        return query_ifc_graph(
            ctx.deps, "get_adjacent_elements", {"element_id": element_id}
        )

    @agent.tool
    def get_topology_neighbors(
        ctx: RunContext[nx.DiGraph],
        element_id: str,
        relation: str,
    ) -> dict[str, Any]:
        """Get topology neighbors for one relation.

        relation must be one of: above, below, overlaps_xy, intersects_bbox,
        intersects_3d, touches_surface.
        """
        return query_ifc_graph(
            ctx.deps,
            "get_topology_neighbors",
            {"element_id": element_id, "relation": relation},
        )

    @agent.tool
    def get_intersections_3d(
        ctx: RunContext[nx.DiGraph],
        element_id: str,
    ) -> dict[str, Any]:
        """Get mesh-informed 3D intersection neighbors for an element."""
        return query_ifc_graph(
            ctx.deps, "get_intersections_3d", {"element_id": element_id}
        )

    @agent.tool
    def find_elements_above(
        ctx: RunContext[nx.DiGraph],
        element_id: str,
        max_gap: float | None = None,
    ) -> dict[str, Any]:
        """Find elements above a reference element."""
        params: dict[str, Any] = {"element_id": element_id}
        if max_gap is not None:
            params["max_gap"] = max_gap
        return query_ifc_graph(ctx.deps, "find_elements_above", params)

    @agent.tool
    def find_elements_below(
        ctx: RunContext[nx.DiGraph],
        element_id: str,
        max_gap: float | None = None,
    ) -> dict[str, Any]:
        """Find elements below a reference element."""
        params: dict[str, Any] = {"element_id": element_id}
        if max_gap is not None:
            params["max_gap"] = max_gap
        return query_ifc_graph(ctx.deps, "find_elements_below", params)

    @agent.tool
    def list_property_keys(
        ctx: RunContext[nx.DiGraph],
        class_: str | None = None,
        sample_values: bool = False,
    ) -> dict[str, Any]:
        """List property keys available in the graph, optionally scoped to one class.

        Returns two kinds of keys:
        - Flat keys (e.g. ``GlobalId``, ``Name``) sourced from the node's
          ``properties`` dict.
        - Dotted keys (e.g. ``Pset_WallCommon.FireRating``) sourced from nested
          ``PropertySets.Official`` / ``PropertySets.Custom`` blocks in the node
          payload.

        Both key formats are accepted by ``find_nodes`` ``property_filters``.
        Use this tool first to discover valid filter keys before calling
        ``find_nodes`` with property constraints.

        Note: When the graph was built in ``minimal`` payload mode
        (``GRAPH_PAYLOAD_MODE=minimal``), PropertySet and Quantity blocks are
        not stored in-memory.  When a DB path is wired into the graph context,
        this tool falls back to SQLite to recover dotted keys
        (e.g. ``Pset_WallCommon.*``).  Flat keys derived from ``properties``
        (e.g. ``GlobalId``, ``Name``) are always available regardless of
        payload mode.
        """
        G: nx.DiGraph = ctx.deps
        key_samples: dict[str, list[Any]] = {}

        def _record_key(key: str, value: Any) -> None:
            if key not in key_samples:
                key_samples[key] = []
            if sample_values and len(key_samples[key]) < 3:
                key_samples[key].append(value)

        def _collect_pset_leaf_keys(
            pset_name: str,
            node: dict[str, Any],
            path_prefix: str = "",
        ) -> None:
            for raw_key, raw_value in node.items():
                key_part = str(raw_key)
                path = f"{path_prefix}.{key_part}" if path_prefix else key_part
                if isinstance(raw_value, dict):
                    _collect_pset_leaf_keys(pset_name, raw_value, path)
                else:
                    _record_key(f"{pset_name}.{path}", raw_value)

        for _, data in G.nodes(data=True):
            if class_ is not None:
                if str(data.get("class_", "")).lower() != class_.lower():
                    continue
            props = data.get("properties") or {}
            if not isinstance(props, dict):
                props = {}
            for key, value in props.items():
                _record_key(str(key), value)

            # Also enumerate dotted keys from nested PropertySets in payload.
            payload = data.get("payload") or {}
            if not isinstance(payload, dict):
                continue

            pset_block = payload.get("PropertySets") or {}
            if not isinstance(pset_block, dict):
                continue

            for section in ("Official", "Custom"):
                section_block = pset_block.get(section) or {}
                if not isinstance(section_block, dict):
                    continue
                for pset_name, pset_props in section_block.items():
                    if not isinstance(pset_props, dict):
                        continue
                    _collect_pset_leaf_keys(str(pset_name), pset_props)

            # Also enumerate dotted keys from the Quantities block in payload.
            # Quantities (e.g. Qto_WallBaseQuantities) are stored at
            # payload["Quantities"], not inside PropertySets.
            quantities_block = payload.get("Quantities") or {}
            if not isinstance(quantities_block, dict):
                continue
            for qto_name, qto_data in quantities_block.items():
                if not isinstance(qto_data, dict):
                    continue
                _collect_pset_leaf_keys(str(qto_name), qto_data)

        # Minimal payload mode drops nested pset/quantity blocks.  If a DB path
        # is available in graph context, augment dotted keys from SQLite.
        payload_mode = str(G.graph.get("_payload_mode", "full")).lower()
        db_path_raw = G.graph.get("_db_path")
        if payload_mode == "minimal" and db_path_raw is not None:
            db_path = Path(db_path_raw)
            cache = G.graph.setdefault("_property_key_cache", {})
            cache_key = (str(db_path.resolve()), class_ or "")
            db_key_samples = cache.get(cache_key)
            if db_key_samples is None:
                db_key_samples = _collect_dotted_keys_from_sqlite(db_path, class_)
                cache[cache_key] = db_key_samples

            for key, samples in db_key_samples.items():
                if sample_values:
                    existing = key_samples.setdefault(key, [])
                    for sample in samples:
                        if len(existing) >= 3:
                            break
                        existing.append(sample)
                else:
                    _record_key(key, None)

        result: dict[str, Any] = {
            "status": "ok",
            "data": {"keys": sorted(key_samples.keys()), "class_filter": class_},
            "error": None,
        }
        if sample_values:
            result["data"]["samples"] = key_samples
        return result

    @agent.tool
    def get_element_properties(
        ctx: RunContext[nx.DiGraph],
        element_id: str,
    ) -> dict[str, Any]:
        """Fetch ALL properties for a specific element (DB-backed when available).

        Looks up *element_id* in the SQLite database wired into the graph
        context (when available), then merges with the in-memory graph
        payload.  Falls back to in-memory data only when no DB is configured.
        Returns the full, unredacted property envelope including PropertySets,
        Quantities, and flat properties.
        """
        return query_ifc_graph(
            ctx.deps, "get_element_properties", {"element_id": element_id}
        )
