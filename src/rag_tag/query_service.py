"""Shared query execution service for CLI and TUI."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import networkx as nx

from rag_tag.agent import GraphAgent
from rag_tag.ifc_sql_tool import SqlQueryError, query_ifc_sql
from rag_tag.paths import find_project_root
from rag_tag.router import RouteDecision, route_question


def find_sqlite_dbs() -> list[Path]:
    # return all .db files sorted by name so we query every loaded model,
    # not just whichever one was modified most recently
    project_root = find_project_root(Path(__file__).resolve().parent)
    if project_root is None:
        return []
    candidates: list[Path] = []
    for folder_name in ("output", "db"):
        folder = project_root / folder_name
        if not folder.exists():
            continue
        candidates.extend(folder.glob("*.db"))
    candidates.sort(key=lambda p: p.name)
    return candidates


def load_graph(
    dataset: str | None = None,
    payload_mode: str | None = None,
) -> nx.DiGraph:
    """Build the NetworkX graph from JSONL output files.

    Args:
        dataset: JSONL stem to load (e.g. ``"Building-Architecture"``).
            When ``None``, all ``.jsonl`` files in ``output/`` are used.
        payload_mode: Graph payload mode (``"full"`` or ``"minimal"``).
            When ``None``, the ``GRAPH_PAYLOAD_MODE`` env var is used,
            defaulting to ``"full"`` if unset.
    """
    from rag_tag.parser.jsonl_to_graph import build_graph  # noqa: PLC0415

    return build_graph(dataset=dataset, payload_mode=payload_mode)


def _resolve_context_db_path(
    db_paths: list[Path],
    graph_dataset: str | None,
) -> Path | None:
    """Resolve the primary DB path to wire into the graph context.

    Used so that ``get_element_properties`` can do DB-backed property lookups
    against the correct database rather than relying solely on in-memory graph
    data.

    Priority:
    1. Single DB path → use directly.
    2. Multiple DBs + *graph_dataset* → first DB whose stem matches the dataset.
    3. Multiple DBs + no dataset → ``None`` (ambiguous; skip DB lookup).
    4. No DBs → ``None``.
    """
    if not db_paths:
        return None
    if len(db_paths) == 1:
        return db_paths[0]
    if graph_dataset:
        for p in db_paths:
            if p.stem == graph_dataset:
                return p
    # Multiple databases with no clear selection — do not guess.
    return None


def execute_sql_query(
    decision: RouteDecision,
    db_paths: list[Path],
) -> dict[str, Any]:
    if decision.sql_request is None:
        return _sql_error(decision, "Router did not produce a SQL request.")
    if not db_paths:
        return _sql_error(
            decision, "No SQLite database found. Run rag-tag-jsonl-to-sql."
        )

    req = decision.sql_request
    # Canonical effective limit for this request; used as the global cap across DBs.
    effective_limit = req.limit or 50

    # Query every database and merge counts/items across all models.
    combined_count = 0
    combined_total = 0
    combined_items: list[Any] = []
    last_payload: dict[str, Any] = {}

    for db_path in db_paths:
        try:
            envelope = query_ifc_sql(db_path, decision.sql_request)
        except SqlQueryError:
            continue
        if envelope["status"] != "ok":
            continue
        payload = envelope["data"]
        last_payload = payload
        combined_count += payload.get("count", 0)
        combined_total += payload.get("total_count", 0)
        combined_items.extend(payload.get("items") or [])

    if not last_payload:
        return _sql_error(decision, "All database queries failed.")

    # Enforce global list limit: merged items across all DBs must not exceed the
    # requested limit.  Each per-DB query already applies the same LIMIT clause,
    # so the only way to exceed it is when results come from multiple databases.
    if req.intent == "list":
        combined_items = combined_items[:effective_limit]

    # Rebuild summary string with the combined count.
    label = req.ifc_class or "elements"
    if req.intent == "count":
        if req.level_like:
            summary = (
                f"Found {combined_count} {label} matching level '{req.level_like}'."
            )
        else:
            summary = f"Found {combined_count} {label}."
        result_count = combined_count
    else:
        # Use actual item count post-cap so summary matches displayed rows.
        shown = len(combined_items)
        if req.level_like:
            summary = (
                f"Found {combined_total} {label} matching level"
                f" '{req.level_like}', showing {shown}."
            )
        else:
            summary = f"Found {combined_total} {label}, showing {shown}."
        result_count = shown

    return {
        "route": "sql",
        "decision": decision.reason,
        "db_paths": [str(p) for p in db_paths],
        "answer": summary,
        "data": {
            "intent": last_payload.get("intent"),
            "filters": last_payload.get("filters"),
            "count": result_count,
            "total_count": combined_total,
            # Return the canonical effective limit, not a per-DB artifact.
            "limit": effective_limit,
            "items": combined_items,
        },
        "sql": last_payload.get("sql"),
    }


def _sql_error(decision: RouteDecision, message: str) -> dict[str, Any]:
    """Build a SQL error result payload."""
    return {
        "route": "sql",
        "decision": decision.reason,
        "error": message,
    }


def execute_graph_query(
    question: str,
    graph: nx.DiGraph,
    agent: GraphAgent,
    decision: RouteDecision,
) -> dict[str, Any]:
    """Execute graph query via agent.

    Args:
        question: User question
        graph: NetworkX graph
        agent: Graph agent instance
        decision: Routing decision

    Returns:
        Result dict with answer, data, or error
    """
    agent_result = agent.run(question, graph, max_steps=6)
    return {
        "route": "graph",
        "decision": decision.reason,
        **agent_result,
    }


def execute_query(
    question: str,
    db_paths: list[Path],
    graph: nx.DiGraph | None,
    agent: GraphAgent | None,
    *,
    decision: RouteDecision | None = None,
    debug_llm_io: bool = False,
    graph_dataset: str | None = None,
    context_db: Path | None = None,
    payload_mode: str | None = None,
) -> dict[str, Any]:
    """Execute a query through the full pipeline (routing + execution).

    Args:
        question: User question
        db_paths: All SQLite databases to query
        graph: NetworkX graph (or None, will be loaded if needed)
        agent: Graph agent (or None, will be created if needed)
        decision: Optional precomputed routing decision
        debug_llm_io: Enable debug printing
        graph_dataset: JSONL stem to load (e.g. "Building-Architecture").
            When None, all .jsonl files in output/ are used.
        context_db: Explicit DB path to wire into the graph context for
            ``get_element_properties`` lookups.  When None, it is inferred
            from *db_paths* and *graph_dataset* via
            ``_resolve_context_db_path``.  Callers that already know the
            selected DB (e.g. when ``--db`` was passed explicitly) should
            supply it here for clarity and correctness.
        payload_mode: Optional graph payload mode override (``"full"`` or
            ``"minimal"``).  When None, graph construction uses the
            ``GRAPH_PAYLOAD_MODE`` env var defaulting to ``"full"``.

    Returns:
        Result dict with answer, route, decision, data, or error.
        Also returns updated graph and agent if they were loaded/created.
    """
    try:
        if decision is None:
            decision = route_question(question, debug_llm_io=debug_llm_io)

        if decision.route == "sql":
            result = execute_sql_query(decision, db_paths)
            return {"result": result, "graph": graph, "agent": agent}

        # Resolve the DB path for graph context if not provided by the caller.
        resolved_context_db = context_db or _resolve_context_db_path(
            db_paths, graph_dataset
        )

        # Graph route
        # Keep backward compatibility with existing helper call sites and
        # lightweight monkeypatched checks: only pass db_path when we actually
        # resolved one.
        if resolved_context_db is None and payload_mode is None:
            graph, agent = _ensure_graph_context(
                graph,
                agent,
                debug_llm_io,
                graph_dataset,
            )
        elif resolved_context_db is None:
            graph, agent = _ensure_graph_context(
                graph,
                agent,
                debug_llm_io,
                graph_dataset,
                payload_mode=payload_mode,
            )
        elif payload_mode is None:
            graph, agent = _ensure_graph_context(
                graph,
                agent,
                debug_llm_io,
                graph_dataset,
                resolved_context_db,
            )
        else:
            graph, agent = _ensure_graph_context(
                graph,
                agent,
                debug_llm_io,
                graph_dataset,
                resolved_context_db,
                payload_mode=payload_mode,
            )
        result = execute_graph_query(question, graph, agent, decision)
        return {"result": result, "graph": graph, "agent": agent}

    except Exception as exc:
        error_result = _routing_error(decision, str(exc))
        return {"result": error_result, "graph": graph, "agent": agent}


def _normalize_db_path(raw_path: Path | str | None) -> str | None:
    if raw_path is None:
        return None
    return str(Path(raw_path).expanduser().resolve())


def _clear_graph_db_caches(graph: nx.DiGraph) -> None:
    """Clear graph-scoped DB caches after context DB changes."""
    graph.graph.pop("_property_cache", None)
    graph.graph.pop("_property_key_cache", None)

    cached_conn = graph.graph.pop("_db_lookup_conn", None)
    if cached_conn is not None:
        try:
            cached_conn.close()
        except Exception:  # noqa: BLE001
            pass


def _ensure_graph_context(
    graph: nx.DiGraph | None,
    agent: GraphAgent | None,
    debug_llm_io: bool,
    graph_dataset: str | None = None,
    db_path: Path | None = None,
    payload_mode: str | None = None,
) -> tuple[nx.DiGraph, GraphAgent]:
    """Load graph and agent instances when missing; wire DB path into graph context.

    Args:
        graph: Existing graph or None to trigger loading.
        agent: Existing agent or None to trigger creation.
        debug_llm_io: Passed through to GraphAgent constructor.
        graph_dataset: JSONL stem for ``build_graph`` (None = all datasets).
        db_path: DB path to store on ``graph.graph["_db_path"]`` so that
            ``get_element_properties`` can perform DB-backed lookups.
            When None, any previously wired context is preserved.
        payload_mode: Optional graph payload mode override for graph loading.
    """
    if graph is None:
        graph = load_graph(graph_dataset, payload_mode=payload_mode)
    if db_path is not None:
        # Wire the active DB path into the graph for tool-level property lookup.
        # When the DB context changes on an existing graph instance, clear
        # graph-scoped property caches to avoid stale cross-DB reads.
        resolved_db_path = db_path.expanduser().resolve()
        previous_db_path = _normalize_db_path(graph.graph.get("_db_path"))
        current_db_path = str(resolved_db_path)
        if previous_db_path != current_db_path:
            _clear_graph_db_caches(graph)
        graph.graph["_db_path"] = resolved_db_path
    if agent is None:
        agent = GraphAgent(debug_llm_io=debug_llm_io)
    return graph, agent


def _routing_error(decision: RouteDecision | None, message: str) -> dict[str, Any]:
    """Build a routing error result payload."""
    error_result = {"error": message, "route": "?", "decision": "routing failed"}
    if decision is not None:
        error_result["route"] = decision.route
        error_result["decision"] = decision.reason
    return error_result
