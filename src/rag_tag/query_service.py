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


def load_graph() -> nx.DiGraph:
    from rag_tag.parser.jsonl_to_graph import build_graph
    return build_graph()


def execute_sql_query(
    decision: RouteDecision,
    db_paths: list[Path],
) -> dict[str, Any]:
    if decision.sql_request is None:
        return _sql_error(decision, "Router did not produce a SQL request.")
    if not db_paths:
        return _sql_error(
            decision, "No SQLite database found. Run parser/csv_to_sql.py."
        )

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

    # Rebuild summary string with the combined count.
    req = decision.sql_request
    label = req.ifc_class or "elements"
    if req.intent == "count":
        if req.level_like:
            summary = (
                f"Found {combined_count} {label}"
                f" matching level '{req.level_like}'."
            )
        else:
            summary = f"Found {combined_count} {label}."
    else:
        limit = req.limit or 50
        if req.level_like:
            summary = (
                f"Found {combined_total} {label} matching level"
                f" '{req.level_like}', showing {min(combined_total, limit)}."
            )
        else:
            summary = (
                f"Found {combined_total} {label},"
                f" showing {min(combined_total, limit)}."
            )

    return {
        "route": "sql",
        "decision": decision.reason,
        "db_paths": [str(p) for p in db_paths],
        "answer": summary,
        "data": {
            "intent": last_payload.get("intent"),
            "filters": last_payload.get("filters"),
            "count": combined_count,
            "total_count": combined_total,
            "limit": last_payload.get("limit"),
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
) -> dict[str, Any]:
    """Execute a query through the full pipeline (routing + execution).

    Args:
        question: User question
        db_paths: All SQLite databases to query
        graph: NetworkX graph (or None, will be loaded if needed)
        agent: Graph agent (or None, will be created if needed)
        decision: Optional precomputed routing decision
        debug_llm_io: Enable debug printing

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

        # Graph route
        graph, agent = _ensure_graph_context(graph, agent, debug_llm_io)
        result = execute_graph_query(question, graph, agent, decision)
        return {"result": result, "graph": graph, "agent": agent}

    except Exception as exc:
        error_result = _routing_error(decision, str(exc))
        return {"result": error_result, "graph": graph, "agent": agent}


def _ensure_graph_context(
    graph: nx.DiGraph | None,
    agent: GraphAgent | None,
    debug_llm_io: bool,
) -> tuple[nx.DiGraph, GraphAgent]:
    """Load graph and agent instances when missing."""
    if graph is None:
        graph = load_graph()
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
