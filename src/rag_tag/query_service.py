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
    """Return all discovered SQLite DBs in deterministic name order."""
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


def load_graph(dataset: str | None = None) -> nx.DiGraph:
    """Build and return the IFC graph for the selected dataset."""
    from rag_tag.parser.jsonl_to_graph import build_graph  # noqa: PLC0415

    return build_graph(dataset=dataset)


def execute_sql_query(
    decision: RouteDecision,
    db_paths: list[Path],
) -> dict[str, Any]:
    """Execute SQL intent across all DBs and merge into one response."""
    if decision.sql_request is None:
        return _sql_error(decision, "Router did not produce a SQL request.")
    if not db_paths:
        return _sql_error(
            decision, "No SQLite database found. Run rag-tag-jsonl-to-sql."
        )

    req = decision.sql_request
    # NOTE: Keep one canonical limit for merged multi-DB list responses.
    effective_limit = req.limit or 50

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

    # NOTE: Per-DB queries each apply LIMIT; merged rows can still exceed it.
    if req.intent == "list":
        combined_items = combined_items[:effective_limit]

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
    """Execute a graph-route question through the graph agent."""
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
) -> dict[str, Any]:
    """Route a question and execute it via SQL or graph tools."""
    try:
        if decision is None:
            decision = route_question(question, debug_llm_io=debug_llm_io)

        if decision.route == "sql":
            result = execute_sql_query(decision, db_paths)
            return {"result": result, "graph": graph, "agent": agent}

        graph, agent = _ensure_graph_context(graph, agent, debug_llm_io, graph_dataset)
        result = execute_graph_query(question, graph, agent, decision)
        return {"result": result, "graph": graph, "agent": agent}

    except Exception as exc:
        error_result = _routing_error(decision, str(exc))
        return {"result": error_result, "graph": graph, "agent": agent}


def _ensure_graph_context(
    graph: nx.DiGraph | None,
    agent: GraphAgent | None,
    debug_llm_io: bool,
    graph_dataset: str | None = None,
) -> tuple[nx.DiGraph, GraphAgent]:
    """Load graph and agent instances when missing."""
    if graph is None:
        graph = load_graph(graph_dataset)
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
