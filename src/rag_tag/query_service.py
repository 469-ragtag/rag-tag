"""Shared query execution service for CLI and TUI."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import networkx as nx

from rag_tag.agent import GraphAgent
from rag_tag.ifc_sql_tool import SqlQueryError, query_ifc_sql
from rag_tag.paths import find_project_root
from rag_tag.router import RouteDecision, route_question


def find_sqlite_db() -> Path | None:
    """Find the most recent SQLite database in output/ or db/ folders.

    Returns:
        Path to the newest .db file, or None if none found.
    """
    project_root = find_project_root(Path(__file__).resolve().parent)
    if project_root is None:
        return None
    candidates: list[Path] = []
    for folder_name in ("output", "db"):
        folder = project_root / folder_name
        if not folder.exists():
            continue
        candidates.extend(folder.glob("*.db"))
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def load_graph() -> nx.DiGraph:
    """Load NetworkX graph from CSV data.

    Returns:
        NetworkX DiGraph instance.
    """
    from rag_tag.parser.csv_to_graph import build_graph

    return build_graph()


def execute_sql_query(
    decision: RouteDecision,
    db_path: Path | None,
) -> dict[str, Any]:
    """Execute SQL query based on routing decision.

    Args:
        decision: Routing decision with SQL request
        db_path: Path to SQLite database

    Returns:
        Result dict with answer, data, or error
    """
    if decision.sql_request is None:
        error_msg = "Router did not produce a SQL request."
        return {
            "route": "sql",
            "decision": decision.reason,
            "error": error_msg,
        }
    if db_path is None:
        error_msg = "No SQLite database found. Run parser/csv_to_sql.py."
        return {
            "route": "sql",
            "decision": decision.reason,
            "error": error_msg,
        }

    try:
        envelope = query_ifc_sql(db_path, decision.sql_request)
    except SqlQueryError as exc:
        error_str = str(exc)
        return {
            "route": "sql",
            "decision": decision.reason,
            "error": error_str,
        }

    # Check envelope status
    if envelope["status"] == "error":
        error_info = envelope.get("error", {})
        error_msg = error_info.get("message", "Unknown SQL error")
        error_code = error_info.get("code")
        if error_code:
            error_msg = f"{error_msg} (code: {error_code})"
        return {
            "route": "sql",
            "decision": decision.reason,
            "error": error_msg,
        }

    # Extract payload from envelope
    payload = envelope["data"]
    result: dict[str, Any] = {
        "route": "sql",
        "decision": decision.reason,
        "db_path": str(db_path),
        "answer": payload.get("summary"),
        "data": {
            "intent": payload.get("intent"),
            "filters": payload.get("filters"),
            "count": payload.get("count"),
            "total_count": payload.get("total_count"),
            "limit": payload.get("limit"),
            "items": payload.get("items"),
        },
        "sql": payload.get("sql"),
    }

    return result


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
    db_path: Path | None,
    graph: nx.DiGraph | None,
    agent: GraphAgent | None,
    *,
    decision: RouteDecision | None = None,
    debug_llm_io: bool = False,
) -> dict[str, Any]:
    """Execute a query through the full pipeline (routing + execution).

    Args:
        question: User question
        db_path: Path to SQLite database (or None)
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
            result = execute_sql_query(decision, db_path)
            return {"result": result, "graph": graph, "agent": agent}

        # Graph route
        if graph is None:
            graph = load_graph()
        if agent is None:
            agent = GraphAgent(debug_llm_io=debug_llm_io)

        result = execute_graph_query(question, graph, agent, decision)
        return {"result": result, "graph": graph, "agent": agent}

    except Exception as exc:
        error_result = {"error": str(exc), "route": "?", "decision": "routing failed"}
        if decision is not None:
            error_result["route"] = decision.route
            error_result["decision"] = decision.reason
        return {"result": error_result, "graph": graph, "agent": agent}
