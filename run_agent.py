from __future__ import annotations

import json
import sys
from pathlib import Path

from command_r_agent import CommandRAgent
from ifc_graph_tool import query_ifc_graph
from ifc_sql_tool import SqlQueryError, query_ifc_sql
from query_router import RouteDecision, route_question


def _load_graph():
    # Ensure parser directory is importable for existing module structure
    repo_root = Path(__file__).resolve().parent
    parser_dir = repo_root / "parser"
    sys.path.insert(0, str(parser_dir))

    import csv_to_graph  # type: ignore

    return csv_to_graph.G


def _find_sqlite_db() -> Path | None:
    repo_root = Path(__file__).resolve().parent
    candidates: list[Path] = []
    for folder_name in ("output", "db"):
        folder = repo_root / folder_name
        if not folder.exists():
            continue
        candidates.extend(folder.glob("*.db"))
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _sql_result(
    decision: RouteDecision,
    db_path: Path | None,
) -> dict[str, object]:
    if decision.sql_request is None:
        return {
            "route": "sql",
            "decision": decision.reason,
            "error": "Router did not produce a SQL request.",
        }
    if db_path is None:
        return {
            "route": "sql",
            "decision": decision.reason,
            "error": "No SQLite database found. Run parser/csv_to_sql.py.",
        }

    try:
        payload = query_ifc_sql(db_path, decision.sql_request)
    except SqlQueryError as exc:
        return {
            "route": "sql",
            "decision": decision.reason,
            "error": str(exc),
        }

    return {
        "route": "sql",
        "decision": decision.reason,
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


def main() -> int:
    graph = None
    agent = None
    db_path = _find_sqlite_db()
    print("IFC Query Agent ready. Type a question or 'exit'.")

    for line in sys.stdin:
        question = line.strip()
        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            break

        try:
            decision = route_question(question)
            if decision.route == "sql":
                result = _sql_result(decision, db_path)
            else:
                if graph is None:
                    graph = _load_graph()
                if agent is None:
                    agent = CommandRAgent()

                state = {"history": []}
                max_steps = 6
                for _ in range(max_steps):
                    step = agent.plan(question, state)
                    step_type = step.get("type")

                    if step_type == "final":
                        result = {
                            "route": "graph",
                            "decision": decision.reason,
                            "answer": step.get("answer"),
                        }
                        break

                    if step_type != "tool":
                        result = {
                            "route": "graph",
                            "decision": decision.reason,
                            "error": "Invalid step type",
                            "step": step,
                        }
                        break

                    action = step.get("action")
                    params = step.get("params", {})
                    tool_result = query_ifc_graph(graph, action, params)
                    state["history"].append(
                        {
                            "tool": {"action": action, "params": params},
                            "result": tool_result,
                        }
                    )
                else:
                    result = {
                        "route": "graph",
                        "decision": decision.reason,
                        "error": "Max steps exceeded",
                        "history": state["history"],
                    }
        except Exception as exc:
            result = {"error": str(exc)}

        print(json.dumps(result, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
