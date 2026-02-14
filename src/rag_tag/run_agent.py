from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from rag_tag.agent import GraphAgent
from rag_tag.ifc_sql_tool import SqlQueryError, query_ifc_sql
from rag_tag.observability import setup_logfire
from rag_tag.paths import find_project_root
from rag_tag.router import RouteDecision, route_question
from rag_tag.tui import print_answer, print_question, print_welcome


def _load_graph():
    from rag_tag.parser.csv_to_graph import build_graph

    return build_graph()


def _find_sqlite_db() -> Path | None:
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


def _sql_result(
    decision: RouteDecision,
    db_path: Path | None,
) -> dict[str, object]:
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

    payload = envelope["data"]
    result: dict[str, object] = {
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


def _parse_bool(value: str | None) -> bool:
    if value is None:
        return True
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError("Expected a boolean value (true/false).")


def _create_graph_agent() -> GraphAgent:
    """Create graph agent with provider resolution.

    Provider selection priority:
    1. AGENT_PROVIDER env var
    2. LLM_PROVIDER env var
    3. Auto-detect from API keys

    Returns:
        Initialized GraphAgent instance

    Raises:
        RuntimeError: If no provider can be resolved
    """
    provider_name = os.getenv("AGENT_PROVIDER") or os.getenv("LLM_PROVIDER")
    model_name = os.getenv("AGENT_MODEL")

    return GraphAgent(model_name=model_name, provider_name=provider_name)


def main() -> int:
    ap = argparse.ArgumentParser(description="IFC query agent CLI")
    ap.add_argument(
        "--input",
        nargs="?",
        const=True,
        default=False,
        type=_parse_bool,
        help="Deprecated: Use --trace to enable Logfire instrumentation.",
    )
    ap.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        default=False,
        help="Show full JSON details below each answer.",
    )
    ap.add_argument(
        "--db",
        type=Path,
        default=None,
        help=(
            "Path to SQLite database for SQL routing "
            "(defaults to newest .db in output/ or db/)."
        ),
    )
    ap.add_argument(
        "--trace",
        action="store_true",
        default=False,
        help="Enable Logfire tracing of agent execution.",
    )
    ap.add_argument(
        "--trace-path",
        type=Path,
        default=None,
        help="Deprecated: Logfire tracing does not use file paths.",
    )
    args = ap.parse_args()

    if args.input:
        print(
            "Warning: --input is deprecated. Use --trace to enable Logfire.",
            file=sys.stderr,
        )

    if args.trace_path:
        print(
            "Warning: --trace-path is deprecated. Logfire handles trace storage.",
            file=sys.stderr,
        )

    setup_logfire(enabled=args.trace)

    try:
        graph = None
        agent = None
        db_path: Path | None
        if args.db is not None:
            candidate = args.db.expanduser().resolve()
            if candidate.is_file():
                db_path = candidate
            else:
                print(f"SQLite database not found: {candidate}", file=sys.stderr)
                db_path = None
        else:
            db_path = _find_sqlite_db()

        print_welcome(str(db_path) if db_path else None)

        for line in sys.stdin:
            question = line.strip()
            if not question:
                continue
            if question.lower() in {"exit", "quit"}:
                break

            try:
                decision = route_question(question, debug_llm_io=args.input)
                print_question(question, decision.route, decision.reason)

                if decision.route == "sql":
                    result = _sql_result(decision, db_path)
                else:
                    if graph is None:
                        graph = _load_graph()
                    if agent is None:
                        agent = _create_graph_agent()

                    agent_result = agent.run(question, graph, max_steps=6)
                    result = {
                        "route": "graph",
                        "decision": decision.reason,
                        **agent_result,
                    }
            except Exception as exc:
                print_question(question, "?", "routing failed")
                result = {"error": str(exc)}

            show_verbose = args.verbose or args.input
            print_answer(result, verbose=show_verbose)

    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
