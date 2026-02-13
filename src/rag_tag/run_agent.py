from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from rag_tag.agent import GraphAgent
from rag_tag.ifc_sql_tool import SqlQueryError, query_ifc_sql
from rag_tag.llm import resolve_provider
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


def _parse_bool(value: str | None) -> bool:
    if value is None:
        return True
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError("Expected a boolean value (true/false).")


def _create_graph_agent(*, debug_llm_io: bool = False) -> GraphAgent:
    """Create graph agent with provider resolution.

    Provider selection priority:
    1. AGENT_PROVIDER env var
    2. LLM_PROVIDER env var
    3. Auto-detect from API keys

    Args:
        debug_llm_io: Enable debug printing of LLM I/O

    Returns:
        Initialized GraphAgent instance

    Raises:
        RuntimeError: If no provider can be resolved
    """
    provider_name = os.getenv("AGENT_PROVIDER") or os.getenv("LLM_PROVIDER")
    model_name = os.getenv("AGENT_MODEL")

    provider = resolve_provider(
        name=provider_name,
        model=model_name,
        debug_llm_io=debug_llm_io,
    )

    return GraphAgent(provider, debug_llm_io=debug_llm_io)


def main() -> int:
    ap = argparse.ArgumentParser(description="IFC query agent CLI")
    ap.add_argument(
        "--input",
        nargs="?",
        const=True,
        default=False,
        type=_parse_bool,
        help=(
            "Print LLM input/output for router and agent to stderr "
            "(use --input, --input=true, or --input=false)."
        ),
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
    args = ap.parse_args()

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
                    agent = _create_graph_agent(debug_llm_io=args.input)

                agent_result = agent.run(question, graph, max_steps=6)
                result = {
                    "route": "graph",
                    "decision": decision.reason,
                    **agent_result,
                }
        except Exception as exc:
            # Print question even on error (decision may not exist).
            print_question(question, "?", "routing failed")
            result = {"error": str(exc)}

        show_verbose = args.verbose or args.input
        print_answer(result, verbose=show_verbose)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
