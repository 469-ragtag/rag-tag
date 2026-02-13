from __future__ import annotations

import argparse
import os
import sys
import uuid
from pathlib import Path

from rag_tag.agent import GraphAgent
from rag_tag.ifc_sql_tool import SqlQueryError, query_ifc_sql
from rag_tag.llm import resolve_provider
from rag_tag.paths import find_project_root
from rag_tag.router import RouteDecision, route_question
from rag_tag.trace import TraceWriter, to_trace_event, truncate_string
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
    trace: TraceWriter | None = None,
    run_id: str | None = None,
) -> dict[str, object]:
    if decision.sql_request is None:
        error_msg = "Router did not produce a SQL request."
        if trace and run_id:
            trace.write(
                to_trace_event(
                    "error",
                    run_id,
                    payload={"error": error_msg, "route": "sql"},
                )
            )
        return {
            "route": "sql",
            "decision": decision.reason,
            "error": error_msg,
        }
    if db_path is None:
        error_msg = "No SQLite database found. Run parser/csv_to_sql.py."
        if trace and run_id:
            trace.write(
                to_trace_event(
                    "error",
                    run_id,
                    payload={"error": error_msg, "route": "sql"},
                )
            )
        return {
            "route": "sql",
            "decision": decision.reason,
            "error": error_msg,
        }

    try:
        envelope = query_ifc_sql(db_path, decision.sql_request)
    except SqlQueryError as exc:
        error_str = str(exc)
        if trace and run_id:
            trace.write(
                to_trace_event(
                    "error",
                    run_id,
                    payload={"error": error_str, "route": "sql"},
                )
            )
        return {
            "route": "sql",
            "decision": decision.reason,
            "error": error_str,
        }

    # Check envelope status
    if envelope["status"] == "error":
        error_info = envelope.get("error", {})
        error_msg = error_info.get("message", "Unknown SQL error")
        if trace and run_id:
            trace.write(
                to_trace_event(
                    "error",
                    run_id,
                    payload={"error": error_msg, "route": "sql"},
                )
            )
        return {
            "route": "sql",
            "decision": decision.reason,
            "error": error_msg,
        }

    # Extract payload from envelope
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

    if trace and run_id:
        answer = payload.get("summary")
        answer_snippet = truncate_string(str(answer)) if answer else None
        trace.write(
            to_trace_event(
                "final",
                run_id,
                payload={
                    "route": "sql",
                    "answer_length": len(str(answer)) if answer else 0,
                    "answer_snippet": answer_snippet,
                    "count": payload.get("count"),
                    "total_count": payload.get("total_count"),
                },
            )
        )

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
    ap.add_argument(
        "--trace",
        action="store_true",
        default=False,
        help="Enable JSONL tracing of agent execution.",
    )
    ap.add_argument(
        "--trace-path",
        type=Path,
        default=None,
        help="Path to trace output file (defaults to output/agent_trace.jsonl).",
    )
    args = ap.parse_args()

    # Initialize trace writer if requested
    trace: TraceWriter | None = None
    run_id: str | None = None
    if args.trace:
        run_id = uuid.uuid4().hex
        if args.trace_path:
            trace_path = args.trace_path.expanduser().resolve()
        else:
            project_root = find_project_root(Path(__file__).resolve().parent)
            if project_root:
                output_dir = project_root / "output"
                output_dir.mkdir(exist_ok=True)
                trace_path = output_dir / "agent_trace.jsonl"
            else:
                trace_path = Path.cwd() / "agent_trace.jsonl"
        trace = TraceWriter(trace_path)

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

                # Emit route_decision trace event
                if trace and run_id:
                    route_payload: dict[str, object] = {
                        "route": decision.route,
                        "reason": decision.reason,
                    }
                    if decision.sql_request:
                        route_payload["sql_request_intent"] = (
                            decision.sql_request.intent
                        )
                        if decision.sql_request.ifc_class:
                            route_payload["sql_request_class"] = (
                                decision.sql_request.ifc_class
                            )
                    trace.write(
                        to_trace_event("route_decision", run_id, payload=route_payload)
                    )

                if decision.route == "sql":
                    result = _sql_result(decision, db_path, trace, run_id)
                else:
                    if graph is None:
                        graph = _load_graph()
                    if agent is None:
                        agent = _create_graph_agent(debug_llm_io=args.input)

                    agent_result = agent.run(
                        question, graph, max_steps=6, trace=trace, run_id=run_id
                    )
                    result = {
                        "route": "graph",
                        "decision": decision.reason,
                        **agent_result,
                    }
            except Exception as exc:
                # Print question even on error (decision may not exist).
                print_question(question, "?", "routing failed")
                result = {"error": str(exc)}
                if trace and run_id:
                    trace.write(
                        to_trace_event(
                            "error",
                            run_id,
                            payload={"error": str(exc), "stage": "main_loop"},
                        )
                    )

            show_verbose = args.verbose or args.input
            print_answer(result, verbose=show_verbose)
    finally:
        if trace:
            trace.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
