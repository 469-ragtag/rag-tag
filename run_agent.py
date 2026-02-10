from __future__ import annotations

import argparse
import sys
from pathlib import Path

from command_r_agent import CommandRAgent
from ifc_graph_tool import query_ifc_graph
from ifc_sql_tool import SqlQueryError, query_ifc_sql
from router import RouteDecision, route_question
from tui import print_answer, print_question, print_welcome


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
                    agent = CommandRAgent(debug_llm_io=args.input)

                state: dict[str, object] = {"history": []}
                max_steps = 6
                result: dict[str, object] = {}
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

                    action_value = step.get("action")
                    params = step.get("params", {})
                    if not isinstance(action_value, str) or not action_value:
                        result = {
                            "route": "graph",
                            "decision": decision.reason,
                            "error": "Invalid tool action",
                            "step": step,
                        }
                        break
                    tool_result = query_ifc_graph(graph, action_value, params)
                    history = state.get("history", [])
                    if isinstance(history, list):
                        history.append(
                            {
                                "tool": {
                                    "action": action_value,
                                    "params": params,
                                },
                                "result": tool_result,
                            }
                        )
                else:
                    history = state.get("history", [])
                    summary = (
                        _summarize_graph_history(history)
                        if isinstance(history, list)
                        else None
                    )
                    result = {
                        "route": "graph",
                        "decision": decision.reason,
                        "error": "Max steps exceeded",
                        "history": history,
                    }
                    if summary:
                        result.update(summary)
        except Exception as exc:
            # Print question even on error (decision may not exist).
            print_question(question, "?", "routing failed")
            result = {"error": str(exc)}

        show_verbose = args.verbose or args.input
        print_answer(result, verbose=show_verbose)

    return 0


def _summarize_graph_history(
    history: list[dict[str, object]],
) -> dict[str, object] | None:
    for entry in reversed(history):
        result = entry.get("result")
        if not isinstance(result, dict):
            continue
        elements = result.get("elements")
        if isinstance(elements, list):
            labels = [
                e.get("label") or e.get("id") for e in elements if isinstance(e, dict)
            ]
            sample = [label for label in labels if label]
            return {
                "answer": f"Found {len(elements)} elements.",
                "data": {"sample": sample[:5]},
            }
        adjacent = result.get("adjacent")
        if isinstance(adjacent, list):
            labels = [
                e.get("label") or e.get("id") for e in adjacent if isinstance(e, dict)
            ]
            sample = [label for label in labels if label]
            return {
                "answer": f"Found {len(adjacent)} adjacent elements.",
                "data": {"sample": sample[:5]},
            }
        results = result.get("results")
        if isinstance(results, list):
            return {
                "answer": f"Found {len(results)} traversal results.",
                "data": {"sample": results[:5]},
            }
    return None


if __name__ == "__main__":
    raise SystemExit(main())
