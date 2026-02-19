from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rag_tag.observability import LogfireStatus, setup_logfire
from rag_tag.query_service import execute_query, find_sqlite_db
from rag_tag.router import route_question
from rag_tag.tui import print_answer, print_question, print_welcome


def _parse_bool(value: str | None) -> bool:
    if value is None:
        return True
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError("Expected a boolean value (true/false).")


def _resolve_db_path(db_path: Path | None) -> tuple[Path | None, str | None]:
    if db_path is None:
        return find_sqlite_db(), None

    candidate = db_path.expanduser().resolve()
    if candidate.is_file():
        return candidate, None

    return None, f"SQLite database not found: {candidate}"


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
        help="Enable Logfire observability for PydanticAI agents.",
    )
    ap.add_argument(
        "--tui",
        action="store_true",
        default=False,
        help="Launch Textual TUI instead of CLI.",
    )
    args = ap.parse_args()

    # Initialize Logfire if requested.
    # When --tui is used, suppress console output (print/warnings) from
    # setup_logfire so nothing is written to stderr before the TUI starts.
    # The TUI banner will display trace status instead.
    logfire_status: LogfireStatus = setup_logfire(
        enabled=args.trace,
        console=not args.tui,
    )

    # Resolve database path
    db_path, db_error = _resolve_db_path(args.db)
    if db_error:
        print(db_error, file=sys.stderr)
        # The user explicitly specified a path that does not exist.
        # Do not silently fall back to auto-detection; exit with an error
        # so the problem is visible rather than hidden behind a wrong DB.
        return 1

    # Launch TUI if requested
    if args.tui:
        from rag_tag.textual_app import run_tui

        run_tui(
            db_path=db_path,
            debug_llm_io=args.input,
            trace_enabled=args.trace,
            logfire_url=logfire_status.url if logfire_status.enabled else None,
        )
        return 0

    # CLI mode (stdin loop)
    graph = None
    agent = None

    print_welcome(str(db_path) if db_path else None)

    show_verbose = args.verbose or args.input

    for line in sys.stdin:
        question = line.strip()
        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            break

        try:
            decision = route_question(question, debug_llm_io=args.input)
            print_question(question, decision.route, decision.reason)

            # Execute query via shared service
            result_bundle = execute_query(
                question,
                db_path,
                graph,
                agent,
                decision=decision,
                debug_llm_io=args.input,
            )

            # Extract components
            result = result_bundle["result"]
            graph = result_bundle.get("graph") or graph
            agent = result_bundle.get("agent") or agent

        except Exception as exc:
            # Print question even on error (decision may not exist).
            print_question(question, "?", "routing failed")
            result = {"error": str(exc)}

        print_answer(result, verbose=show_verbose)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
