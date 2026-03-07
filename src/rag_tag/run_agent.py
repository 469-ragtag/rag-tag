from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from rag_tag.observability import LogfireStatus, setup_logfire
from rag_tag.query_service import execute_query, find_sqlite_dbs
from rag_tag.router import route_question
from rag_tag.tui import print_answer, print_question, print_welcome

_logger = logging.getLogger(__name__)

_VALID_PAYLOAD_MODES = frozenset({"full", "minimal"})


def _resolve_payload_mode() -> str:
    """Read and validate the ``GRAPH_PAYLOAD_MODE`` environment variable.

    Returns ``'full'`` or ``'minimal'``.  Falls back to ``'full'`` with a
    logged warning when the variable is set to an unrecognised value.
    """
    raw = os.environ.get("GRAPH_PAYLOAD_MODE", "full").strip().lower()
    if raw not in _VALID_PAYLOAD_MODES:
        _logger.warning(
            "Unsupported GRAPH_PAYLOAD_MODE=%r; defaulting to 'full'. "
            "Valid values: %s.",
            raw,
            ", ".join(sorted(_VALID_PAYLOAD_MODES)),
        )
        return "full"
    return raw


def _parse_dataset(value: str) -> str:
    dataset = value.strip()
    if not dataset:
        raise argparse.ArgumentTypeError("Dataset name cannot be empty.")
    return dataset


def _parse_bool(value: str | None) -> bool:
    if value is None:
        return True
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError("Expected a boolean value (true/false).")


def _resolve_db_paths(db_path: Path | None) -> tuple[list[Path], str | None]:
    if db_path is None:
        return find_sqlite_dbs(), None

    candidate = db_path.expanduser().resolve()
    if candidate.is_file():
        return [candidate], None

    return [], f"SQLite database not found: {candidate}"


def _resolve_graph_dataset(
    graph_dataset: str | None, db_path: Path | None
) -> str | None:
    """Return the dataset stem to load for graph queries.

    Priority (highest to lowest):
    1. Explicit ``--graph-dataset`` flag  → return as-is.
    2. Single ``--db`` / auto-detected DB → use the DB file stem so the graph
       file matches the loaded SQL database.
    3. No constraint at all              → return None so graph-query execution
       can require an explicit dataset when multiple datasets are present.
    """
    if graph_dataset:
        return graph_dataset
    if db_path is not None:
        return db_path.stem
    return None


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
            "(defaults to all .db files in output/ or db/ sorted by name; "
            "when exactly one DB is found its stem is used as the graph dataset)."
        ),
    )
    ap.add_argument(
        "--graph-dataset",
        type=_parse_dataset,
        default=None,
        help=(
            "Dataset stem for graph files (<project>/output/<stem>.jsonl). "
            "Overrides --db stem inference. "
            "When omitted, the selected DB stem is used if available. "
            "If multiple graph datasets exist, graph queries require either "
            "--graph-dataset or --db output/<stem>.db."
        ),
    )
    ap.add_argument(
        "--trace",
        action="store_true",
        default=False,
        help="Enable Logfire observability for PydanticAI agents.",
    )
    ap.add_argument(
        "--strict-sql",
        action="store_true",
        default=False,
        help="Fail closed if any SQL database query fails during merged SQL execution.",
    )
    ap.add_argument(
        "--tui",
        action="store_true",
        default=False,
        help="Launch Textual TUI instead of CLI.",
    )
    args = ap.parse_args()

    # Validate GRAPH_PAYLOAD_MODE early so the user sees a clear warning
    # rather than a silent fallback deep in the pipeline.
    graph_payload_mode = _resolve_payload_mode()

    # Initialize Logfire if requested.
    # When --tui is used, suppress console output (print/warnings) from
    # setup_logfire so nothing is written to stderr before the TUI starts.
    # The TUI banner will display trace status instead.
    logfire_status: LogfireStatus = setup_logfire(
        enabled=args.trace,
        console=not args.tui,
    )

    # Resolve database paths
    db_paths, db_error = _resolve_db_paths(args.db)
    if db_error:
        print(db_error, file=sys.stderr)
        # The user explicitly specified a path that does not exist.
        # Do not silently fall back to auto-detection; exit with an error
        # so the problem is visible rather than hidden behind a wrong DB.
        return 1

    resolved_db_path = args.db.expanduser().resolve() if args.db is not None else None
    if resolved_db_path is None and len(db_paths) == 1:
        resolved_db_path = db_paths[0]

    # Resolve dataset once, shared by both TUI and CLI paths.
    # Use resolved_db_path (not args.db) so auto-detected DBs contribute their
    # stem when --graph-dataset is not explicitly supplied.
    graph_dataset = _resolve_graph_dataset(args.graph_dataset, resolved_db_path)

    # Launch TUI if requested
    if args.tui:
        from rag_tag.textual_app import run_tui

        run_tui(
            db_paths=db_paths,
            debug_llm_io=args.input,
            trace_enabled=args.trace,
            logfire_url=logfire_status.url if logfire_status.enabled else None,
            graph_dataset=graph_dataset,
            context_db=resolved_db_path,
            graph_payload_mode=graph_payload_mode,
        )
        return 0

    # CLI mode (stdin loop)
    graph = None
    agent = None

    db_label = ", ".join(p.name for p in db_paths) if db_paths else None
    print_welcome(db_label)

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
                db_paths,
                graph,
                agent,
                decision=decision,
                debug_llm_io=args.input,
                graph_dataset=graph_dataset,
                context_db=resolved_db_path,
                payload_mode=graph_payload_mode,
                strict_sql=args.strict_sql,
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
