from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

from rag_tag.ifc_sql_tool import SqlQueryError, query_ifc_sql
from rag_tag.router import RouteDecision, route_question

QUESTIONS: list[dict[str, Any]] = [
    {
        "question": "How many doors are on Level 2?",
        "expected_route": "sql",
    },
    {
        "question": "Count windows on the ground floor.",
        "expected_route": "sql",
    },
    {
        "question": "List all walls on Level 1.",
        "expected_route": "sql",
    },
    {
        "question": "Which rooms are adjacent to the kitchen?",
        "expected_route": "graph",
    },
    {
        "question": "Find doors near the stair core.",
        "expected_route": "graph",
    },
    {
        "question": "How many elements are in the basement?",
        "expected_route": "sql",
    },
    {
        "question": "Show me all columns on Level 3.",
        "expected_route": "sql",
    },
    {
        "question": "Which spaces are connected to the lobby?",
        "expected_route": "graph",
    },
    {
        "question": "List all windows.",
        "expected_route": "sql",
    },
    {
        "question": "Are there any windows in the building?",
        "expected_route": "sql",
    },
    {
        "question": "Find the path from the lobby to the server room.",
        "expected_route": "graph",
    },
    {
        "question": "Count the number of slabs.",
        "expected_route": "sql",
    },
    {
        "question": "Which rooms are near the stairwell?",
        "expected_route": "graph",
    },
    {
        "question": "List all doors on the ground floor.",
        "expected_route": "sql",
    },
    {
        "question": "How many spaces are in Level 1?",
        "expected_route": "sql",
    },
]


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Evaluate router decisions and SQL outputs for sample queries."
    )
    ap.add_argument(
        "--db",
        type=Path,
        default=None,
        help="Path to SQLite DB for SQL execution (optional).",
    )
    ap.add_argument(
        "--router-mode",
        type=str,
        default=None,
        help="Override ROUTER_MODE (rule or llm).",
    )
    ap.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero if any route mismatches expected_route.",
    )
    args = ap.parse_args()

    if args.router_mode:
        os.environ["ROUTER_MODE"] = args.router_mode

    db_path = args.db
    if db_path is not None:
        db_path = db_path.expanduser().resolve()

    mismatch_count = 0
    for item in QUESTIONS:
        question = item["question"]
        expected = item["expected_route"]
        decision = route_question(question)

        if decision.route != expected:
            mismatch_count += 1

        print("-")
        print(f"Q: {question}")
        print(f"Route: {decision.route} (expected {expected})")
        print(f"Reason: {decision.reason}")

        if decision.route == "sql":
            _print_sql_result(decision, db_path)
        else:
            print("Graph execution: skipped (use run_agent.py for graph reasoning).")

    print("-")
    print(f"Total questions: {len(QUESTIONS)}")
    print(f"Route mismatches: {mismatch_count}")

    if args.strict and mismatch_count > 0:
        return 1
    return 0


def _print_sql_result(decision: RouteDecision, db_path: Path | None) -> None:
    if decision.sql_request is None:
        print("SQL route without SQL request.")
        return

    if db_path is None:
        print("SQL execution: skipped (no --db provided).")
        return

    try:
        payload = query_ifc_sql(db_path, decision.sql_request)
    except SqlQueryError as exc:
        print(f"SQL error: {exc}")
        return

    summary = payload.get("summary")
    if summary:
        print(f"SQL summary: {summary}")
    else:
        print("SQL summary: (none)")


if __name__ == "__main__":
    raise SystemExit(main())
