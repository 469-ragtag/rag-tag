"""Terminal UI formatting for the IFC query agent."""

from __future__ import annotations

import json
import sys
from typing import Any

# ANSI escape codes for terminal styling.
_BOLD = "\033[1m"
_DIM = "\033[2m"
_CYAN = "\033[36m"
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_RED = "\033[31m"
_RESET = "\033[0m"


def _supports_color() -> bool:
    """Return True if stderr/stdout likely support ANSI colours."""
    if not hasattr(sys.stdout, "isatty"):
        return False
    return sys.stdout.isatty()


# Disable colour codes when piped.
if not _supports_color():
    _BOLD = _DIM = _CYAN = _GREEN = _YELLOW = _RED = _RESET = ""


# ── Public API ──────────────────────────────────────────────────────────


def print_welcome(db_path: str | None) -> None:
    """Print the startup banner."""
    print(f"{_BOLD}IFC Query Agent{_RESET}")
    if db_path:
        print(f"  db: {_DIM}{db_path}{_RESET}")
    else:
        print(f"  db: {_DIM}(none — run parser/csv_to_sql.py){_RESET}")
    print(f"  Type a question, or {_BOLD}exit{_RESET} to quit.\n")


def print_question(question: str, route: str, reason: str) -> None:
    """Print the question header with routing info."""
    print(f"\n{_BOLD}{_CYAN}Q:{_RESET} {question}")
    print(f"   {_DIM}[{route}] {reason}{_RESET}")


def print_answer(result: dict[str, Any], *, verbose: bool = False) -> None:
    """Print a human-readable answer, optionally followed by details."""
    error = result.get("error")
    if error:
        _print_error(str(error))
        return

    route = result.get("route", "")

    if route == "sql":
        _print_sql_answer(result, verbose=verbose)
    elif route == "graph":
        _print_graph_answer(result, verbose=verbose)
    else:
        # Unknown route — just dump what we have.
        answer = result.get("answer")
        if answer:
            print(f"\n{_BOLD}{_GREEN}A:{_RESET} {answer}")
        if verbose:
            _print_details(result)


# ── SQL answer ──────────────────────────────────────────────────────────


def _print_sql_answer(result: dict[str, Any], *, verbose: bool) -> None:
    summary = result.get("answer") or ""
    data = result.get("data") or {}
    intent = data.get("intent", "")

    print(f"\n{_BOLD}{_GREEN}A:{_RESET} {summary}")

    if intent == "list":
        _print_item_table(data)

    if verbose:
        _print_details(result)


def _print_item_table(data: dict[str, Any]) -> None:
    """Print a compact table for list results."""
    items = data.get("items")
    if not items or not isinstance(items, list):
        return

    total = data.get("total_count", len(items))
    limit = data.get("limit", len(items))
    shown = min(total, limit)

    # Column definitions: (header, key, max_width)
    columns: list[tuple[str, str, int]] = [
        ("Name", "name", 30),
        ("Class", "ifc_class", 20),
        ("Level", "level", 20),
        ("Type", "type_name", 20),
    ]

    # Filter to columns that actually have data.
    active_cols: list[tuple[str, str, int]] = []
    for header, key, width in columns:
        if any(item.get(key) for item in items):
            active_cols.append((header, key, width))

    if not active_cols:
        return

    # Print header.
    header_parts = [f"{'#':>4}"]
    for header, _key, width in active_cols:
        header_parts.append(f"{header:<{width}}")
    header_line = "  ".join(header_parts)
    separator = "  ".join(["-" * 4] + ["-" * width for _, _, width in active_cols])

    print(f"\n   {_DIM}{header_line}{_RESET}")
    print(f"   {_DIM}{separator}{_RESET}")

    for i, item in enumerate(items, 1):
        row_parts = [f"{i:>4}"]
        for _header, key, width in active_cols:
            val = str(item.get(key) or "")
            if len(val) > width:
                val = val[: width - 1] + "\u2026"
            row_parts.append(f"{val:<{width}}")
        print(f"   {'  '.join(row_parts)}")

    if total > shown:
        print(f"\n   {_DIM}({total - shown} more not shown){_RESET}")


# ── Graph answer ────────────────────────────────────────────────────────


def _print_graph_answer(result: dict[str, Any], *, verbose: bool) -> None:
    answer = result.get("answer") or "No answer produced."
    print(f"\n{_BOLD}{_GREEN}A:{_RESET} {answer}")

    # Show sample elements if present.
    sample = (result.get("data") or {}).get("sample")
    if sample and isinstance(sample, list):
        print(f"\n   {_DIM}Sample:{_RESET}")
        for item in sample:
            print(f"   - {item}")

    if verbose:
        _print_details(result)


# ── Helpers ─────────────────────────────────────────────────────────────


def _print_error(error: str) -> None:
    """Print an error message."""
    print(f"\n{_BOLD}{_RED}Error:{_RESET} {error}")


def _print_details(result: dict[str, Any]) -> None:
    """Print full JSON details under a separator."""
    # Remove keys already shown in the summary to reduce noise.
    detail = {k: v for k, v in result.items() if k not in ("route", "answer")}
    if not detail:
        return
    print(f"\n   {_DIM}--- Details ---{_RESET}")
    formatted = json.dumps(detail, indent=2)
    for line in formatted.splitlines():
        print(f"   {_DIM}{line}{_RESET}")
