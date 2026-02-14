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


def print_answer(
    result: dict[str, Any],
    *,
    verbose: bool = False,
    verbose_short: bool = False,
) -> None:
    """Print a human-readable answer, optionally followed by details."""
    error = result.get("error")
    if error:
        _print_error(str(error))
        return

    route = result.get("route", "")

    if route == "sql":
        _print_sql_answer(result, verbose=verbose, verbose_short=verbose_short)
    elif route == "graph":
        _print_graph_answer(result, verbose=verbose, verbose_short=verbose_short)
    else:
        # Unknown route — just dump what we have.
        answer = result.get("answer")
        if answer:
            print(f"\n{_BOLD}{_GREEN}A:{_RESET} {answer}")
        if verbose:
            _print_details(result)
        elif verbose_short:
            _print_details(result, short=True)


# ── SQL answer ──────────────────────────────────────────────────────────


def _print_sql_answer(
    result: dict[str, Any],
    *,
    verbose: bool,
    verbose_short: bool,
) -> None:
    summary = result.get("answer") or ""
    data = result.get("data") or {}
    intent = data.get("intent", "")

    print(f"\n{_BOLD}{_GREEN}A:{_RESET} {summary}")

    if intent == "list":
        _print_item_table(data)

    if verbose:
        _print_details(result)
    elif verbose_short:
        _print_details(result, short=True)


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


def _print_graph_answer(
    result: dict[str, Any],
    *,
    verbose: bool,
    verbose_short: bool,
) -> None:
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
    elif verbose_short:
        _print_details(result, short=True)


# ── Helpers ─────────────────────────────────────────────────────────────


def _print_error(error: str) -> None:
    """Print an error message."""
    print(f"\n{_BOLD}{_RED}Error:{_RESET} {error}")


def _print_details(result: dict[str, Any], *, short: bool = False) -> None:
    """Print full JSON details under a separator."""
    # Remove keys already shown in the summary to reduce noise.
    detail = {k: v for k, v in result.items() if k not in ("route", "answer")}
    if not detail:
        return
    if short:
        detail = _compact_detail(detail)
    print(f"\n   {_DIM}--- Details ---{_RESET}")
    formatted = json.dumps(detail, indent=2)
    for line in formatted.splitlines():
        print(f"   {_DIM}{line}{_RESET}")


def _compact_detail(detail: dict[str, Any]) -> dict[str, Any]:
    compact = dict(detail)
    llm_debug = compact.get("llm_debug")
    if isinstance(llm_debug, dict):
        compact["llm_debug"] = {
            key: _compact_llm_debug_component(value)
            for key, value in llm_debug.items()
            if isinstance(value, dict)
        }
    return compact


def _compact_llm_debug_component(component: dict[str, Any]) -> dict[str, Any]:
    compact: dict[str, Any] = {}
    for key in (
        "component",
        "provider",
        "model",
        "mode",
        "fallback_used",
        "structured_error",
        "error",
    ):
        if key in component:
            compact[key] = component[key]

    input_payload = component.get("input")
    if isinstance(input_payload, dict):
        question = input_payload.get("question")
        if isinstance(question, str):
            compact["input_question"] = _truncate(question, 220)

    if "output" in component:
        compact["output"] = component["output"]

    messages = component.get("messages")
    if isinstance(messages, list):
        compact["message_summary"] = _summarize_messages(messages)

    return compact


def _summarize_messages(messages: list[Any]) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        item: dict[str, Any] = {}
        kind = message.get("kind")
        if isinstance(kind, str):
            item["kind"] = kind

        parts = message.get("parts")
        part_summaries: list[dict[str, Any]] = []
        if isinstance(parts, list):
            for part in parts:
                if not isinstance(part, dict):
                    continue
                part_kind = part.get("part_kind")
                if part_kind == "tool-call":
                    part_summaries.append(
                        {
                            "part": "tool-call",
                            "tool": part.get("tool_name"),
                            "args": _truncate(str(part.get("args", "")), 180),
                        }
                    )
                elif part_kind == "tool-return":
                    status = None
                    content = part.get("content")
                    if isinstance(content, dict):
                        status = content.get("status")
                    part_summaries.append(
                        {
                            "part": "tool-return",
                            "tool": part.get("tool_name"),
                            "status": status,
                        }
                    )
                elif part_kind == "text":
                    text = part.get("content")
                    if isinstance(text, str):
                        part_summaries.append(
                            {
                                "part": "text",
                                "content": _truncate(text, 180),
                            }
                        )
                elif part_kind == "retry-prompt":
                    part_summaries.append({"part": "retry-prompt"})
                elif part_kind == "user-prompt":
                    prompt = part.get("content")
                    if isinstance(prompt, str):
                        part_summaries.append(
                            {
                                "part": "user-prompt",
                                "content": _truncate(prompt, 160),
                            }
                        )

        if part_summaries:
            item["parts"] = part_summaries
        if item:
            summary.append(item)
    return summary


def _truncate(text: str, max_len: int) -> str:
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + "…"
