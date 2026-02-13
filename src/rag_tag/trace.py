"""Structured tracing for agent execution with JSONL output."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class TraceEvent:
    """Single trace event with timestamp and metadata."""

    timestamp: str
    run_id: str
    event: str
    step_id: int | None
    payload: dict[str, Any]


class TraceWriter:
    """JSONL trace writer for agent execution events."""

    def __init__(self, path: Path) -> None:
        """Initialize trace writer.

        Args:
            path: Output file path (will be opened in append mode)
        """
        self._path = path
        self._file = path.open("a", encoding="utf-8")

    def write(self, event: TraceEvent) -> None:
        """Write a trace event as a single JSON line.

        Args:
            event: TraceEvent to serialize and write
        """
        data = {
            "timestamp": event.timestamp,
            "run_id": event.run_id,
            "event": event.event,
            "step_id": event.step_id,
            "payload": event.payload,
        }
        line = json.dumps(data, ensure_ascii=True, separators=(",", ":"))
        self._file.write(line)
        self._file.write("\n")
        self._file.flush()

    def close(self) -> None:
        """Close the trace file."""
        self._file.close()


def to_trace_event(
    event: str,
    run_id: str,
    step_id: int | None = None,
    payload: dict[str, Any] | None = None,
) -> TraceEvent:
    """Create a trace event with current timestamp.

    Args:
        event: Event type (e.g., 'route_decision', 'tool_call', 'final')
        run_id: Unique run identifier
        step_id: Optional step number
        payload: Optional event-specific metadata

    Returns:
        TraceEvent instance
    """
    return TraceEvent(
        timestamp=datetime.now(timezone.utc).isoformat(),
        run_id=run_id,
        event=event,
        step_id=step_id,
        payload=payload or {},
    )


def truncate_string(text: str, max_length: int = 120) -> str:
    """Truncate string to max length with ellipsis if needed.

    Args:
        text: String to truncate
        max_length: Maximum length (default 120)

    Returns:
        Truncated string
    """
    if len(text) <= max_length:
        return text
    return text[: max_length - 3] + "..."
