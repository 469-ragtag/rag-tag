"""Textual TUI for the IFC query agent."""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

import networkx as nx
from textual.app import App, ComposeResult
from textual.containers import ScrollableContainer, Vertical
from textual.widgets import Footer, Header, Input, Static

from rag_tag.agent import GraphAgent
from rag_tag.query_service import execute_query, find_sqlite_db


class QueryApp(App[None]):
    """A minimal TUI for querying IFC data."""

    CSS = """
    Screen {
        background: $surface;
    }

    #output-container {
        height: 1fr;
        border: solid $primary;
        padding: 1 2;
        margin: 1;
    }

    #status-bar {
        dock: bottom;
        height: 1;
        background: $primary-darken-2;
        color: $text;
        padding: 0 2;
    }

    Input {
        dock: bottom;
        margin: 0 1 1 1;
    }

    .question {
        color: $accent;
        text-style: bold;
    }

    .route {
        color: $text-muted;
    }

    .answer {
        color: $success;
        text-style: bold;
    }

    .error {
        color: $error;
        text-style: bold;
    }

    .divider {
        color: $primary-lighten-1;
    }

    .item-row {
        color: $text-muted;
    }
    """

    BINDINGS = [
        ("ctrl+c", "quit", "Quit"),
        ("ctrl+d", "quit", "Quit"),
    ]

    def __init__(self, db_path: Path | None, *, debug_llm_io: bool = False) -> None:
        """Initialize the TUI app.

        Args:
            db_path: Path to SQLite database
            debug_llm_io: Enable LLM I/O debugging
        """
        super().__init__()
        self.db_path = db_path
        self.debug_llm_io = debug_llm_io
        self.graph: nx.DiGraph | None = None
        self.agent: GraphAgent | None = None

    def compose(self) -> ComposeResult:
        """Create child widgets for the TUI."""
        yield Header(show_clock=True)

        with Vertical():
            with ScrollableContainer(id="output-container"):
                yield Vertical(id="output")
            yield Static(self._status_text(), id="status-bar")
            yield Input(placeholder="Type your question here...", id="query-input")

        yield Footer()

    def on_mount(self) -> None:
        """Called when app is mounted."""
        self._append_output("[Welcome to IFC Query Agent TUI]")
        self._append_output(f"Database: {self.db_path or '(none)'}")
        self._append_output("Type a question and press Enter. Press Ctrl+C to quit.")
        self._append_output("")
        self.query_one(Input).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle input submission."""
        question = event.value.strip()
        if not question:
            return

        # Clear input
        event.input.value = ""

        # Check for exit commands
        if question.lower() in {"exit", "quit"}:
            self.exit()
            return

        # Show question immediately
        self._append_output("")
        self._append_output(f"Q: {question}", style="question")

        # Execute query in background thread worker (keeps UI responsive)
        self.run_worker(
            functools.partial(self._execute_query, question),
            exclusive=True,
            thread=True,
        )

    def _execute_query(self, question: str) -> None:
        """Execute query in a background thread worker.

        Runs in a worker thread (thread=True), so call_from_thread is safe
        and all UI updates are scheduled back onto the main event loop.

        Args:
            question: User question
        """
        # Update status to show processing
        self.call_from_thread(
            self._update_status, f"Processing: {self._truncate(question, 40)}"
        )

        try:
            # execute_query is synchronous; runs here in the thread worker
            result_bundle = execute_query(
                question,
                self.db_path,
                self.graph,
                self.agent,
                debug_llm_io=self.debug_llm_io,
            )

            # Extract components
            result: dict[str, Any] = result_bundle["result"]
            self.graph = result_bundle.get("graph") or self.graph
            self.agent = result_bundle.get("agent") or self.agent

            # Schedule UI update on the main thread
            self.call_from_thread(self._display_result, result)

        except Exception as exc:
            error_msg = f"Query execution failed: {exc}"
            self.call_from_thread(
                self._append_output, f"Error: {error_msg}", style="error"
            )

        finally:
            # Reset status bar on main thread
            self.call_from_thread(self._update_status, self._status_text())

    def _display_result(self, result: dict[str, Any]) -> None:
        """Display query result in output area (called from worker).

        Args:
            result: Query result dict
        """
        # Show route and reason
        route = result.get("route", "?")
        decision = result.get("decision", "")
        self._append_output(f"   [{route}] {decision}", style="route")

        # Check for error
        error = result.get("error")
        if error:
            self._append_output(f"Error: {error}", style="error")
            return

        # Show answer
        answer = result.get("answer", "No answer produced.")
        self._append_output(f"A: {answer}", style="answer")

        # Show list items if present (SQL route)
        if route == "sql":
            self._display_sql_items(result)

        # Show sample if present (graph route)
        if route == "graph":
            self._display_graph_sample(result)

        # Add divider
        self._append_output("-" * 60, style="divider")

    def _display_sql_items(self, result: dict[str, Any]) -> None:
        """Display SQL list items in compact format.

        Args:
            result: SQL result dict
        """
        data = result.get("data") or {}
        intent = data.get("intent", "")

        if intent != "list":
            return

        items = data.get("items")
        if not items or not isinstance(items, list):
            return

        total = data.get("total_count", len(items))
        limit = data.get("limit", len(items))
        shown = min(total, limit)

        # Show max 10 items in TUI to avoid clutter
        display_limit = min(10, shown)
        truncated_items = items[:display_limit]

        self._append_output("")
        for i, item in enumerate(truncated_items, 1):
            # Build compact row: name, class, level
            parts = []
            if item.get("name"):
                parts.append(f"Name: {self._truncate(item['name'], 25)}")
            if item.get("ifc_class"):
                parts.append(f"Class: {item['ifc_class']}")
            if item.get("level"):
                parts.append(f"Level: {self._truncate(item['level'], 15)}")

            row_text = (
                f"   {i}. " + " | ".join(parts) if parts else f"   {i}. (no data)"
            )
            self._append_output(row_text, style="item-row")

        # Show truncation message
        not_shown = total - display_limit
        if not_shown > 0:
            self._append_output(
                f"   ({not_shown} more items not shown)", style="item-row"
            )

    def _display_graph_sample(self, result: dict[str, Any]) -> None:
        """Display graph sample elements.

        Args:
            result: Graph result dict
        """
        sample = (result.get("data") or {}).get("sample")
        if not sample or not isinstance(sample, list):
            return

        self._append_output("")
        self._append_output("   Sample:", style="route")
        for item in sample[:10]:  # Limit to 10 items
            item_str = str(item)
            self._append_output(f"   - {self._truncate(item_str, 70)}", style="route")

    def _append_output(self, text: str, *, style: str = "") -> None:
        """Append text to output area.

        Args:
            text: Text to append
            style: CSS class name for styling
        """
        output = self.query_one("#output", Vertical)
        line = Static(text, classes=style, markup=False)
        output.mount(line)

        # Auto-scroll to bottom
        container = self.query_one("#output-container", ScrollableContainer)
        container.scroll_end(animate=False)

    def _update_status(self, text: str) -> None:
        """Update status bar text.

        Args:
            text: Status text
        """
        status_bar = self.query_one("#status-bar", Static)
        status_bar.update(text)

    def _status_text(self) -> str:
        """Generate default status bar text.

        Returns:
            Status bar text
        """
        db_name = self.db_path.name if self.db_path else "(no database)"
        return f"Database: {db_name} | Press Ctrl+C to quit"

    @staticmethod
    def _truncate(text: str, max_len: int) -> str:
        """Truncate text to max length with ellipsis.

        Args:
            text: Text to truncate
            max_len: Maximum length

        Returns:
            Truncated text
        """
        if len(text) <= max_len:
            return text
        if max_len <= 3:
            return text[:max_len]
        return text[: max_len - 3] + "..."


def run_tui(db_path: Path | None = None, *, debug_llm_io: bool = False) -> None:
    """Launch the Textual TUI.

    Args:
        db_path: Path to SQLite database (or None to auto-detect)
        debug_llm_io: Enable LLM I/O debugging
    """
    if db_path is None:
        db_path = find_sqlite_db()

    app = QueryApp(db_path, debug_llm_io=debug_llm_io)
    app.run()
