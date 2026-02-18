"""Textual TUI for the IFC query agent."""

from __future__ import annotations

import functools
import json
import time
from pathlib import Path
from typing import Any

import networkx as nx
from textual.app import App, ComposeResult
from textual.containers import ScrollableContainer, Vertical
from textual.reactive import reactive
from textual.theme import Theme
from textual.widgets import Footer, Header, Input, Static
from textual.worker import Worker

from rag_tag.agent import GraphAgent
from rag_tag.query_service import execute_query, find_sqlite_db

# Maximum Static widgets kept in the output area before oldest are pruned.
_HISTORY_MAX = 200

# Maximum JSON lines shown in verbose detail per answer.
_VERBOSE_MAX_LINES = 60

# Maximum list items shown per SQL Q/A block.
_LIST_DISPLAY_LIMIT = 10

# Maximum sample items shown per graph Q/A block.
_SAMPLE_DISPLAY_LIMIT = 10

# ---------------------------------------------------------------------------
# Catppuccin Mocha theme (https://catppuccin.com/palette)
# ---------------------------------------------------------------------------
_CATPPUCCIN_MOCHA = Theme(
    name="catppuccin-mocha",
    primary="#89b4fa",  # Blue
    secondary="#cba6f7",  # Mauve
    accent="#fab387",  # Peach
    foreground="#cdd6f4",  # Text
    background="#1e1e2e",  # Base
    surface="#313244",  # Surface0
    panel="#181825",  # Mantle
    success="#a6e3a1",  # Green
    warning="#f9e2af",  # Yellow
    error="#f38ba8",  # Red
    dark=True,
)


class QueryApp(App[None]):
    """A minimal Textual TUI for querying IFC data."""

    CSS = """
    Screen {
        background: $background;
    }

    #output-container {
        height: 1fr;
        border: solid $primary;
        padding: 1 2;
        margin: 1;
    }

    /* height: auto lets the Vertical grow with its children so the
       ScrollableContainer has real content to scroll over. */
    #output {
        height: auto;
    }

    #status-bar {
        dock: bottom;
        height: 1;
        background: $panel;
        color: $foreground;
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
        color: $foreground;
        text-style: dim;
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
        color: $primary;
        text-style: dim;
    }

    .item-row {
        color: $foreground;
        text-style: dim;
    }

    /* Verbose JSON detail: hidden by default, shown when .visible is added. */
    .verbose-detail {
        display: none;
        color: $foreground;
        text-style: dim;
    }

    .verbose-detail.visible {
        display: block;
    }
    """

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("ctrl+c", "quit", "Quit"),
        ("ctrl+d", "quit", "Quit"),
        ("ctrl+l", "clear_output", "Clear"),
        ("v", "toggle_verbose", "Toggle details"),
        # Page Up / Page Down scroll the output area even while the Input
        # widget has focus (Input does not consume these keys).
        ("page_up", "scroll_page_up", "Scroll up"),
        ("page_down", "scroll_page_down", "Scroll down"),
    ]

    # Reactive flag: when True, verbose-detail widgets carry the .visible class.
    show_verbose: reactive[bool] = reactive(False)

    def __init__(self, db_path: Path | None, *, debug_llm_io: bool = False) -> None:
        """Initialize the TUI app.

        Args:
            db_path: Path to SQLite database (or None to auto-detect).
            debug_llm_io: Enable LLM I/O debugging. Suppressed in TUI mode
                because stderr output would corrupt the Textual display; a
                warning is shown in the welcome banner instead.
        """
        super().__init__()
        self.db_path = db_path
        # --input / debug_llm_io would write to stderr and corrupt the TUI.
        # Suppress it and record that we did so we can warn the user.
        self._input_flag_ignored: bool = bool(debug_llm_io)
        self.debug_llm_io: bool = False
        self.graph: nx.DiGraph | None = None
        self.agent: GraphAgent | None = None
        self._last_route: str = ""
        self._last_duration_ms: float = 0.0
        # Reference to the "working..." placeholder widget for the active query.
        self._working_widget: Static | None = None
        # Handle to the active background worker (None when idle).
        self._worker: Worker | None = None

    def compose(self) -> ComposeResult:
        """Build the widget tree."""
        yield Header(show_clock=True)

        with Vertical():
            with ScrollableContainer(id="output-container"):
                yield Vertical(id="output")
            yield Static(self._status_text(), id="status-bar")
            yield Input(placeholder="Type your question here...", id="query-input")

        yield Footer()

    def on_load(self) -> None:
        """Register the Catppuccin Mocha theme before the UI is composed."""
        self.register_theme(_CATPPUCCIN_MOCHA)
        self.theme = "catppuccin-mocha"

    def on_mount(self) -> None:
        """Populate the welcome banner and focus the input."""
        self._append_output("IFC Query Agent TUI")
        self._append_output(f"Database: {self.db_path or '(none)'}")
        self._append_output("Type a question and press Enter.")
        self._append_output(
            "Keys: q/ctrl+d=quit  ctrl+l=clear  v=toggle JSON details"
            "  PgUp/PgDn=scroll  mouse-wheel=scroll"
        )
        if self._input_flag_ignored:
            self._append_output(
                "Note: --input flag ignored in TUI mode"
                " (stderr output would corrupt the display).",
                style="route",
            )
        self._append_output("")
        self.query_one(Input).focus()

    # ------------------------------------------------------------------ actions

    async def action_quit(self) -> None:
        """Cancel any running worker then exit the application."""
        if self._worker is not None and not self._worker.is_finished:
            self._worker.cancel()
        self.exit()

    def on_unmount(self) -> None:
        """Cancel any running worker when the app tears down."""
        if self._worker is not None and not self._worker.is_finished:
            self._worker.cancel()

    def action_clear_output(self) -> None:
        """Remove all lines from the output area."""
        output = self.query_one("#output", Vertical)
        for child in list(output.children):
            child.remove()
        # Working widget was a child of output; it no longer exists.
        self._working_widget = None
        self._append_output("[Output cleared]")

    def action_toggle_verbose(self) -> None:
        """Toggle visibility of verbose JSON detail for all Q/A blocks."""
        self.show_verbose = not self.show_verbose
        for widget in self.query(".verbose-detail"):
            if self.show_verbose:
                widget.add_class("visible")
            else:
                widget.remove_class("visible")
        self._update_status(self._status_text())

    def action_scroll_page_up(self) -> None:
        """Scroll the output area up one page (works while Input has focus)."""
        self.query_one("#output-container", ScrollableContainer).scroll_page_up(
            animate=True
        )

    def action_scroll_page_down(self) -> None:
        """Scroll the output area down one page (works while Input has focus)."""
        self.query_one("#output-container", ScrollableContainer).scroll_page_down(
            animate=True
        )

    # --------------------------------------------------------------- input flow

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle a submitted question."""
        question = event.value.strip()
        if not question:
            return

        event.input.value = ""

        if question.lower() in {"exit", "quit"}:
            self.exit()
            return

        # Disable the input while the worker runs to prevent double submission.
        event.input.disabled = True

        # Append the question line immediately.
        self._append_output("")
        self._append_output(f"Q: {question}", style="question")

        # Placeholder that will be replaced with [route] reason once done.
        self._working_widget = Static("   working...", classes="route", markup=False)
        output = self.query_one("#output", Vertical)
        output.mount(self._working_widget)
        self.query_one("#output-container", ScrollableContainer).scroll_end(
            animate=False
        )

        # Run the (blocking) query in a background thread to keep UI live.
        # Store the handle so it can be cancelled on quit or unmount.
        self._worker = self.run_worker(
            functools.partial(self._execute_query, question),
            exclusive=True,
            thread=True,
        )

    # ------------------------------------------------------------------ worker

    def _execute_query(self, question: str) -> None:
        """Execute a query in the worker thread.

        All UI mutations are marshalled back to the main event loop via
        call_from_thread, as required by Textual's thread-safety contract.
        Cancellation is checked after any long operation so that a quit
        or a new query does not produce stale UI updates.
        """
        # Capture worker reference once; self._worker may be reassigned later.
        worker = self._worker
        t0 = time.monotonic()

        # Bail immediately if already cancelled (e.g. user pressed quit before
        # the thread even started).
        if worker is not None and worker.is_cancelled:
            return

        try:
            self.call_from_thread(
                self._update_status, f"Processing: {self._truncate(question, 40)}"
            )
        except Exception:
            # App may have shut down between submit and first update.
            return

        try:
            result_bundle = execute_query(
                question,
                self.db_path,
                self.graph,
                self.agent,
                debug_llm_io=self.debug_llm_io,
            )

            result: dict[str, Any] = result_bundle["result"]
            self.graph = result_bundle.get("graph") or self.graph
            self.agent = result_bundle.get("agent") or self.agent

            duration_ms = (time.monotonic() - t0) * 1000.0

            # Re-check after the (potentially slow) LLM call.
            if worker is not None and worker.is_cancelled:
                return

            try:
                self.call_from_thread(self._display_result, result, duration_ms)
            except Exception:
                pass  # App shut down while query was executing.

        except Exception as exc:
            duration_ms = (time.monotonic() - t0) * 1000.0
            if worker is not None and worker.is_cancelled:
                return
            try:
                self.call_from_thread(
                    self._on_query_error,
                    f"Query execution failed: {exc}",
                    duration_ms,
                )
            except Exception:
                pass

    # ----------------------------------------------------------------- display

    def _display_result(self, result: dict[str, Any], duration_ms: float) -> None:
        """Render a completed result.  Always runs on the main event loop."""
        route = result.get("route", "?")
        decision = result.get("decision", "")

        self._last_route = route
        self._last_duration_ms = duration_ms

        # Replace the "working..." placeholder with the [route] + reason line.
        route_line = f"   [{route}] {self._truncate(decision, 80)}"
        if self._working_widget is not None:
            self._working_widget.update(route_line)
            self._working_widget = None
        else:
            self._append_output(route_line, style="route")

        # Show error or answer.
        error = result.get("error")
        if error:
            self._append_output(
                f"Error: {self._truncate(str(error), 120)}", style="error"
            )
        else:
            answer = result.get("answer") or "No answer produced."
            self._append_output(
                f"A: {self._truncate(str(answer), 400)}", style="answer"
            )
            if route == "sql":
                self._display_sql_items(result)
            if route == "graph":
                self._display_graph_sample(result)

        # Verbose JSON block (hidden by default; toggled with v).
        self._append_verbose_detail(result)

        self._append_output("-" * 60, style="divider")
        self._update_status(self._status_text())
        self._finalize_input()

    def _on_query_error(self, error_msg: str, duration_ms: float) -> None:
        """Handle a top-level worker exception on the main event loop."""
        self._last_duration_ms = duration_ms
        if self._working_widget is not None:
            # Switch from .route to .error so the placeholder looks like an error.
            self._working_widget.set_classes("error")
            self._working_widget.update(f"   Error: {self._truncate(error_msg, 80)}")
            self._working_widget = None
        else:
            self._append_output(f"Error: {error_msg}", style="error")
        self._update_status(self._status_text())
        self._finalize_input()

    def _finalize_input(self) -> None:
        """Re-enable the input field and return focus to it."""
        try:
            inp = self.query_one(Input)
            inp.disabled = False
            inp.focus()
        except Exception:
            pass

    # ---------------------------------------------------------- result helpers

    def _display_sql_items(self, result: dict[str, Any]) -> None:
        """Show SQL list items in compact rows (capped at _LIST_DISPLAY_LIMIT)."""
        data = result.get("data") or {}
        if data.get("intent") != "list":
            return

        items = data.get("items")
        if not items or not isinstance(items, list):
            return

        total = data.get("total_count", len(items))
        display_limit = min(_LIST_DISPLAY_LIMIT, len(items))

        self._append_output("")
        for i, item in enumerate(items[:display_limit], 1):
            parts: list[str] = []
            if item.get("name"):
                parts.append(f"Name: {self._truncate(item['name'], 25)}")
            if item.get("ifc_class"):
                parts.append(f"Class: {item['ifc_class']}")
            if item.get("level"):
                parts.append(f"Level: {self._truncate(item['level'], 15)}")
            row = f"   {i}. " + " | ".join(parts) if parts else f"   {i}. (no data)"
            self._append_output(row, style="item-row")

        not_shown = total - display_limit
        if not_shown > 0:
            self._append_output(
                f"   ({not_shown} more items not shown)", style="item-row"
            )

    def _display_graph_sample(self, result: dict[str, Any]) -> None:
        """Show a short sample from a graph result."""
        sample = (result.get("data") or {}).get("sample")
        if not sample or not isinstance(sample, list):
            return

        self._append_output("")
        self._append_output("   Sample:", style="route")
        for item in sample[:_SAMPLE_DISPLAY_LIMIT]:
            self._append_output(f"   - {self._truncate(str(item), 70)}", style="route")

    def _append_verbose_detail(self, result: dict[str, Any]) -> None:
        """Emit the full result as indented JSON lines (verbose mode only)."""
        try:
            lines = json.dumps(result, indent=2, default=str).splitlines()
        except Exception:
            return

        if len(lines) > _VERBOSE_MAX_LINES:
            lines = lines[:_VERBOSE_MAX_LINES] + ["  ... (truncated)"]

        for line in lines:
            self._append_output(line, verbose=True)

    # ----------------------------------------------------------- output / status

    def _append_output(
        self, text: str, *, style: str = "", verbose: bool = False
    ) -> None:
        """Mount a plain-text Static line, pruning history when over _HISTORY_MAX.

        Args:
            text: Plain text (never interpreted as markup).
            style: CSS class(es) to apply.
            verbose: When True, the widget also gets the verbose-detail class.
                Verbose-detail widgets are excluded from the history cap so that
                hidden JSON lines do not displace visible Q/A content.
        """
        output = self.query_one("#output", Vertical)

        # Keep memory bounded.  Verbose-detail widgets (hidden by default) must
        # not count toward the cap or they would silently crowd out visible lines.
        if not verbose:
            children = list(output.children)
            non_verbose = [c for c in children if "verbose-detail" not in c.classes]
            if len(non_verbose) >= _HISTORY_MAX:
                remove_count = len(non_verbose) - _HISTORY_MAX + 1
                for child in non_verbose[:remove_count]:
                    child.remove()

        # Build the CSS class string.
        classes = style
        if verbose:
            classes = (f"{style} verbose-detail").strip()
            if self.show_verbose:
                classes += " visible"

        line = Static(text, classes=classes, markup=False)
        output.mount(line)

        self.query_one("#output-container", ScrollableContainer).scroll_end(
            animate=False
        )

    def _update_status(self, text: str) -> None:
        """Replace the status-bar content."""
        self.query_one("#status-bar", Static).update(text)

    def _status_text(self) -> str:
        """Build the default status bar string.

        Format: DB: <name> [| Route: <r>] [| <N>ms] | details:<on|off>
        """
        db_name = self.db_path.name if self.db_path else "(no database)"
        parts: list[str] = [f"DB: {db_name}"]
        if self._last_route:
            parts.append(f"Route: {self._last_route}")
        if self._last_duration_ms > 0:
            parts.append(f"{self._last_duration_ms:.0f}ms")
        parts.append("details:on" if self.show_verbose else "details:off")
        return " | ".join(parts)

    # ---------------------------------------------------------------- utilities

    @staticmethod
    def _truncate(text: str, max_len: int) -> str:
        """Return text truncated to max_len characters (ASCII-safe, no Unicode)."""
        if len(text) <= max_len:
            return text
        if max_len <= 3:
            return text[:max_len]
        return text[: max_len - 3] + "..."


def run_tui(db_path: Path | None = None, *, debug_llm_io: bool = False) -> None:
    """Launch the Textual TUI.

    Args:
        db_path: Path to SQLite database (or None to auto-detect).
        debug_llm_io: Pass --input flag through to router and agent.
    """
    if db_path is None:
        db_path = find_sqlite_db()

    app = QueryApp(db_path, debug_llm_io=debug_llm_io)
    app.run()
