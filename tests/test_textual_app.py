from __future__ import annotations

import sys
from pathlib import Path

from textual.widgets import Markdown

from rag_tag.textual_app import QueryApp


def test_display_result_uses_markdown_widget_for_answer(monkeypatch) -> None:
    app = QueryApp([Path("output/Building-Architecture.db")])
    emitted: list[tuple[str, str, bool]] = []
    mounted: list[tuple[object, bool]] = []

    def fake_append_output(
        text: str, *, style: str = "", verbose: bool = False
    ) -> None:
        emitted.append((text, style, verbose))

    def fake_mount_output_widget(widget: object, *, verbose: bool = False) -> None:
        mounted.append((widget, verbose))

    monkeypatch.setattr(app, "_append_output", fake_append_output)
    monkeypatch.setattr(app, "_mount_output_widget", fake_mount_output_widget)
    monkeypatch.setattr(app, "_append_verbose_detail", lambda result: None)
    monkeypatch.setattr(app, "_finish_query", lambda: None)

    markdown_answer = "**Beam count:** 42\n\n- East wing\n- West wing"

    app._display_result(
        {
            "route": "graph",
            "decision": "graph route",
            "answer": markdown_answer,
            "data": None,
        },
        123.0,
    )

    assert emitted[1] == ("A:", "answer", False)
    assert len(mounted) == 1
    markdown_widget = mounted[0][0]
    assert isinstance(markdown_widget, Markdown)
    assert mounted[0][1] is False
    assert markdown_widget._initial_markdown == markdown_answer


def test_display_result_keeps_full_answer_text(monkeypatch) -> None:
    app = QueryApp([Path("output/Building-Architecture.db")])
    emitted: list[tuple[str, str, bool]] = []
    mounted: list[tuple[object, bool]] = []

    def fake_append_output(
        text: str, *, style: str = "", verbose: bool = False
    ) -> None:
        emitted.append((text, style, verbose))

    def fake_mount_output_widget(widget: object, *, verbose: bool = False) -> None:
        mounted.append((widget, verbose))

    monkeypatch.setattr(app, "_append_output", fake_append_output)
    monkeypatch.setattr(app, "_mount_output_widget", fake_mount_output_widget)
    monkeypatch.setattr(app, "_append_verbose_detail", lambda result: None)
    monkeypatch.setattr(app, "_finish_query", lambda: None)

    long_answer = "The answer starts here. " + ("x" * 450) + " END"

    app._display_result(
        {
            "route": "graph",
            "decision": "graph route",
            "answer": long_answer,
            "data": None,
        },
        123.0,
    )

    assert emitted[1] == ("A:", "answer", False)
    markdown_widget = mounted[0][0]
    assert isinstance(markdown_widget, Markdown)
    initial_markdown = markdown_widget._initial_markdown
    assert initial_markdown == long_answer
    assert initial_markdown is not None
    assert initial_markdown.endswith(" END")


def test_display_result_plain_text_answer_uses_markdown_path(monkeypatch) -> None:
    app = QueryApp([Path("output/Building-Architecture.db")])
    emitted: list[tuple[str, str, bool]] = []
    mounted: list[tuple[object, bool]] = []

    def fake_append_output(
        text: str, *, style: str = "", verbose: bool = False
    ) -> None:
        emitted.append((text, style, verbose))

    def fake_mount_output_widget(widget: object, *, verbose: bool = False) -> None:
        mounted.append((widget, verbose))

    monkeypatch.setattr(app, "_append_output", fake_append_output)
    monkeypatch.setattr(app, "_mount_output_widget", fake_mount_output_widget)
    monkeypatch.setattr(app, "_append_verbose_detail", lambda result: None)
    monkeypatch.setattr(app, "_finish_query", lambda: None)

    plain_text_answer = "This is a plain text answer."

    app._display_result(
        {
            "route": "sql",
            "decision": "sql route",
            "answer": plain_text_answer,
            "data": None,
        },
        123.0,
    )

    assert emitted[1] == ("A:", "answer", False)
    markdown_widget = mounted[0][0]
    assert isinstance(markdown_widget, Markdown)
    assert markdown_widget._initial_markdown == plain_text_answer


def test_build_transcript_markdown_preserves_answer_markdown(monkeypatch) -> None:
    app = QueryApp([Path("output/Building-Architecture.db")])

    monkeypatch.setattr(app, "_mount_output_widget", lambda widget, verbose=False: None)

    app._append_output("IFC Query Agent TUI")
    app._append_output("Q: How many beams are there?", style="question")
    app._append_output("   [sql] deterministic count", style="route")
    app._append_answer_output("**Beam count:** 42\n\n- East wing\n- West wing")
    app._append_output("Warning: Using cached graph context.", style="warning")

    transcript = app._build_transcript_markdown()

    assert "IFC Query Agent TUI" in transcript
    assert "**Q:** How many beams are there?" in transcript
    assert "   [sql] deterministic count" in transcript
    assert "**A:**" in transcript
    assert "**Beam count:** 42" in transcript
    assert "- East wing" in transcript
    assert "Warning: Using cached graph context." in transcript


def test_build_transcript_markdown_includes_verbose_only_when_enabled(
    monkeypatch,
) -> None:
    app = QueryApp([Path("output/Building-Architecture.db")])

    monkeypatch.setattr(app, "_mount_output_widget", lambda widget, verbose=False: None)

    app._append_output("Q: Show details", style="question")
    app._append_answer_output("Done.")
    app._append_verbose_detail({"route": "graph", "data": {"count": 2}})

    transcript_without_verbose = app._build_transcript_markdown()
    assert "```json" not in transcript_without_verbose
    assert '"count": 2' not in transcript_without_verbose

    app.show_verbose = True
    transcript_with_verbose = app._build_transcript_markdown()
    assert "```json" in transcript_with_verbose
    assert '"count": 2' in transcript_with_verbose


def test_copy_text_to_clipboard_prefers_platform_command(monkeypatch) -> None:
    app = QueryApp([Path("output/Building-Architecture.db")])
    copied: list[str] = []

    monkeypatch.setattr(app, "_copy_with_platform_clipboard", lambda text: "pbcopy")
    monkeypatch.setattr(
        app,
        "copy_to_clipboard",
        lambda text: copied.append(text),
    )

    method = app._copy_text_to_clipboard("transcript")

    assert method == "pbcopy"
    assert copied == []


def test_copy_text_to_clipboard_falls_back_to_textual(monkeypatch) -> None:
    app = QueryApp([Path("output/Building-Architecture.db")])
    copied: list[str] = []

    monkeypatch.setattr(app, "_copy_with_platform_clipboard", lambda text: None)
    monkeypatch.setattr(
        app,
        "copy_to_clipboard",
        lambda text: copied.append(text),
    )

    method = app._copy_text_to_clipboard("transcript")

    assert method == "terminal clipboard"
    assert copied == ["transcript"]


def test_platform_clipboard_commands_match_platform(monkeypatch) -> None:
    app = QueryApp([Path("output/Building-Architecture.db")])

    monkeypatch.setattr(sys, "platform", "darwin")
    assert app._platform_clipboard_commands() == [["pbcopy"]]

    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(
        "rag_tag.textual_app.shutil.which", lambda name: name == "xclip"
    )
    assert app._platform_clipboard_commands() == [["xclip", "-selection", "clipboard"]]
