from __future__ import annotations

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
