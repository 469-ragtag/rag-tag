from __future__ import annotations

from pathlib import Path

from rag_tag.textual_app import QueryApp


def test_display_result_keeps_full_answer_text(monkeypatch) -> None:
    app = QueryApp([Path("output/Building-Architecture.db")])
    emitted: list[tuple[str, str, bool]] = []

    def fake_append_output(
        text: str, *, style: str = "", verbose: bool = False
    ) -> None:
        emitted.append((text, style, verbose))

    monkeypatch.setattr(app, "_append_output", fake_append_output)
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

    assert emitted[1] == (f"A: {long_answer}", "answer", False)
    assert emitted[1][0].endswith(" END")
