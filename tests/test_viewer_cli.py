from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from rag_tag.viewer import cli as viewer_cli


class _FakeViewerState:
    def __init__(self) -> None:
        self.imported_ifc: Path | None = None

    def import_ifc(self, ifc_path: Path) -> None:
        self.imported_ifc = ifc_path


def test_viewer_cli_enables_trace_for_ifc_launch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    ifc_path = tmp_path / "demo.ifc"
    ifc_path.write_text("ISO-10303-21;", encoding="utf-8")

    fake_state = _FakeViewerState()
    setup_calls: list[tuple[bool, bool]] = []
    default_calls: list[tuple[str, list[Path]]] = []
    create_app_calls: list[object] = []
    uvicorn_calls: list[tuple[object, str, int, str]] = []

    def fake_setup_logfire(*, enabled: bool = False, console: bool = True):
        setup_calls.append((enabled, console))
        return None

    def fake_default(*, payload_mode: str, db_paths: list[Path], graph_dataset=None):
        del graph_dataset
        default_calls.append((payload_mode, db_paths))
        return fake_state

    def fake_create_app(state: object) -> str:
        create_app_calls.append(state)
        return "fake-app"

    def fake_uvicorn_run(app: object, *, host: str, port: int, log_level: str) -> None:
        uvicorn_calls.append((app, host, port, log_level))

    monkeypatch.setattr(viewer_cli, "setup_logfire", fake_setup_logfire)
    monkeypatch.setattr(viewer_cli.ViewerState, "default", fake_default)
    monkeypatch.setattr(viewer_cli, "create_app", fake_create_app)
    monkeypatch.setitem(
        sys.modules,
        "uvicorn",
        types.SimpleNamespace(run=fake_uvicorn_run),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "rag-tag-viewer",
            "--ifc",
            str(ifc_path),
            "--trace",
        ],
    )

    assert viewer_cli.main() == 0
    assert setup_calls == [(True, True)]
    assert default_calls == [("minimal", [])]
    assert fake_state.imported_ifc == ifc_path.resolve()
    assert create_app_calls == [fake_state]
    assert uvicorn_calls == [("fake-app", "127.0.0.1", 8000, "info")]


def test_viewer_cli_leaves_trace_disabled_by_default(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "demo.db"
    db_path.write_bytes(b"")

    fake_state = _FakeViewerState()
    setup_calls: list[tuple[bool, bool]] = []

    def fake_setup_logfire(*, enabled: bool = False, console: bool = True):
        setup_calls.append((enabled, console))
        return None

    monkeypatch.setattr(viewer_cli, "setup_logfire", fake_setup_logfire)
    monkeypatch.setattr(
        viewer_cli.ViewerState,
        "default",
        lambda *, graph_dataset, payload_mode, db_paths: fake_state,
    )
    monkeypatch.setattr(viewer_cli, "create_app", lambda state: state)
    monkeypatch.setitem(
        sys.modules,
        "uvicorn",
        types.SimpleNamespace(run=lambda *args, **kwargs: None),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "rag-tag-viewer",
            "--db",
            str(db_path),
        ],
    )

    assert viewer_cli.main() == 0
    assert setup_calls == [(False, True)]
