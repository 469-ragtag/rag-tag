from __future__ import annotations

import argparse
from pathlib import Path

from rag_tag.query_service import find_sqlite_dbs

from .app import create_app
from .state import ViewerState


def _parse_dataset(value: str) -> str:
    dataset = value.strip()
    if not dataset:
        raise argparse.ArgumentTypeError("Dataset name cannot be empty.")
    return dataset


def _parse_db_path(value: str) -> Path:
    candidate = Path(value).expanduser().resolve()
    if not candidate.is_file():
        raise argparse.ArgumentTypeError(f"SQLite database not found: {candidate}")
    return candidate


def _parse_ifc_path(value: str) -> Path:
    candidate = Path(value).expanduser().resolve()
    if not candidate.is_file():
        raise argparse.ArgumentTypeError(f"IFC file not found: {candidate}")
    if candidate.suffix.lower() != ".ifc":
        raise argparse.ArgumentTypeError(
            f"Expected an .ifc file, got: {candidate.name}"
        )
    return candidate


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the RAGTAG local viewer.")
    parser.add_argument(
        "--graph-dataset",
        type=_parse_dataset,
        default=None,
        help="Dataset stem for the graph runtime (<project>/output/<stem>.jsonl).",
    )
    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument(
        "--ifc",
        type=_parse_ifc_path,
        default=None,
        help=(
            "IFC file to ingest and build into JSONL, SQLite, Plotly HTML, and "
            "WebGL viewer assets before starting the viewer."
        ),
    )
    source_group.add_argument(
        "--db",
        type=_parse_db_path,
        action="append",
        default=None,
        help=(
            "SQLite database to expose to the viewer. "
            "May be passed more than once. Defaults to auto-detected DBs."
        ),
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host interface for the local viewer server.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for the local viewer server.",
    )
    parser.add_argument(
        "--graph-payload-mode",
        choices=("full", "minimal"),
        default="minimal",
        help="Payload mode to use for the graph runtime.",
    )
    args = parser.parse_args()

    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover - exercised manually
        raise RuntimeError(
            "uvicorn is not installed. Run `uv sync --group dev` first."
        ) from exc

    if args.ifc and args.graph_dataset:
        parser.error("--graph-dataset cannot be used with --ifc.")

    if args.ifc:
        state = ViewerState.default(
            payload_mode=args.graph_payload_mode,
            db_paths=[],
        )
        state.import_ifc(args.ifc)
    else:
        db_paths = args.db if args.db else find_sqlite_dbs()
        state = ViewerState.default(
            graph_dataset=args.graph_dataset,
            payload_mode=args.graph_payload_mode,
            db_paths=db_paths,
        )
    app = create_app(state)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
