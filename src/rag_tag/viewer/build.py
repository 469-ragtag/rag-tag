from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from rag_tag.graph import GraphRuntime, wrap_networkx_graph
from rag_tag.parser.ifc_to_jsonl import convert_ifc_to_jsonl
from rag_tag.parser.jsonl_to_graph import (
    build_graph,
    export_webgl_graph_assets,
    plot_interactive_graph,
)
from rag_tag.parser.jsonl_to_sql import jsonl_to_sql

_DATASET_CHARS_RE = re.compile(r"[^A-Za-z0-9._-]+")
BuildProgressCallback = Callable[[str, str, int], None]
_JSONL_PROGRESS_START = 18
_JSONL_PROGRESS_END = 36


def sanitize_dataset_stem(raw_name: str) -> str:
    sanitized = _DATASET_CHARS_RE.sub("-", raw_name.strip()).strip("._-")
    return sanitized or "ifc_dataset"


@dataclass(slots=True, frozen=True)
class ViewerArtifactPaths:
    dataset: str
    output_dir: Path
    jsonl_path: Path
    db_path: Path
    debug_graph_html_path: Path
    webgl_graph_bundle_path: Path


@dataclass(slots=True, frozen=True)
class ViewerBuildArtifacts:
    dataset: str
    ifc_path: Path
    jsonl_path: Path
    db_path: Path
    debug_graph_html_path: Path
    webgl_graph_bundle_path: Path
    runtime: GraphRuntime
    node_count: int
    edge_count: int
    reused_existing_build: bool = False


def default_output_dir(project_root: Path | None = None) -> Path:
    base_dir = project_root if project_root is not None else Path.cwd()
    return (base_dir / "output").expanduser().resolve()


def _emit_progress(
    progress_callback: BuildProgressCallback | None,
    stage: str,
    message: str,
    progress: int,
) -> None:
    if progress_callback is None:
        return
    progress_callback(stage, message, progress)


def viewer_artifact_paths(output_dir: Path, dataset: str) -> ViewerArtifactPaths:
    resolved_output_dir = output_dir.expanduser().resolve()
    return ViewerArtifactPaths(
        dataset=dataset,
        output_dir=resolved_output_dir,
        jsonl_path=resolved_output_dir / f"{dataset}.jsonl",
        db_path=resolved_output_dir / f"{dataset}.db",
        debug_graph_html_path=resolved_output_dir / f"{dataset}_graph.html",
        webgl_graph_bundle_path=resolved_output_dir / f"{dataset}_graph_viewer",
    )


def viewer_artifacts_ready(paths: ViewerArtifactPaths) -> bool:
    return (
        paths.jsonl_path.is_file()
        and paths.db_path.is_file()
        and paths.debug_graph_html_path.is_file()
        and (paths.webgl_graph_bundle_path / "manifest.json").is_file()
    )


def build_viewer_artifacts_from_ifc(
    ifc_path: Path,
    *,
    output_dir: Path,
    payload_mode: str = "minimal",
    reuse_existing: bool = False,
    progress_callback: BuildProgressCallback | None = None,
) -> ViewerBuildArtifacts:
    resolved_ifc_path = ifc_path.expanduser().resolve()
    dataset = sanitize_dataset_stem(resolved_ifc_path.stem)
    resolved_output_dir = output_dir.expanduser().resolve()
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    artifact_paths = viewer_artifact_paths(resolved_output_dir, dataset)

    if reuse_existing and viewer_artifacts_ready(artifact_paths):
        _emit_progress(
            progress_callback,
            "reuse",
            f"Reusing cached viewer artifacts for {resolved_ifc_path.name}...",
            58,
        )
        graph = build_graph(
            jsonl_paths=[artifact_paths.jsonl_path],
            payload_mode=payload_mode,
        )
        runtime = wrap_networkx_graph(
            graph,
            context_db_path=artifact_paths.db_path,
            payload_mode=payload_mode,
        )
        return ViewerBuildArtifacts(
            dataset=dataset,
            ifc_path=resolved_ifc_path,
            jsonl_path=artifact_paths.jsonl_path,
            db_path=artifact_paths.db_path,
            debug_graph_html_path=artifact_paths.debug_graph_html_path,
            webgl_graph_bundle_path=artifact_paths.webgl_graph_bundle_path,
            runtime=runtime,
            node_count=int(graph.number_of_nodes()),
            edge_count=int(graph.number_of_edges()),
            reused_existing_build=True,
        )

    def emit_jsonl_progress(processed: int, total: int) -> None:
        if total <= 0:
            _emit_progress(
                progress_callback,
                "jsonl",
                f"Extracting IFC elements from {resolved_ifc_path.name}...",
                _JSONL_PROGRESS_START,
            )
            return

        bounded_ratio = max(0.0, min(1.0, processed / total))
        progress = _JSONL_PROGRESS_START + int(
            round(bounded_ratio * (_JSONL_PROGRESS_END - _JSONL_PROGRESS_START))
        )
        progress = max(_JSONL_PROGRESS_START, min(_JSONL_PROGRESS_END, progress))
        _emit_progress(
            progress_callback,
            "jsonl",
            (
                f"Extracting IFC elements from {resolved_ifc_path.name}... "
                f"({processed}/{total})"
            ),
            progress,
        )

    _emit_progress(
        progress_callback,
        "jsonl",
        f"Extracting IFC elements from {resolved_ifc_path.name}...",
        _JSONL_PROGRESS_START,
    )
    convert_ifc_to_jsonl(
        resolved_ifc_path,
        artifact_paths.jsonl_path,
        progress_callback=emit_jsonl_progress,
    )

    _emit_progress(
        progress_callback,
        "sqlite",
        "Building the SQLite dataset...",
        _JSONL_PROGRESS_END,
    )
    jsonl_to_sql(artifact_paths.jsonl_path, artifact_paths.db_path)

    _emit_progress(
        progress_callback,
        "graph",
        "Constructing the graph runtime...",
        58,
    )
    graph = build_graph(
        jsonl_paths=[artifact_paths.jsonl_path],
        payload_mode=payload_mode,
    )

    _emit_progress(
        progress_callback,
        "legacy_plot",
        "Rendering the legacy Plotly debugger...",
        76,
    )
    plot_interactive_graph(graph, artifact_paths.debug_graph_html_path)

    _emit_progress(
        progress_callback,
        "webgl",
        "Exporting WebGL graph assets...",
        90,
    )
    webgl_graph_bundle_path = export_webgl_graph_assets(graph, resolved_output_dir)

    _emit_progress(
        progress_callback,
        "runtime",
        "Finalizing the viewer runtime...",
        96,
    )
    runtime = wrap_networkx_graph(
        graph,
        context_db_path=artifact_paths.db_path,
        payload_mode=payload_mode,
    )
    return ViewerBuildArtifacts(
        dataset=dataset,
        ifc_path=resolved_ifc_path,
        jsonl_path=artifact_paths.jsonl_path,
        db_path=artifact_paths.db_path,
        debug_graph_html_path=artifact_paths.debug_graph_html_path,
        webgl_graph_bundle_path=webgl_graph_bundle_path,
        runtime=runtime,
        node_count=int(graph.number_of_nodes()),
        edge_count=int(graph.number_of_edges()),
        reused_existing_build=False,
    )
