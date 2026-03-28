from __future__ import annotations

import importlib.resources
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, Response, UploadFile, status
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from rag_tag.query_presentation import build_query_presentation

from .models import (
    BootstrapResponse,
    ElementDetailResponse,
    ImportIfcJobStatusResponse,
    ModelBootstrapResponse,
    QueryRequest,
    QueryResponse,
    SearchResponse,
)
from .state import ViewerState


def _static_dir() -> Path:
    return Path(str(importlib.resources.files("rag_tag.viewer").joinpath("static")))


def _no_store_headers() -> dict[str, str]:
    return {
        "Cache-Control": "no-store, max-age=0",
        "Pragma": "no-cache",
        "Expires": "0",
    }


def create_app(state: ViewerState | None = None) -> FastAPI:
    viewer_state = state or ViewerState.default()
    app = FastAPI(
        title="RAGTAG Local Viewer",
        version="0.1.0",
        docs_url="/api/docs",
        redoc_url="/api/redoc",
    )
    app.state.viewer_state = viewer_state

    static_dir = _static_dir()
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(static_dir / "index.html", headers=_no_store_headers())

    @app.get("/model", include_in_schema=False)
    def model() -> FileResponse:
        return FileResponse(static_dir / "model.html", headers=_no_store_headers())

    @app.get("/legend", include_in_schema=False)
    def legend() -> FileResponse:
        return FileResponse(static_dir / "legend.html")

    @app.get("/how-to-use", include_in_schema=False)
    def how_to_use() -> FileResponse:
        return FileResponse(
            static_dir / "how-to-use.html",
            headers=_no_store_headers(),
        )

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/bootstrap", response_model=BootstrapResponse)
    def bootstrap(response: Response) -> BootstrapResponse:
        response.headers.update(_no_store_headers())
        state = app.state.viewer_state
        try:
            graph_summary = state.graph_summary()
        except FileNotFoundError:
            graph_summary = state.empty_graph_summary()
        return BootstrapResponse(
            datasets=state.list_datasets(),
            selected_dataset=state.graph_dataset,
            graph=graph_summary,
            debug_graph_url="/debug/graph",
            debug_graph_available=state.debug_graph_available(),
            webgl_graph_manifest_url="/debug/graph-assets/manifest",
            webgl_graph_available=state.webgl_graph_available(),
            model_viewer_url="/model",
            model_ifc_available=state.model_ifc_available(),
        )

    @app.get("/api/model/bootstrap", response_model=ModelBootstrapResponse)
    def model_bootstrap(response: Response) -> ModelBootstrapResponse:
        response.headers.update(_no_store_headers())
        state = app.state.viewer_state
        try:
            graph_summary = state.graph_summary()
        except FileNotFoundError:
            graph_summary = state.empty_graph_summary()
        model_ifc_available = state.model_ifc_available()
        return ModelBootstrapResponse(
            selected_dataset=state.graph_dataset,
            graph=graph_summary,
            model_viewer_url="/model",
            graph_viewer_url="/",
            model_ifc_available=model_ifc_available,
            model_ifc_url="/api/model/ifc" if model_ifc_available else None,
            model_ifc_name=state.active_ifc_name(),
            model_fragments_cache_key=state.active_ifc_cache_key(),
        )

    @app.get("/api/model/ifc", include_in_schema=False)
    def model_ifc_file():
        state = app.state.viewer_state
        active_ifc_path = state.active_ifc_path()
        if active_ifc_path is None or not active_ifc_path.is_file():
            raise HTTPException(
                status_code=404,
                detail=(
                    "No active IFC file is available for the 3D model viewer. "
                    "Start the viewer with --ifc or upload an IFC from the UI."
                ),
            )
        return FileResponse(
            active_ifc_path,
            media_type="application/octet-stream",
            filename=active_ifc_path.name,
            headers=_no_store_headers(),
        )

    @app.get("/debug/graph-assets/manifest", include_in_schema=False)
    def debug_graph_assets_manifest():
        state = app.state.viewer_state
        manifest_path = state.webgl_graph_manifest_path()
        if manifest_path.is_file():
            return FileResponse(
                manifest_path,
                media_type="application/json",
                headers=_no_store_headers(),
            )
        raise HTTPException(
            status_code=404,
            detail=(
                "WebGL graph manifest not found. "
                "Regenerate graph assets with "
                "`uv run rag-tag-jsonl-to-graph --jsonl-dir output --out-dir output`."
            ),
        )

    @app.get("/debug/graph-assets/{asset_path:path}", include_in_schema=False)
    def debug_graph_asset(asset_path: str):
        state = app.state.viewer_state
        try:
            resolved = state.webgl_graph_asset_path(asset_path)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        if not resolved.is_file():
            raise HTTPException(status_code=404, detail=asset_path)
        return FileResponse(resolved, headers=_no_store_headers())

    @app.get("/debug/graph", include_in_schema=False, response_model=None)
    def debug_graph():
        state = app.state.viewer_state
        graph_path = state.debug_graph_html_path()
        if graph_path.is_file():
            return FileResponse(graph_path)
        missing_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Debug Graph Missing</title>
  <style>
    body {{
      margin: 0;
      min-height: 100vh;
      display: grid;
      place-items: center;
      background: #1e1e2e;
      color: #cdd6f4;
      font-family: "Segoe UI", Tahoma, sans-serif;
    }}
    .panel {{
      max-width: 680px;
      padding: 24px 28px;
      border-radius: 16px;
      background: #181825;
      border: 1px solid #45475a;
      box-shadow: 0 12px 32px rgba(0, 0, 0, 0.28);
    }}
    code {{
      color: #89b4fa;
    }}
  </style>
</head>
<body>
  <div class="panel">
    <h1>Debug graph HTML not found</h1>
    <p>
      The viewer shell is ready, but the generated Plotly debug graph file is
      missing.
    </p>
    <p>Expected path: <code>{graph_path}</code></p>
    <p>
      Regenerate it with
      <code>uv run rag-tag-jsonl-to-graph --jsonl-dir output --out-dir output</code>.
    </p>
  </div>
</body>
</html>
"""
        return HTMLResponse(missing_html, status_code=404)

    @app.post("/api/query", response_model=QueryResponse)
    def query(request: QueryRequest) -> QueryResponse:
        state = app.state.viewer_state
        result = state.execute_user_query(
            request.question,
            strict_sql=request.strict_sql,
        )
        return QueryResponse(
            route=result.get("route"),
            decision=result.get("decision"),
            answer=result.get("answer"),
            presentation=build_query_presentation(result),
            result=result,
        )

    @app.get("/api/search", response_model=SearchResponse)
    def search(
        q: str = Query(default=""),
        class_name: str | None = Query(default=None),
        page: int = Query(default=1, ge=1),
        page_size: int = Query(default=25, ge=1, le=100),
    ) -> SearchResponse:
        state = app.state.viewer_state
        try:
            return state.search_nodes(
                q.strip() or None,
                class_name=class_name,
                page=page,
                page_size=page_size,
            )
        except FileNotFoundError:
            return SearchResponse(
                query=q.strip(),
                class_name=class_name.strip() or None if class_name else None,
                page=1,
                page_size=page_size,
                total=0,
                total_pages=0,
                has_next=False,
                has_previous=False,
                class_options=[],
                results=[],
            )

    @app.get("/api/element/{element_id}", response_model=ElementDetailResponse)
    def element(element_id: str) -> ElementDetailResponse:
        state = app.state.viewer_state
        try:
            payload = state.get_element(element_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return ElementDetailResponse(element=payload)

    @app.post(
        "/api/import-ifc",
        response_model=ImportIfcJobStatusResponse,
        status_code=status.HTTP_202_ACCEPTED,
    )
    async def import_ifc(
        ifc_file: UploadFile = File(...),
    ) -> ImportIfcJobStatusResponse:
        state = app.state.viewer_state
        try:
            payload = await ifc_file.read()
            return state.start_import_uploaded_ifc(ifc_file.filename, payload)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        finally:
            await ifc_file.close()

    @app.get(
        "/api/import-ifc/{job_id}",
        response_model=ImportIfcJobStatusResponse,
    )
    def import_ifc_status(job_id: str) -> ImportIfcJobStatusResponse:
        state = app.state.viewer_state
        try:
            return state.get_import_job(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    return app
