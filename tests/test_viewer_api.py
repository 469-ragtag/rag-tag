from __future__ import annotations

import time
from pathlib import Path

import networkx as nx
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient

import rag_tag.viewer.state as viewer_state_module
from rag_tag.graph import wrap_networkx_graph
from rag_tag.viewer.app import create_app
from rag_tag.viewer.build import ViewerBuildArtifacts
from rag_tag.viewer.state import ViewerState


class _FakeViewerState(ViewerState):
    def __init__(
        self,
        runtime,
        *,
        debug_graph_html_override: Path | None = None,
        webgl_graph_bundle_override: Path | None = None,
        source_ifc_path: Path | None = None,
    ) -> None:
        super().__init__(
            db_paths=[],
            graph_dataset="demo",
            payload_mode="minimal",
            runtime=runtime,
            debug_graph_html_override=debug_graph_html_override,
            webgl_graph_bundle_override=webgl_graph_bundle_override,
            source_ifc_path=source_ifc_path,
        )

    def list_datasets(self):
        return []

    def execute_user_query(self, question: str, *, strict_sql: bool = False):
        return {
            "route": "graph",
            "decision": "fake",
            "answer": f"Echo: {question}",
            "data": {"question": question, "strict_sql": strict_sql},
        }


def _build_runtime():
    graph = nx.MultiDiGraph()
    graph.graph["datasets"] = ["demo"]
    graph.add_node(
        "Storey::L1",
        label="Level 1",
        class_="IfcBuildingStorey",
        geometry=(0.0, 0.0, 0.0),
        properties={},
    )
    graph.add_node(
        "Element::wall-a",
        label="Wall A",
        class_="IfcWall",
        geometry=(1.0, 0.0, 0.0),
        properties={
            "Name": "Wall A",
            "GlobalId": "wall-guid-a",
            "ExpressId": 101,
        },
    )
    graph.add_node(
        "Element::wall-b",
        label="Wall B",
        class_="IfcWall",
        geometry=(2.0, 0.5, 0.0),
        properties={
            "Name": "Wall B",
            "GlobalId": "wall-guid-b",
            "ExpressId": 102,
        },
    )
    graph.add_edge("Storey::L1", "Element::wall-a", relation="contains")
    graph.add_edge("Storey::L1", "Element::wall-b", relation="contains")
    graph.add_edge(
        "Element::wall-a",
        "Element::wall-b",
        relation="adjacent_to",
        source="heuristic",
    )
    return wrap_networkx_graph(graph)


def _build_runtime_for_dataset(dataset: str):
    graph = nx.MultiDiGraph()
    graph.graph["datasets"] = [dataset]
    graph.add_node(
        "Storey::L1",
        label="Level 1",
        class_="IfcBuildingStorey",
        dataset=dataset,
        geometry=(0.0, 0.0, 0.0),
        properties={},
    )
    graph.add_node(
        "Element::wall-a",
        label="Wall A",
        class_="IfcWall",
        dataset=dataset,
        geometry=(1.0, 0.0, 0.0),
        properties={
            "Name": "Wall A",
            "GlobalId": "wall-guid-a",
            "ExpressId": 101,
        },
    )
    graph.add_edge("Storey::L1", "Element::wall-a", relation="contains")
    return wrap_networkx_graph(graph)


@pytest.fixture()
def client(tmp_path: Path):
    debug_graph_html = tmp_path / "ifc_graph.html"
    debug_graph_html.write_text(
        "<html><body><h1>debug graph</h1></body></html>",
        encoding="utf-8",
    )
    bundle_dir = tmp_path / "demo_graph_viewer"
    bundle_dir.mkdir()
    (bundle_dir / "manifest.json").write_text(
        """
        {
          "dataset": "demo",
          "node_count": 3,
          "edge_count": 3,
          "files": {
            "nodes": {"path": "nodes.bin"},
            "node_meta": {"path": "nodes_meta.json"},
            "edges": {
              "hierarchy": {"path": "edges_hierarchy.bin", "stride_bytes": 12}
            },
            "overlays": {
              "bbox": {"available": false, "path": null},
              "mesh": {
                "available": false,
                "manifest_path": "overlays_mesh_manifest.json"
              }
            }
          },
          "viewer_modes": {"all": ["contains"], "hierarchy": ["contains"]},
          "legend": {"total_edges": 1, "entries": []},
          "node_groups": {},
          "relation_names": ["contains"],
          "relation_colors": {"contains": "#1d4ed8"},
          "relation_explanations": {"contains": "container to child"},
          "source_kinds": ["unknown"],
          "bounds": {"min": [0, 0, 0], "max": [1, 1, 1]},
          "render_defaults": {
            "nodes": true,
            "edges": true,
            "edge_labels": false,
            "bboxes": false,
            "meshes": false,
            "mode": "all"
          }
        }
        """.strip(),
        encoding="utf-8",
    )
    (bundle_dir / "nodes.bin").write_bytes(b"\x00" * 36)
    (bundle_dir / "nodes_meta.json").write_text("[]", encoding="utf-8")
    (bundle_dir / "edges_hierarchy.bin").write_bytes(b"\x00" * 12)
    (bundle_dir / "overlays_mesh_manifest.json").write_text(
        '{"available": false, "path": null}',
        encoding="utf-8",
    )
    source_ifc = tmp_path / "demo.ifc"
    source_ifc.write_text("ISO-10303-21;", encoding="utf-8")
    app = create_app(
        _FakeViewerState(
            _build_runtime(),
            debug_graph_html_override=debug_graph_html,
            webgl_graph_bundle_override=bundle_dir,
            source_ifc_path=source_ifc,
        )
    )
    return TestClient(app)


def test_bootstrap_returns_graph_summary_and_model_link(client: TestClient) -> None:
    response = client.get("/api/bootstrap")

    assert response.status_code == 200
    payload = response.json()
    assert payload["selected_dataset"] == "demo"
    assert payload["graph"]["node_count"] == 3
    assert payload["graph"]["edge_count"] == 3
    assert payload["debug_graph_url"] == "/debug/graph"
    assert payload["debug_graph_available"] is True
    assert payload["webgl_graph_manifest_url"] == "/debug/graph-assets/manifest"
    assert payload["webgl_graph_available"] is True
    assert payload["model_viewer_url"] == "/model"
    assert payload["model_ifc_available"] is True


def test_model_bootstrap_reports_active_ifc_source(client: TestClient) -> None:
    response = client.get("/api/model/bootstrap")

    assert response.status_code == 200
    payload = response.json()
    assert payload["selected_dataset"] == "demo"
    assert payload["graph"]["node_count"] == 3
    assert payload["model_viewer_url"] == "/model"
    assert payload["graph_viewer_url"] == "/"
    assert payload["model_ifc_available"] is True
    assert payload["model_ifc_url"] == "/api/model/ifc"
    assert payload["model_ifc_name"] == "demo.ifc"
    assert payload["model_fragments_cache_key"].startswith("fragments-v1:demo.ifc:")


def test_root_serves_graph_shell_layout(client: TestClient) -> None:
    response = client.get("/")

    assert response.status_code == 200
    assert 'id="graph3d-canvas"' in response.text
    assert 'id="toggle-legend"' in response.text
    assert 'id="query-form"' in response.text
    assert 'id="query-tool-fullscreen"' in response.text
    assert 'id="search-class-filter"' in response.text
    assert 'id="open-model-viewer"' in response.text
    assert "3D Model" in response.text
    assert 'id="graph-tool-how-to-use"' in response.text
    assert "Know More" in response.text
    assert 'id="graph-tool-legacy"' in response.text
    assert "Neighborhood" not in response.text


def test_model_route_serves_model_viewer_shell(client: TestClient) -> None:
    response = client.get("/model")

    assert response.status_code == 200
    assert 'id="model-viewport"' in response.text
    assert 'id="toggle-model-search-panel"' in response.text
    assert 'id="toggle-model-inspector-panel"' in response.text
    assert 'id="model-search-form"' in response.text
    assert 'id="model-search-results"' in response.text
    assert 'id="model-selection-details"' in response.text
    assert 'id="model-tool-fullscreen"' in response.text
    assert 'id="model-tool-reset"' in response.text
    assert 'id="model-tool-how-to-use"' in response.text
    assert 'id="model-tool-graph"' in response.text
    assert 'src="/static/model-viewer.js"' in response.text
    assert "Node Explorer" in response.text
    assert 'id="model-fit-view"' not in response.text
    assert 'id="model-clear-selection"' not in response.text
    assert 'id="model-isolate-selection"' not in response.text
    assert 'id="model-show-all"' not in response.text


def test_model_bundle_uses_local_fragments_worker_asset(client: TestClient) -> None:
    bundle_response = client.get("/static/model-viewer.js")
    worker_response = client.get("/static/vendor/thatopen/worker.mjs")

    assert bundle_response.status_code == 200
    assert "/static/vendor/thatopen/worker.mjs" in bundle_response.text
    assert worker_response.status_code == 200
    assert len(worker_response.text) > 1000


def test_legend_route_serves_explainer_page(client: TestClient) -> None:
    response = client.get("/legend")

    assert response.status_code == 200
    assert "Viewer Legend Guide" in response.text
    assert 'id="legend-mode-cards"' in response.text
    assert 'src="/static/legend.js"' in response.text


def test_how_to_use_route_serves_viewer_guide(client: TestClient) -> None:
    response = client.get("/how-to-use")

    assert response.status_code == 200
    assert "RAGTAG Viewer Guide" in response.text
    assert "How to Launch It" in response.text
    assert "Using the Graph Viewer" in response.text
    assert "Using the Query Runner" in response.text
    assert "Known Limitations" in response.text
    assert "uv run rag-tag-viewer --ifc" in response.text


def test_debug_graph_route_serves_html_artifact(client: TestClient) -> None:
    response = client.get("/debug/graph")

    assert response.status_code == 200
    assert "debug graph" in response.text


def test_model_ifc_route_serves_active_ifc_source(client: TestClient) -> None:
    response = client.get("/api/model/ifc")

    assert response.status_code == 200
    assert response.content.startswith(b"ISO-10303-21;")


def test_webgl_graph_asset_routes_serve_manifest_and_bundle_files(
    client: TestClient,
) -> None:
    manifest_response = client.get("/debug/graph-assets/manifest")
    nodes_response = client.get("/debug/graph-assets/nodes.bin")

    assert manifest_response.status_code == 200
    assert manifest_response.json()["dataset"] == "demo"
    assert nodes_response.status_code == 200
    assert len(nodes_response.content) == 36


def test_search_returns_matching_nodes(client: TestClient) -> None:
    response = client.get("/api/search", params={"q": "wall"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["total"] == 2
    assert payload["page"] == 1
    assert payload["page_size"] == 25
    assert payload["total_pages"] == 1
    assert payload["class_name"] is None
    assert payload["class_options"][0]["value"] == "IfcWall"
    assert payload["results"][0]["label"].startswith("Wall")


def test_search_supports_class_filter_and_pagination(client: TestClient) -> None:
    response = client.get(
        "/api/search",
        params={"class_name": "IfcWall", "page": 2, "page_size": 1},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["query"] == ""
    assert payload["class_name"] == "IfcWall"
    assert payload["total"] == 2
    assert payload["page"] == 2
    assert payload["page_size"] == 1
    assert payload["total_pages"] == 2
    assert payload["has_previous"] is True
    assert payload["has_next"] is False
    assert len(payload["results"]) == 1
    assert payload["results"][0]["class_name"] == "IfcWall"


def test_neighborhood_route_is_removed(client: TestClient) -> None:
    response = client.get("/api/graph/neighborhood/Element::wall-a")

    assert response.status_code == 404


def test_element_returns_properties_for_node_id(client: TestClient) -> None:
    response = client.get("/api/element/Element::wall-a")

    assert response.status_code == 200
    payload = response.json()
    assert payload["element"]["id"] == "Element::wall-a"
    assert payload["element"]["class_"] == "IfcWall"
    assert payload["element"]["properties"]["Name"] == "Wall A"


def test_element_lookup_supports_global_id_and_express_id(client: TestClient) -> None:
    by_guid = client.get("/api/element/wall-guid-b")
    by_express_id = client.get("/api/element/101")

    assert by_guid.status_code == 200
    assert by_guid.json()["element"]["id"] == "Element::wall-b"
    assert by_guid.json()["element"]["properties"]["GlobalId"] == "wall-guid-b"

    assert by_express_id.status_code == 200
    assert by_express_id.json()["element"]["id"] == "Element::wall-a"
    assert by_express_id.json()["element"]["properties"]["ExpressId"] == 101


def test_query_uses_existing_service_shape(client: TestClient) -> None:
    response = client.post("/api/query", json={"question": "What is this?"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["route"] == "graph"
    assert payload["answer"] == "Echo: What is this?"
    assert payload["presentation"]["answer"] == "Echo: What is this?"
    assert payload["presentation"]["details_json"]
    assert payload["result"]["data"]["question"] == "What is this?"


def test_bootstrap_reports_missing_webgl_bundle_when_manifest_absent(
    tmp_path: Path,
) -> None:
    debug_graph_html = tmp_path / "ifc_graph.html"
    debug_graph_html.write_text("<html></html>", encoding="utf-8")
    missing_bundle_dir = tmp_path / "missing_graph_viewer"
    source_ifc = tmp_path / "demo.ifc"
    source_ifc.write_text("ISO-10303-21;", encoding="utf-8")
    app = create_app(
        _FakeViewerState(
            _build_runtime(),
            debug_graph_html_override=debug_graph_html,
            webgl_graph_bundle_override=missing_bundle_dir,
            source_ifc_path=source_ifc,
        )
    )
    client = TestClient(app)

    bootstrap = client.get("/api/bootstrap")
    manifest = client.get("/debug/graph-assets/manifest")

    assert bootstrap.status_code == 200
    assert bootstrap.json()["webgl_graph_available"] is False
    assert bootstrap.json()["model_ifc_available"] is True
    assert manifest.status_code == 404


def test_model_bootstrap_reports_missing_ifc_when_no_source_is_available(
    tmp_path: Path,
) -> None:
    debug_graph_html = tmp_path / "ifc_graph.html"
    debug_graph_html.write_text("<html></html>", encoding="utf-8")
    bundle_dir = tmp_path / "demo_graph_viewer"
    bundle_dir.mkdir()
    (bundle_dir / "manifest.json").write_text("{}", encoding="utf-8")
    app = create_app(
        _FakeViewerState(
            _build_runtime(),
            debug_graph_html_override=debug_graph_html,
            webgl_graph_bundle_override=bundle_dir,
            source_ifc_path=None,
        )
    )
    client = TestClient(app)

    bootstrap = client.get("/api/model/bootstrap")
    ifc_response = client.get("/api/model/ifc")

    assert bootstrap.status_code == 200
    assert bootstrap.json()["model_ifc_available"] is False
    assert bootstrap.json()["model_ifc_url"] is None
    assert ifc_response.status_code == 404


def test_viewer_state_resolves_assets_from_db_output_dir_when_root_is_wrong(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    db_path = output_dir / "demo.db"
    db_path.write_text("", encoding="utf-8")
    (output_dir / "demo.jsonl").write_text('{"type": "element"}\n', encoding="utf-8")

    debug_graph_html = output_dir / "ifc_graph.html"
    debug_graph_html.write_text(
        "<html><body>debug graph</body></html>",
        encoding="utf-8",
    )

    bundle_dir = output_dir / "ifc_graph_viewer"
    bundle_dir.mkdir()
    (bundle_dir / "manifest.json").write_text("{}", encoding="utf-8")

    state = ViewerState(
        db_paths=[db_path],
        project_root=tmp_path / "missing-root",
    )

    assert state.graph_dataset == "demo"
    assert state.list_datasets()[0].name == "demo"
    assert state.debug_graph_html_path() == debug_graph_html.resolve()
    assert state.debug_graph_available() is True
    assert state.webgl_graph_bundle_path() == bundle_dir.resolve()
    assert state.webgl_graph_available() is True


def test_viewer_state_resolves_ifc_from_project_ifc_files_dir(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    output_dir = project_root / "output"
    ifc_dir = project_root / "IFC-Files"
    output_dir.mkdir(parents=True)
    ifc_dir.mkdir(parents=True)

    db_path = output_dir / "demo.db"
    db_path.write_text("", encoding="utf-8")
    (output_dir / "demo.jsonl").write_text('{"type": "element"}\n', encoding="utf-8")
    ifc_path = ifc_dir / "demo.ifc"
    ifc_path.write_text("ISO-10303-21;", encoding="utf-8")

    state = ViewerState(
        db_paths=[db_path],
        project_root=project_root,
    )

    assert state.active_ifc_path() == ifc_path.resolve()
    assert state.active_ifc_name() == "demo.ifc"
    assert state.model_ifc_available() is True


def test_uploaded_ifc_is_available_to_graph_and_model_views(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_root = tmp_path / "project"
    output_dir = project_root / "output"
    output_dir.mkdir(parents=True)

    uploaded_dataset = "Uploaded-Model"
    bundle_dir = output_dir / f"{uploaded_dataset}_graph_viewer"
    bundle_dir.mkdir()
    (bundle_dir / "manifest.json").write_text("{}", encoding="utf-8")

    debug_graph_html = output_dir / f"{uploaded_dataset}_graph.html"
    debug_graph_html.write_text(
        "<html><body>uploaded graph</body></html>",
        encoding="utf-8",
    )

    def fake_build_viewer_artifacts_from_ifc(
        ifc_path: Path,
        *,
        output_dir: Path,
        payload_mode: str = "minimal",
        reuse_existing: bool = False,
        progress_callback=None,
    ) -> ViewerBuildArtifacts:
        del payload_mode, reuse_existing, progress_callback
        output_dir.mkdir(parents=True, exist_ok=True)
        jsonl_path = output_dir / f"{uploaded_dataset}.jsonl"
        jsonl_path.write_text('{"type": "element"}\n', encoding="utf-8")
        db_path = output_dir / f"{uploaded_dataset}.db"
        db_path.write_text("", encoding="utf-8")
        bundle_dir = output_dir / f"{uploaded_dataset}_graph_viewer"
        bundle_dir.mkdir(parents=True, exist_ok=True)
        (bundle_dir / "manifest.json").write_text("{}", encoding="utf-8")
        debug_graph_html = output_dir / f"{uploaded_dataset}_graph.html"
        debug_graph_html.write_text(
            "<html><body>uploaded graph</body></html>",
            encoding="utf-8",
        )
        return ViewerBuildArtifacts(
            dataset=uploaded_dataset,
            ifc_path=ifc_path.expanduser().resolve(),
            jsonl_path=jsonl_path.resolve(),
            db_path=db_path.resolve(),
            debug_graph_html_path=debug_graph_html.resolve(),
            webgl_graph_bundle_path=bundle_dir.resolve(),
            runtime=_build_runtime_for_dataset(uploaded_dataset),
            node_count=2,
            edge_count=1,
        )

    monkeypatch.setattr(
        viewer_state_module,
        "build_viewer_artifacts_from_ifc",
        fake_build_viewer_artifacts_from_ifc,
    )

    state = ViewerState(
        db_paths=[],
        project_root=project_root,
    )

    response = state.import_uploaded_ifc("Uploaded Model.ifc", b"ISO-10303-21;")

    assert response.dataset == uploaded_dataset
    assert state.graph_dataset == uploaded_dataset
    assert state.webgl_graph_available() is True
    assert state.model_ifc_available() is True
    assert state.active_ifc_name() == "Uploaded Model.ifc"

    client = TestClient(create_app(state))

    graph_bootstrap = client.get("/api/bootstrap")
    model_bootstrap = client.get("/api/model/bootstrap")

    assert graph_bootstrap.status_code == 200
    assert model_bootstrap.status_code == 200
    assert graph_bootstrap.json()["selected_dataset"] == uploaded_dataset
    assert graph_bootstrap.json()["webgl_graph_available"] is True
    assert graph_bootstrap.json()["model_ifc_available"] is True
    assert model_bootstrap.json()["selected_dataset"] == uploaded_dataset
    assert model_bootstrap.json()["model_ifc_available"] is True
    assert model_bootstrap.json()["model_ifc_name"] == "Uploaded Model.ifc"


def test_uploaded_ifc_job_reports_intermediate_jsonl_progress(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_root = tmp_path / "project"
    output_dir = project_root / "output"
    output_dir.mkdir(parents=True)

    uploaded_dataset = "Uploaded-Model"
    bundle_dir = output_dir / f"{uploaded_dataset}_graph_viewer"
    bundle_dir.mkdir()
    (bundle_dir / "manifest.json").write_text("{}", encoding="utf-8")

    debug_graph_html = output_dir / f"{uploaded_dataset}_graph.html"
    debug_graph_html.write_text(
        "<html><body>uploaded graph</body></html>",
        encoding="utf-8",
    )

    def fake_build_viewer_artifacts_from_ifc(
        ifc_path: Path,
        *,
        output_dir: Path,
        payload_mode: str = "minimal",
        reuse_existing: bool = False,
        progress_callback=None,
    ) -> ViewerBuildArtifacts:
        del payload_mode, reuse_existing
        output_dir.mkdir(parents=True, exist_ok=True)
        jsonl_path = output_dir / f"{uploaded_dataset}.jsonl"
        jsonl_path.write_text('{"type": "element"}\n', encoding="utf-8")
        db_path = output_dir / f"{uploaded_dataset}.db"
        db_path.write_text("", encoding="utf-8")
        bundle_dir = output_dir / f"{uploaded_dataset}_graph_viewer"
        bundle_dir.mkdir(parents=True, exist_ok=True)
        (bundle_dir / "manifest.json").write_text("{}", encoding="utf-8")
        debug_graph_html = output_dir / f"{uploaded_dataset}_graph.html"
        debug_graph_html.write_text(
            "<html><body>uploaded graph</body></html>",
            encoding="utf-8",
        )
        if progress_callback is not None:
            progress_callback(
                "jsonl",
                f"Extracting IFC elements from {ifc_path.name}... (12/100)",
                20,
            )
            time.sleep(0.03)
            progress_callback(
                "jsonl",
                f"Extracting IFC elements from {ifc_path.name}... (57/100)",
                28,
            )
            time.sleep(0.03)
        return ViewerBuildArtifacts(
            dataset=uploaded_dataset,
            ifc_path=ifc_path.expanduser().resolve(),
            jsonl_path=jsonl_path.resolve(),
            db_path=db_path.resolve(),
            debug_graph_html_path=debug_graph_html.resolve(),
            webgl_graph_bundle_path=bundle_dir.resolve(),
            runtime=_build_runtime_for_dataset(uploaded_dataset),
            node_count=2,
            edge_count=1,
        )

    monkeypatch.setattr(
        viewer_state_module,
        "build_viewer_artifacts_from_ifc",
        fake_build_viewer_artifacts_from_ifc,
    )

    state = ViewerState(
        db_paths=[],
        project_root=project_root,
    )

    job = state.start_import_uploaded_ifc("Uploaded Model.ifc", b"ISO-10303-21;")
    seen_running_updates: list[tuple[int, str]] = []

    deadline = time.time() + 2.0
    final_status = state.get_import_job(job.job_id)
    while time.time() < deadline:
        final_status = state.get_import_job(job.job_id)
        if final_status.status == "running":
            seen_running_updates.append((final_status.progress, final_status.message))
        if final_status.status == "completed":
            break
        time.sleep(0.01)

    assert final_status.status == "completed"
    assert any(
        progress in {20, 28} and "/" in message
        for progress, message in seen_running_updates
    )
