from __future__ import annotations

import shutil
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock, Thread
from typing import Any
from uuid import uuid4

from rapidfuzz import fuzz

from rag_tag.graph import (
    GraphRuntime,
    ensure_graph_runtime,
    get_networkx_graph,
    wrap_networkx_graph,
)
from rag_tag.graph.payloads import INTERNAL_PAYLOAD_MODE, build_node_payload
from rag_tag.ifc_graph_tool import query_ifc_graph
from rag_tag.paths import find_project_root
from rag_tag.query_service import execute_query, find_sqlite_dbs

from .build import (
    BuildProgressCallback,
    ViewerBuildArtifacts,
    build_viewer_artifacts_from_ifc,
    default_output_dir,
    sanitize_dataset_stem,
)
from .models import (
    DatasetInfo,
    GraphSummaryResponse,
    ImportIfcJobStatusResponse,
    ImportIfcResponse,
    NodeSummary,
    SearchClassOption,
    SearchResponse,
)


def _resolve_context_db(
    db_paths: list[Path],
    graph_dataset: str | None,
) -> Path | None:
    if len(db_paths) == 1:
        return db_paths[0]
    if graph_dataset:
        for path in db_paths:
            if path.stem == graph_dataset:
                return path
    return None


def _resolve_graph_dataset(
    db_paths: list[Path],
    graph_dataset: str | None,
) -> str | None:
    if graph_dataset:
        return graph_dataset
    if len(db_paths) == 1:
        return db_paths[0].stem
    return None


def _dedupe_paths(paths: list[Path]) -> list[Path]:
    deduped: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        resolved = path.expanduser().resolve()
        key = str(resolved).lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(resolved)
    return deduped


def _node_geometry(node_data: dict[str, object]) -> tuple[float, float, float] | None:
    geometry = node_data.get("geometry")
    if not isinstance(geometry, (list, tuple)) or len(geometry) < 3:
        return None
    try:
        return (
            float(geometry[0]),
            float(geometry[1]),
            float(geometry[2]),
        )
    except (TypeError, ValueError):
        return None


def _node_summary(node_id: str, node_data: dict[str, object]) -> NodeSummary:
    label = str(node_data.get("label") or node_id)
    class_name = node_data.get("class_")
    dataset = node_data.get("dataset")
    return NodeSummary(
        id=node_id,
        label=label,
        class_name=str(class_name) if class_name is not None else None,
        dataset=str(dataset) if isinstance(dataset, str) else None,
        geometry=_node_geometry(node_data),
    )


@dataclass(slots=True, frozen=True)
class _SearchNodeRecord:
    summary: NodeSummary
    normalized_id: str
    normalized_label: str
    normalized_class_name: str
    sort_key: tuple[str, str]


@dataclass(slots=True, frozen=True)
class _SearchIndex:
    records: tuple[_SearchNodeRecord, ...]
    records_by_class: dict[str, tuple[_SearchNodeRecord, ...]]
    class_options: list[SearchClassOption]


@dataclass(slots=True)
class _ImportIfcJob:
    job_id: str
    source_name: str
    status: str = "queued"
    stage: str = "queued"
    message: str = "Queued IFC import."
    progress: int = 0
    dataset: str | None = None
    graph: GraphSummaryResponse | None = None
    debug_graph_available: bool = False
    webgl_graph_available: bool = False
    error: str | None = None


@dataclass(slots=True)
class ViewerState:
    db_paths: list[Path]
    graph_dataset: str | None = None
    payload_mode: str = "minimal"
    runtime: GraphRuntime | None = None
    agent: Any = None
    context_db: Path | None = None
    project_root: Path | None = None
    debug_graph_html_override: Path | None = None
    webgl_graph_bundle_override: Path | None = None
    source_ifc_path: Path | None = None
    _search_index: _SearchIndex | None = field(default=None, init=False, repr=False)
    _build_lock: Lock = field(default_factory=Lock, init=False, repr=False)
    _import_jobs: dict[str, _ImportIfcJob] = field(
        default_factory=dict, init=False, repr=False
    )
    _import_jobs_lock: Lock = field(default_factory=Lock, init=False, repr=False)

    def __post_init__(self) -> None:
        self.db_paths = _dedupe_paths(self.db_paths)
        self.graph_dataset = _resolve_graph_dataset(self.db_paths, self.graph_dataset)
        if self.context_db is None:
            self.context_db = _resolve_context_db(self.db_paths, self.graph_dataset)
        if self.project_root is None:
            self.project_root = find_project_root(Path(__file__).resolve().parent)
        elif not isinstance(self.project_root, Path):
            self.project_root = Path(self.project_root)
        if self.project_root is not None:
            self.project_root = self.project_root.expanduser().resolve()
        if self.debug_graph_html_override is not None:
            self.debug_graph_html_override = (
                self.debug_graph_html_override.expanduser().resolve()
            )
        if self.webgl_graph_bundle_override is not None:
            self.webgl_graph_bundle_override = (
                self.webgl_graph_bundle_override.expanduser().resolve()
            )
        if self.source_ifc_path is not None:
            self.source_ifc_path = self.source_ifc_path.expanduser().resolve()

    @classmethod
    def default(
        cls,
        *,
        graph_dataset: str | None = None,
        payload_mode: str = "minimal",
        db_paths: list[Path] | None = None,
    ) -> ViewerState:
        resolved_db_paths = db_paths if db_paths is not None else find_sqlite_dbs()
        return cls(
            db_paths=resolved_db_paths,
            graph_dataset=graph_dataset,
            payload_mode=payload_mode,
        )

    def _primary_output_dir(self) -> Path:
        if self.project_root is not None:
            return default_output_dir(self.project_root)
        candidates = self._output_dir_candidates()
        if candidates:
            return candidates[0]
        return default_output_dir()

    def _output_dir_candidates(self) -> list[Path]:
        candidates: list[Path] = []
        if self.project_root is not None:
            candidates.append(self.project_root / "output")

        for db_path in self.db_paths:
            parent = db_path.parent
            if parent.name.lower() == "output":
                candidates.append(parent)
            elif parent.name.lower() == "db":
                candidates.append(parent.parent / "output")
            else:
                candidates.append(parent / "output")

        if self.project_root is None:
            candidates.append(Path.cwd() / "output")
        return _dedupe_paths(candidates)

    def _selected_output_dir(self) -> Path:
        output_dirs = self._output_dir_candidates()
        if self.graph_dataset:
            for output_dir in output_dirs:
                candidate = output_dir / f"{self.graph_dataset}.jsonl"
                if candidate.is_file():
                    return output_dir
        for output_dir in output_dirs:
            if any(output_dir.glob("*.jsonl")):
                return output_dir
        return self._primary_output_dir()

    def _discover_graph_datasets(self) -> list[str]:
        output_dir = self._selected_output_dir()
        if not output_dir.is_dir():
            return []
        return sorted(path.stem for path in output_dir.glob("*.jsonl"))

    def _resolved_jsonl_paths(self) -> list[Path]:
        output_dirs = self._output_dir_candidates()
        if self.graph_dataset:
            for output_dir in output_dirs:
                candidate = output_dir / f"{self.graph_dataset}.jsonl"
                if candidate.is_file():
                    return [candidate]
            return [self._primary_output_dir() / f"{self.graph_dataset}.jsonl"]

        for output_dir in output_dirs:
            jsonl_paths = sorted(output_dir.glob("*.jsonl"))
            if jsonl_paths:
                return jsonl_paths
        return []

    def _load_runtime_from_output_dir(self) -> GraphRuntime:
        from rag_tag.parser.jsonl_to_graph import build_graph

        jsonl_paths = self._resolved_jsonl_paths()
        if not jsonl_paths:
            raise FileNotFoundError(
                "No viewer graph data found in "
                f"{self._selected_output_dir()}. Upload an IFC or generate JSONL "
                "graph data first."
            )

        graph = build_graph(
            jsonl_paths=jsonl_paths,
            payload_mode=self.payload_mode,
        )
        return wrap_networkx_graph(
            graph,
            context_db_path=self.context_db,
            payload_mode=self.payload_mode,
        )

    def list_datasets(self) -> list[DatasetInfo]:
        db_by_stem = {path.stem: path for path in self.db_paths}
        dataset_names = self._discover_graph_datasets()
        active_dataset = self.graph_dataset
        if active_dataset is None and len(dataset_names) == 1:
            active_dataset = dataset_names[0]
        return [
            DatasetInfo(
                name=name,
                db_path=str(db_by_stem[name]) if name in db_by_stem else None,
                selected=name == active_dataset,
            )
            for name in dataset_names
        ]

    def ensure_runtime(self) -> GraphRuntime:
        if self.runtime is None:
            self.runtime = self._load_runtime_from_output_dir()
        else:
            self.runtime = ensure_graph_runtime(
                self.runtime,
                graph_dataset=self.graph_dataset,
                context_db_path=self.context_db,
                payload_mode=self.payload_mode,
            )
        return self.runtime

    def graph_summary(self) -> GraphSummaryResponse:
        runtime = self.ensure_runtime()
        graph = get_networkx_graph(runtime)
        return GraphSummaryResponse(
            dataset=self.graph_dataset,
            datasets=list(runtime.selected_datasets),
            node_count=int(graph.number_of_nodes()),
            edge_count=int(graph.number_of_edges()),
        )

    def empty_graph_summary(self) -> GraphSummaryResponse:
        return GraphSummaryResponse(
            dataset=self.graph_dataset,
            datasets=[],
            node_count=0,
            edge_count=0,
        )

    def debug_graph_html_path(self) -> Path:
        if self.debug_graph_html_override is not None:
            return self.debug_graph_html_override
        output_dirs = self._output_dir_candidates()
        if self.graph_dataset:
            for output_dir in output_dirs:
                dataset_candidate = output_dir / f"{self.graph_dataset}_graph.html"
                if dataset_candidate.is_file():
                    return dataset_candidate
        for output_dir in output_dirs:
            candidate = output_dir / "ifc_graph.html"
            if candidate.is_file():
                return candidate
        primary_output_dir = self._primary_output_dir()
        if self.graph_dataset:
            return primary_output_dir / f"{self.graph_dataset}_graph.html"
        return primary_output_dir / "ifc_graph.html"

    def debug_graph_available(self) -> bool:
        return self.debug_graph_html_path().is_file()

    def webgl_graph_bundle_path(self) -> Path:
        if self.webgl_graph_bundle_override is not None:
            return self.webgl_graph_bundle_override
        candidates: list[Path] = []
        output_dirs = self._output_dir_candidates()
        if self.graph_dataset:
            candidates.extend(
                output_dir / f"{self.graph_dataset}_graph_viewer"
                for output_dir in output_dirs
            )
        candidates.extend(output_dir / "ifc_graph_viewer" for output_dir in output_dirs)
        for candidate in candidates:
            if (candidate / "manifest.json").is_file():
                return candidate
        return candidates[0]

    def webgl_graph_manifest_path(self) -> Path:
        return self.webgl_graph_bundle_path() / "manifest.json"

    def webgl_graph_available(self) -> bool:
        return self.webgl_graph_manifest_path().is_file()

    def webgl_graph_asset_path(self, relative_path: str) -> Path:
        bundle_dir = self.webgl_graph_bundle_path().resolve()
        asset_path = (bundle_dir / relative_path).resolve()
        try:
            asset_path.relative_to(bundle_dir)
        except ValueError as exc:
            raise FileNotFoundError(relative_path) from exc
        return asset_path

    def _ifc_dir_candidates(self) -> list[Path]:
        candidates: list[Path] = []
        if self.project_root is not None:
            candidates.append(self.project_root / "IFC-Files")
        candidates.append(Path.cwd() / "IFC-Files")
        return _dedupe_paths(candidates)

    def _active_dataset_stems(self) -> list[str]:
        if self.graph_dataset:
            return [self.graph_dataset]
        if len(self.db_paths) == 1:
            return [self.db_paths[0].stem]
        discovered = self._discover_graph_datasets()
        if len(discovered) == 1:
            return [discovered[0]]
        return []

    def active_ifc_path(self) -> Path | None:
        if self.source_ifc_path is not None:
            resolved = self.source_ifc_path.expanduser().resolve()
            if resolved.is_file() and resolved.suffix.lower() == ".ifc":
                return resolved

        for dataset in self._active_dataset_stems():
            for ifc_dir in self._ifc_dir_candidates():
                candidate = (ifc_dir / f"{dataset}.ifc").expanduser().resolve()
                if candidate.is_file():
                    return candidate
        return None

    def active_ifc_name(self) -> str | None:
        active_ifc_path = self.active_ifc_path()
        return active_ifc_path.name if active_ifc_path is not None else None

    def active_ifc_cache_key(self) -> str | None:
        active_ifc_path = self.active_ifc_path()
        if active_ifc_path is None:
            return None
        try:
            stat_result = active_ifc_path.stat()
        except OSError:
            return None
        return (
            f"fragments-v1:{active_ifc_path.name}:"
            f"{stat_result.st_size}:{stat_result.st_mtime_ns}"
        )

    def model_ifc_available(self) -> bool:
        return self.active_ifc_path() is not None

    def _viewer_upload_dir(self) -> Path:
        return self._primary_output_dir() / "_viewer_uploads"

    def _is_viewer_upload_path(self, path: Path | None) -> bool:
        if path is None:
            return False
        upload_dir = self._viewer_upload_dir().resolve()
        try:
            path.expanduser().resolve().relative_to(upload_dir)
        except ValueError:
            return False
        return True

    def _remove_output_artifact(self, path: Path) -> None:
        output_dir = self._primary_output_dir().resolve()
        resolved = path.expanduser().resolve()
        try:
            resolved.relative_to(output_dir)
        except ValueError:
            return
        if resolved.is_dir():
            shutil.rmtree(resolved, ignore_errors=True)
            return
        resolved.unlink(missing_ok=True)

    def _cleanup_previous_viewer_uploads(
        self,
        *,
        preserve_upload_path: Path | None,
        preserve_dataset: str,
    ) -> None:
        upload_dir = self._viewer_upload_dir()
        managed_datasets: set[str] = set()
        preserved_upload = (
            preserve_upload_path.expanduser().resolve()
            if preserve_upload_path is not None
            else None
        )

        if upload_dir.is_dir():
            for upload_path in upload_dir.glob("*.ifc"):
                resolved_upload = upload_path.expanduser().resolve()
                managed_datasets.add(sanitize_dataset_stem(upload_path.stem))
                if preserved_upload is not None and resolved_upload == preserved_upload:
                    continue
                resolved_upload.unlink(missing_ok=True)

        managed_datasets.discard(preserve_dataset)
        output_dir = self._primary_output_dir()
        for dataset in sorted(managed_datasets):
            for artifact_path in (
                output_dir / f"{dataset}.jsonl",
                output_dir / f"{dataset}.db",
                output_dir / f"{dataset}_graph.html",
                output_dir / f"{dataset}_graph_viewer",
            ):
                self._remove_output_artifact(artifact_path)

    def execute_user_query(
        self,
        question: str,
        *,
        strict_sql: bool = False,
    ) -> dict[str, Any]:
        runtime = self.ensure_runtime()
        bundle = execute_query(
            question,
            self.db_paths,
            runtime,
            self.agent,
            graph_dataset=self.graph_dataset,
            context_db=self.context_db,
            payload_mode=self.payload_mode,
            strict_sql=strict_sql,
        )
        self.runtime = bundle.get("runtime") or self.runtime
        self.agent = bundle.get("agent") or self.agent
        return bundle["result"]

    def _ensure_search_index(self) -> _SearchIndex:
        if self._search_index is not None:
            return self._search_index

        runtime = self.ensure_runtime()
        graph = get_networkx_graph(runtime)
        records: list[_SearchNodeRecord] = []
        records_by_class_lists: dict[str, list[_SearchNodeRecord]] = {}
        class_counts: Counter[str] = Counter()

        for node_id, node_data in graph.nodes(data=True):
            summary = _node_summary(str(node_id), dict(node_data))
            class_name = summary.class_name or "Unknown"
            record = _SearchNodeRecord(
                summary=summary,
                normalized_id=summary.id.lower(),
                normalized_label=summary.label.lower(),
                normalized_class_name=class_name.lower(),
                sort_key=(summary.label.lower(), summary.id.lower()),
            )
            records.append(record)
            records_by_class_lists.setdefault(class_name, []).append(record)
            class_counts[class_name] += 1

        records.sort(key=lambda item: item.sort_key)
        records_by_class = {
            class_name: tuple(sorted(items, key=lambda item: item.sort_key))
            for class_name, items in records_by_class_lists.items()
        }
        class_options = [
            SearchClassOption(
                value=class_name,
                label=class_name,
                count=count,
            )
            for class_name, count in sorted(
                class_counts.items(),
                key=lambda item: (-item[1], item[0].lower()),
            )
        ]
        self._search_index = _SearchIndex(
            records=tuple(records),
            records_by_class=records_by_class,
            class_options=class_options,
        )
        return self._search_index

    def _score_search_record(
        self,
        normalized_query: str,
        record: _SearchNodeRecord,
    ) -> int | None:
        haystacks = [
            record.normalized_id,
            record.normalized_label,
            record.normalized_class_name,
        ]
        if normalized_query in haystacks[0]:
            return 100
        if normalized_query in haystacks[1]:
            return 96
        if normalized_query in haystacks[2]:
            return 92
        score = max(fuzz.WRatio(normalized_query, haystack) for haystack in haystacks)
        if score < 55:
            return None
        return score

    def search_nodes(
        self,
        query: str | None = None,
        *,
        class_name: str | None = None,
        page: int = 1,
        page_size: int = 25,
    ) -> SearchResponse:
        normalized_query = (query or "").strip().lower()
        normalized_class_name = (class_name or "").strip() or None
        index = self._ensure_search_index()
        candidate_records = (
            index.records_by_class.get(normalized_class_name, ())
            if normalized_class_name
            else index.records
        )

        if normalized_query:
            scored: list[tuple[int, _SearchNodeRecord]] = []
            for record in candidate_records:
                score = self._score_search_record(normalized_query, record)
                if score is None:
                    continue
                scored.append((score, record))
            scored.sort(key=lambda item: (-item[0], *item[1].sort_key))
            total = len(scored)
            total_pages = (total + page_size - 1) // page_size if total else 0
            current_page = min(page, total_pages) if total_pages else 1
            start = (current_page - 1) * page_size
            end = start + page_size
            page_results = [record.summary for _score, record in scored[start:end]]
        else:
            total = len(candidate_records)
            total_pages = (total + page_size - 1) // page_size if total else 0
            current_page = min(page, total_pages) if total_pages else 1
            start = (current_page - 1) * page_size
            end = start + page_size
            page_results = [record.summary for record in candidate_records[start:end]]

        return SearchResponse(
            query=query or "",
            class_name=normalized_class_name,
            page=current_page,
            page_size=page_size,
            total=total,
            total_pages=total_pages,
            has_next=bool(total_pages and current_page < total_pages),
            has_previous=current_page > 1,
            class_options=index.class_options,
            results=page_results,
        )

    def _resolve_element_node_id(
        self,
        graph,
        element_id: str,
    ) -> str | None:
        normalized_query = element_id.strip()
        if not normalized_query:
            return None
        if normalized_query in graph:
            return normalized_query

        preferred_dataset = self.graph_dataset
        express_query_int: int | None = None
        try:
            express_query_int = int(normalized_query)
        except ValueError:
            express_query_int = None

        candidates: list[tuple[int, str]] = []
        for node_id, node_data in graph.nodes(data=True):
            node_id_text = str(node_id)
            properties = node_data.get("properties")
            if not isinstance(properties, dict):
                properties = {}

            match_score = 0
            if properties.get("GlobalId") == normalized_query:
                match_score = 100
            else:
                express_id = properties.get("ExpressId")
                if express_query_int is not None and express_id == express_query_int:
                    match_score = 96
                elif express_id is not None and str(express_id) == normalized_query:
                    match_score = 96
                elif node_id_text.endswith(f"::{normalized_query}"):
                    match_score = 88

            if match_score <= 0:
                continue

            dataset = node_data.get("dataset")
            dataset_bonus = (
                8
                if preferred_dataset is not None and dataset == preferred_dataset
                else 0
            )
            candidates.append((match_score + dataset_bonus, node_id_text))

        if not candidates:
            return None
        candidates.sort(key=lambda item: (-item[0], item[1]))
        return candidates[0][1]

    def get_element(self, element_id: str) -> dict[str, Any]:
        runtime = self.ensure_runtime()
        graph = get_networkx_graph(runtime)
        resolved_element_id = self._resolve_element_node_id(graph, element_id)
        result = query_ifc_graph(
            runtime,
            "get_element_properties",
            {"element_id": resolved_element_id or element_id},
        )
        if result["status"] != "ok":
            if resolved_element_id is not None:
                return build_node_payload(
                    resolved_element_id,
                    dict(graph.nodes[resolved_element_id]),
                    payload_mode=INTERNAL_PAYLOAD_MODE,
                )

            error = result.get("error") or {"message": "Unknown graph error"}
            raise KeyError(str(error))
        return result["data"]

    def _apply_build_artifacts(self, artifacts: ViewerBuildArtifacts) -> None:
        self.db_paths = [artifacts.db_path]
        self.graph_dataset = artifacts.dataset
        self.context_db = artifacts.db_path
        self.runtime = artifacts.runtime
        self.agent = None
        self.debug_graph_html_override = artifacts.debug_graph_html_path
        self.webgl_graph_bundle_override = artifacts.webgl_graph_bundle_path
        self.source_ifc_path = artifacts.ifc_path
        self._search_index = None

    @staticmethod
    def _normalize_progress(value: int) -> int:
        return max(0, min(100, int(value)))

    def _serialize_import_job(
        self,
        job: _ImportIfcJob,
    ) -> ImportIfcJobStatusResponse:
        return ImportIfcJobStatusResponse(
            job_id=job.job_id,
            status=job.status,
            stage=job.stage,
            message=job.message,
            progress=self._normalize_progress(job.progress),
            dataset=job.dataset,
            graph=job.graph,
            debug_graph_available=job.debug_graph_available,
            webgl_graph_available=job.webgl_graph_available,
            error=job.error,
        )

    def _store_import_job(self, job: _ImportIfcJob) -> None:
        with self._import_jobs_lock:
            self._import_jobs[job.job_id] = job

    def _update_import_job(self, job_id: str, **changes: Any) -> _ImportIfcJob:
        with self._import_jobs_lock:
            job = self._import_jobs[job_id]
            for key, value in changes.items():
                setattr(job, key, value)
            return job

    def get_import_job(self, job_id: str) -> ImportIfcJobStatusResponse:
        with self._import_jobs_lock:
            job = self._import_jobs.get(job_id)
        if job is None:
            raise KeyError(job_id)
        return self._serialize_import_job(job)

    def _run_import_ifc_job(
        self,
        job_id: str,
        ifc_path: Path,
    ) -> None:
        def progress(stage: str, message: str, progress_value: int) -> None:
            self._update_import_job(
                job_id,
                status="running",
                stage=stage,
                message=message,
                progress=self._normalize_progress(progress_value),
                error=None,
            )

        try:
            progress(
                "queued",
                "Upload complete. Waiting for the viewer build slot...",
                12,
            )
            response = self.import_ifc(ifc_path, progress_callback=progress)
        except Exception as exc:
            self._update_import_job(
                job_id,
                status="failed",
                stage="failed",
                message=f"IFC import failed for {ifc_path.name}.",
                error=str(exc),
            )
            return

        self._update_import_job(
            job_id,
            status="completed",
            stage="completed",
            message=response.message,
            progress=100,
            dataset=response.dataset,
            graph=response.graph,
            debug_graph_available=response.debug_graph_available,
            webgl_graph_available=response.webgl_graph_available,
            error=None,
        )

    def import_ifc(
        self,
        ifc_path: Path,
        *,
        progress_callback: BuildProgressCallback | None = None,
    ) -> ImportIfcResponse:
        if progress_callback is not None:
            waiting_message = "Waiting for the current viewer build to finish..."
            progress_callback(
                "queued",
                (
                    waiting_message
                    if self._build_lock.locked()
                    else "Preparing IFC build..."
                ),
                12,
            )
        with self._build_lock:
            if progress_callback is not None:
                progress_callback("preparing", "Starting IFC conversion...", 14)
            artifacts = build_viewer_artifacts_from_ifc(
                ifc_path,
                output_dir=self._primary_output_dir(),
                payload_mode=self.payload_mode,
                progress_callback=progress_callback,
            )
            if progress_callback is not None:
                progress_callback("finalizing", "Loading rebuilt viewer assets...", 98)
            self._apply_build_artifacts(artifacts)
            if self._is_viewer_upload_path(artifacts.ifc_path):
                self._cleanup_previous_viewer_uploads(
                    preserve_upload_path=artifacts.ifc_path,
                    preserve_dataset=artifacts.dataset,
                )
            return ImportIfcResponse(
                dataset=artifacts.dataset,
                graph=self.graph_summary(),
                message=f"Viewer rebuilt from {artifacts.ifc_path.name}.",
                debug_graph_available=self.debug_graph_available(),
                webgl_graph_available=self.webgl_graph_available(),
            )

    def import_uploaded_ifc(
        self,
        filename: str | None,
        payload: bytes,
    ) -> ImportIfcResponse:
        if not payload:
            raise ValueError("Uploaded IFC file is empty.")
        source_name = Path(filename or "uploaded.ifc").name
        suffix = Path(source_name).suffix.lower()
        if suffix != ".ifc":
            raise ValueError("Only .ifc uploads are supported.")

        dataset = sanitize_dataset_stem(Path(source_name).stem)
        upload_dir = self._viewer_upload_dir()
        upload_dir.mkdir(parents=True, exist_ok=True)
        upload_path = upload_dir / f"{dataset}.ifc"
        upload_path.write_bytes(payload)
        return self.import_ifc(upload_path)

    def start_import_uploaded_ifc(
        self,
        filename: str | None,
        payload: bytes,
    ) -> ImportIfcJobStatusResponse:
        if not payload:
            raise ValueError("Uploaded IFC file is empty.")
        source_name = Path(filename or "uploaded.ifc").name
        suffix = Path(source_name).suffix.lower()
        if suffix != ".ifc":
            raise ValueError("Only .ifc uploads are supported.")

        dataset = sanitize_dataset_stem(Path(source_name).stem)
        upload_dir = self._viewer_upload_dir()
        upload_dir.mkdir(parents=True, exist_ok=True)
        upload_path = upload_dir / f"{dataset}.ifc"
        upload_path.write_bytes(payload)

        job = _ImportIfcJob(
            job_id=uuid4().hex,
            source_name=source_name,
            message=f"Upload received for {source_name}. Starting viewer build...",
            progress=12,
        )
        self._store_import_job(job)
        Thread(
            target=self._run_import_ifc_job,
            args=(job.job_id, upload_path),
            daemon=True,
        ).start()
        return self._serialize_import_job(job)
