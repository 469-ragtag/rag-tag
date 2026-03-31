from __future__ import annotations

import hashlib
import json
import shutil
import time
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
    ViewerArtifactPaths,
    ViewerBuildArtifacts,
    build_viewer_artifacts_from_ifc,
    default_output_dir,
    sanitize_dataset_stem,
    viewer_artifact_paths,
    viewer_artifacts_ready,
)
from .models import (
    DatasetInfo,
    GraphSummaryResponse,
    ImportIfcJobStatusResponse,
    ImportIfcResponse,
    NodeSummary,
    RecentIfcEntry,
    SearchClassOption,
    SearchResponse,
)

_RECENT_IFC_LIMIT = 8


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


def _current_time_ms() -> int:
    return int(time.time() * 1000)


def _hash_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _recent_ifc_id(dataset: str, fingerprint_sha256: str) -> str:
    return f"{dataset}-{fingerprint_sha256[:12]}"


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
    reused_existing_build: bool = False
    error: str | None = None


@dataclass(slots=True, frozen=True)
class _RecentIfcRecord:
    id: str
    source_name: str
    dataset: str
    upload_path: Path
    artifact_dir: Path
    fingerprint_sha256: str
    size_bytes: int
    created_at_ms: int
    last_used_at_ms: int

    def to_manifest_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "source_name": self.source_name,
            "dataset": self.dataset,
            "upload_path": str(self.upload_path),
            "artifact_dir": str(self.artifact_dir),
            "fingerprint_sha256": self.fingerprint_sha256,
            "size_bytes": self.size_bytes,
            "created_at_ms": self.created_at_ms,
            "last_used_at_ms": self.last_used_at_ms,
        }


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
    active_output_dir: Path | None = None
    _search_index: _SearchIndex | None = field(default=None, init=False, repr=False)
    _build_lock: Lock = field(default_factory=Lock, init=False, repr=False)
    _import_jobs: dict[str, _ImportIfcJob] = field(
        default_factory=dict, init=False, repr=False
    )
    _import_jobs_lock: Lock = field(default_factory=Lock, init=False, repr=False)
    _recent_ifcs: dict[str, _RecentIfcRecord] = field(
        default_factory=dict, init=False, repr=False
    )

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
        if self.active_output_dir is not None:
            self.active_output_dir = self.active_output_dir.expanduser().resolve()
        self._recent_ifcs = self._load_recent_ifc_records()

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
        if self.active_output_dir is not None:
            candidates.append(self.active_output_dir)
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

    def _viewer_artifact_root_dir(self) -> Path:
        return self._primary_output_dir() / "_viewer_artifacts"

    def _recent_ifc_manifest_path(self) -> Path:
        return self._viewer_upload_dir() / "recent_ifcs.json"

    def _load_recent_ifc_records(self) -> dict[str, _RecentIfcRecord]:
        manifest_path = self._recent_ifc_manifest_path()
        if not manifest_path.is_file():
            return {}
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
        if not isinstance(payload, list):
            return {}

        records: dict[str, _RecentIfcRecord] = {}
        for item in payload:
            if not isinstance(item, dict):
                continue
            try:
                record = _RecentIfcRecord(
                    id=str(item["id"]),
                    source_name=str(item["source_name"]),
                    dataset=str(item["dataset"]),
                    upload_path=Path(str(item["upload_path"])).expanduser().resolve(),
                    artifact_dir=Path(str(item["artifact_dir"])).expanduser().resolve(),
                    fingerprint_sha256=str(item["fingerprint_sha256"]),
                    size_bytes=int(item["size_bytes"]),
                    created_at_ms=int(item["created_at_ms"]),
                    last_used_at_ms=int(item["last_used_at_ms"]),
                )
            except (KeyError, TypeError, ValueError):
                continue
            if record.upload_path.suffix.lower() != ".ifc":
                continue
            records[record.id] = record
        return records

    def _save_recent_ifc_records(self) -> None:
        manifest_path = self._recent_ifc_manifest_path()
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        records = sorted(
            self._recent_ifcs.values(),
            key=lambda item: (-item.last_used_at_ms, item.source_name.lower(), item.id),
        )
        manifest_path.write_text(
            json.dumps(
                [record.to_manifest_dict() for record in records],
                indent=2,
            ),
            encoding="utf-8",
        )

    def _recent_ifc_record_for_path(
        self,
        ifc_path: Path | None,
    ) -> _RecentIfcRecord | None:
        if ifc_path is None:
            return None
        resolved_path = ifc_path.expanduser().resolve()
        for record in self._recent_ifcs.values():
            if record.upload_path == resolved_path:
                return record
        return None

    def _recent_ifc_record(self, recent_ifc_id: str) -> _RecentIfcRecord:
        try:
            return self._recent_ifcs[recent_ifc_id]
        except KeyError as exc:
            raise KeyError(recent_ifc_id) from exc

    def _artifact_paths_for_recent_ifc(
        self,
        record: _RecentIfcRecord,
    ) -> ViewerArtifactPaths:
        return viewer_artifact_paths(record.artifact_dir, record.dataset)

    def _remove_recent_ifc_record(self, record: _RecentIfcRecord) -> None:
        record.upload_path.unlink(missing_ok=True)
        self._remove_output_artifact(record.artifact_dir)
        self._recent_ifcs.pop(record.id, None)

    def _prune_recent_ifc_records(
        self,
        preserve_recent_ifc_id: str | None = None,
    ) -> None:
        ordered_records = sorted(
            self._recent_ifcs.values(),
            key=lambda item: (
                0 if item.id == preserve_recent_ifc_id else 1,
                -item.last_used_at_ms,
                item.source_name.lower(),
                item.id,
            ),
        )
        kept_ids: set[str] = set()
        for record in ordered_records:
            if not record.upload_path.is_file():
                self._remove_recent_ifc_record(record)
                continue
            if len(kept_ids) < _RECENT_IFC_LIMIT or record.id == preserve_recent_ifc_id:
                kept_ids.add(record.id)
                continue
            self._remove_recent_ifc_record(record)
        self._save_recent_ifc_records()

    def _upsert_recent_ifc_record(
        self,
        *,
        source_name: str,
        dataset: str,
        upload_path: Path,
        fingerprint_sha256: str,
        size_bytes: int,
    ) -> _RecentIfcRecord:
        now_ms = _current_time_ms()
        recent_ifc_id = _recent_ifc_id(dataset, fingerprint_sha256)
        existing = self._recent_ifcs.get(recent_ifc_id)
        record = _RecentIfcRecord(
            id=recent_ifc_id,
            source_name=source_name,
            dataset=dataset,
            upload_path=upload_path.expanduser().resolve(),
            artifact_dir=(self._viewer_artifact_root_dir() / recent_ifc_id)
            .expanduser()
            .resolve(),
            fingerprint_sha256=fingerprint_sha256,
            size_bytes=size_bytes,
            created_at_ms=existing.created_at_ms if existing is not None else now_ms,
            last_used_at_ms=now_ms,
        )
        self._recent_ifcs[record.id] = record
        self._prune_recent_ifc_records(preserve_recent_ifc_id=record.id)
        return record

    def _mark_recent_ifc_used(self, ifc_path: Path | None) -> None:
        record = self._recent_ifc_record_for_path(ifc_path)
        if record is None:
            return
        updated = _RecentIfcRecord(
            id=record.id,
            source_name=record.source_name,
            dataset=record.dataset,
            upload_path=record.upload_path,
            artifact_dir=record.artifact_dir,
            fingerprint_sha256=record.fingerprint_sha256,
            size_bytes=record.size_bytes,
            created_at_ms=record.created_at_ms,
            last_used_at_ms=_current_time_ms(),
        )
        self._recent_ifcs[record.id] = updated
        self._prune_recent_ifc_records(preserve_recent_ifc_id=record.id)

    def active_recent_ifc_id(self) -> str | None:
        active_record = self._recent_ifc_record_for_path(self.active_ifc_path())
        return active_record.id if active_record is not None else None

    def list_recent_ifcs(self) -> list[RecentIfcEntry]:
        active_recent_ifc_id = self.active_recent_ifc_id()
        entries: list[RecentIfcEntry] = []
        for record in sorted(
            self._recent_ifcs.values(),
            key=lambda item: (-item.last_used_at_ms, item.source_name.lower(), item.id),
        ):
            if not record.upload_path.is_file():
                continue
            entries.append(
                RecentIfcEntry(
                    id=record.id,
                    source_name=record.source_name,
                    dataset=record.dataset,
                    size_bytes=record.size_bytes,
                    created_at_ms=record.created_at_ms,
                    last_used_at_ms=record.last_used_at_ms,
                    build_ready=viewer_artifacts_ready(
                        self._artifact_paths_for_recent_ifc(record)
                    ),
                    active=record.id == active_recent_ifc_id,
                )
            )
        return entries

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
        active_recent_ifc = self._recent_ifc_record_for_path(active_ifc_path)
        if active_recent_ifc is not None:
            return active_recent_ifc.source_name
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

    def _output_dir_for_ifc_path(self, ifc_path: Path) -> Path:
        recent_ifc = self._recent_ifc_record_for_path(ifc_path)
        if recent_ifc is not None:
            return recent_ifc.artifact_dir
        return self._primary_output_dir()

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
        self.active_output_dir = artifacts.jsonl_path.parent
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
            reused_existing_build=job.reused_existing_build,
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
        with self._import_jobs_lock:
            job = self._import_jobs[job_id]
            source_name = job.source_name

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
            response = self.import_ifc(
                ifc_path,
                source_name=source_name,
                progress_callback=progress,
            )
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
            reused_existing_build=response.reused_existing_build,
            error=None,
        )

    def import_ifc(
        self,
        ifc_path: Path,
        *,
        source_name: str | None = None,
        progress_callback: BuildProgressCallback | None = None,
    ) -> ImportIfcResponse:
        resolved_ifc_path = ifc_path.expanduser().resolve()
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
            recent_ifc_record = self._recent_ifc_record_for_path(resolved_ifc_path)
            artifacts = build_viewer_artifacts_from_ifc(
                resolved_ifc_path,
                output_dir=self._output_dir_for_ifc_path(resolved_ifc_path),
                payload_mode=self.payload_mode,
                reuse_existing=recent_ifc_record is not None,
                progress_callback=progress_callback,
            )
            if progress_callback is not None:
                progress_callback(
                    "finalizing",
                    (
                        "Loading cached viewer assets..."
                        if artifacts.reused_existing_build
                        else "Loading rebuilt viewer assets..."
                    ),
                    98,
                )
            self._apply_build_artifacts(artifacts)
            self._mark_recent_ifc_used(artifacts.ifc_path)
            display_name = (
                source_name or self.active_ifc_name() or artifacts.ifc_path.name
            )
            return ImportIfcResponse(
                dataset=artifacts.dataset,
                graph=self.graph_summary(),
                message=(
                    f"Viewer reopened from {display_name} using cached artifacts."
                    if artifacts.reused_existing_build
                    else f"Viewer rebuilt from {display_name}."
                ),
                debug_graph_available=self.debug_graph_available(),
                webgl_graph_available=self.webgl_graph_available(),
                reused_existing_build=artifacts.reused_existing_build,
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
        fingerprint_sha256 = _hash_bytes(payload)
        recent_ifc_id = _recent_ifc_id(dataset, fingerprint_sha256)
        upload_dir = self._viewer_upload_dir()
        upload_dir.mkdir(parents=True, exist_ok=True)
        upload_path = upload_dir / f"{recent_ifc_id}.ifc"
        upload_path.write_bytes(payload)
        self._upsert_recent_ifc_record(
            source_name=source_name,
            dataset=dataset,
            upload_path=upload_path,
            fingerprint_sha256=fingerprint_sha256,
            size_bytes=len(payload),
        )
        return self.import_ifc(upload_path, source_name=source_name)

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
        fingerprint_sha256 = _hash_bytes(payload)
        recent_ifc_id = _recent_ifc_id(dataset, fingerprint_sha256)
        upload_dir = self._viewer_upload_dir()
        upload_dir.mkdir(parents=True, exist_ok=True)
        upload_path = upload_dir / f"{recent_ifc_id}.ifc"
        upload_path.write_bytes(payload)
        self._upsert_recent_ifc_record(
            source_name=source_name,
            dataset=dataset,
            upload_path=upload_path,
            fingerprint_sha256=fingerprint_sha256,
            size_bytes=len(payload),
        )

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

    def start_activate_recent_ifc(
        self,
        recent_ifc_id: str,
    ) -> ImportIfcJobStatusResponse:
        record = self._recent_ifc_record(recent_ifc_id)
        if not record.upload_path.is_file():
            raise FileNotFoundError(record.upload_path)

        job = _ImportIfcJob(
            job_id=uuid4().hex,
            source_name=record.source_name,
            message=f"Opening recent IFC {record.source_name}...",
            progress=12,
        )
        self._store_import_job(job)
        Thread(
            target=self._run_import_ifc_job,
            args=(job.job_id, record.upload_path),
            daemon=True,
        ).start()
        return self._serialize_import_job(job)
