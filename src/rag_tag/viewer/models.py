from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class DatasetInfo(BaseModel):
    name: str
    db_path: str | None = None
    selected: bool = False


class QueryRequest(BaseModel):
    question: str = Field(min_length=1)
    strict_sql: bool = False


class QueryPresentation(BaseModel):
    error: str | None = None
    answer: str | None = None
    warning: str | None = None
    sql_items: list[str] = Field(default_factory=list)
    sql_more_count: int = 0
    graph_sample: list[str] = Field(default_factory=list)
    details_json: str = ""
    details_truncated: bool = False


class QueryResponse(BaseModel):
    route: str | None = None
    decision: str | None = None
    answer: str | None = None
    presentation: QueryPresentation
    result: dict[str, Any]


class NodeSummary(BaseModel):
    id: str
    label: str
    class_name: str | None = None
    dataset: str | None = None
    geometry: tuple[float, float, float] | None = None


class SearchClassOption(BaseModel):
    value: str
    label: str
    count: int


class SearchResponse(BaseModel):
    query: str = ""
    class_name: str | None = None
    page: int = 1
    page_size: int = 25
    total: int
    total_pages: int = 0
    has_next: bool = False
    has_previous: bool = False
    class_options: list[SearchClassOption] = Field(default_factory=list)
    results: list[NodeSummary]


class ElementDetailResponse(BaseModel):
    element: dict[str, Any]


class GraphSummaryResponse(BaseModel):
    dataset: str | None = None
    datasets: list[str]
    node_count: int
    edge_count: int


class RecentIfcEntry(BaseModel):
    id: str
    source_name: str
    dataset: str
    size_bytes: int
    created_at_ms: int
    last_used_at_ms: int
    build_ready: bool = False
    active: bool = False


class BootstrapResponse(BaseModel):
    datasets: list[DatasetInfo]
    selected_dataset: str | None = None
    graph: GraphSummaryResponse
    debug_graph_url: str | None = None
    debug_graph_available: bool = False
    webgl_graph_manifest_url: str | None = None
    webgl_graph_available: bool = False
    model_viewer_url: str | None = None
    model_ifc_available: bool = False
    recent_ifcs: list[RecentIfcEntry] = Field(default_factory=list)
    active_recent_ifc_id: str | None = None
    active_ifc_name: str | None = None


class ModelBootstrapResponse(BaseModel):
    selected_dataset: str | None = None
    graph: GraphSummaryResponse
    model_viewer_url: str | None = None
    graph_viewer_url: str | None = None
    model_ifc_available: bool = False
    model_ifc_url: str | None = None
    model_ifc_name: str | None = None
    model_fragments_cache_key: str | None = None
    recent_ifcs: list[RecentIfcEntry] = Field(default_factory=list)
    active_recent_ifc_id: str | None = None


class ImportIfcResponse(BaseModel):
    dataset: str
    graph: GraphSummaryResponse
    message: str
    debug_graph_available: bool = False
    webgl_graph_available: bool = False
    reused_existing_build: bool = False


class ImportIfcJobStatusResponse(BaseModel):
    job_id: str
    status: str
    stage: str
    message: str
    progress: int = Field(default=0, ge=0, le=100)
    dataset: str | None = None
    graph: GraphSummaryResponse | None = None
    debug_graph_available: bool = False
    webgl_graph_available: bool = False
    reused_existing_build: bool = False
    error: str | None = None
