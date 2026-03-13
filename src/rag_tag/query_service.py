"""Shared query execution service for CLI and TUI."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import networkx as nx

from rag_tag.agent import GraphAgent
from rag_tag.config import GraphOrchestrationConfig, load_project_config
from rag_tag.graph import GraphRuntime, ensure_graph_runtime, load_graph_runtime
from rag_tag.ifc_sql_tool import SqlQueryError, query_ifc_sql
from rag_tag.paths import find_project_root
from rag_tag.router import RouteDecision, route_question

_VALID_GRAPH_ORCHESTRATORS = frozenset({"pydanticai", "langgraph"})


def find_sqlite_dbs() -> list[Path]:
    # return all .db files sorted by name so we query every loaded model,
    # not just whichever one was modified most recently
    project_root = find_project_root(Path(__file__).resolve().parent)
    if project_root is None:
        return []
    candidates: list[Path] = []
    for folder_name in ("output", "db"):
        folder = project_root / folder_name
        if not folder.exists():
            continue
        candidates.extend(folder.glob("*.db"))
    candidates.sort(key=lambda p: p.name)
    return candidates


def find_graph_datasets() -> list[str]:
    project_root = find_project_root(Path(__file__).resolve().parent)
    if project_root is None:
        return []
    out_dir = project_root / "output"
    if not out_dir.exists():
        return []
    return sorted(path.stem for path in out_dir.glob("*.jsonl"))


def load_graph(
    dataset: str | None = None,
    payload_mode: str | None = None,
) -> GraphRuntime:
    """Build the graph runtime from JSONL output files.

    Args:
        dataset: JSONL stem to load (e.g. ``"Building-Architecture"``).
            When ``None``, all ``.jsonl`` files in ``output/`` are used.
        payload_mode: Graph payload mode (``"full"`` or ``"minimal"``).
            When ``None``, the ``GRAPH_PAYLOAD_MODE`` env var is used,
            defaulting to ``"full"`` if unset.
    """
    return load_graph_runtime(dataset, payload_mode=payload_mode)


def resolve_graph_orchestrator() -> str:
    """Resolve the configured graph orchestrator name.

    Defaults to ``"pydanticai"`` when the config key is absent or blank.
    Raises ``ValueError`` for unrecognized configured values.
    """
    loaded = load_project_config(Path(__file__).resolve().parent)
    configured = loaded.config.defaults.graph_orchestrator
    if configured is None or not configured.strip():
        return "pydanticai"

    orchestrator = configured.strip().lower()
    if orchestrator not in _VALID_GRAPH_ORCHESTRATORS:
        allowed = ", ".join(sorted(_VALID_GRAPH_ORCHESTRATORS))
        raise ValueError(
            "Unsupported defaults.graph_orchestrator="
            f"{configured!r}. Allowed values: {allowed}."
        )
    return orchestrator


def get_graph_orchestration_config() -> GraphOrchestrationConfig:
    """Return the typed graph orchestration config block."""
    loaded = load_project_config(Path(__file__).resolve().parent)
    return loaded.config.graph_orchestration


def _available_graph_datasets(
    runtime: GraphRuntime | nx.DiGraph | nx.MultiDiGraph | None,
) -> list[str]:
    if isinstance(runtime, GraphRuntime):
        return sorted(runtime.selected_datasets)
    if runtime is not None:
        raw = runtime.graph.get("datasets")
        if isinstance(raw, list) and all(isinstance(item, str) for item in raw):
            return sorted(raw)
    return find_graph_datasets()


def _require_explicit_graph_dataset(
    runtime: GraphRuntime | nx.DiGraph | nx.MultiDiGraph | None,
    graph_dataset: str | None,
) -> None:
    if graph_dataset:
        return
    datasets = _available_graph_datasets(runtime)
    if len(datasets) <= 1:
        return
    raise ValueError(
        "Multiple graph datasets are available "
        f"({', '.join(datasets)}). Pass --graph-dataset <stem> "
        "or --db output/<stem>.db."
    )


def _resolve_context_db_path(
    db_paths: list[Path],
    graph_dataset: str | None,
) -> Path | None:
    """Resolve the primary DB path to wire into the graph context.

    Used so that ``get_element_properties`` can do DB-backed property lookups
    against the correct database rather than relying solely on in-memory graph
    data.

    Priority:
    1. Single DB path → use directly.
    2. Multiple DBs + *graph_dataset* → first DB whose stem matches the dataset.
    3. Multiple DBs + no dataset → ``None`` (ambiguous; skip DB lookup).
    4. No DBs → ``None``.
    """
    if not db_paths:
        return None
    if len(db_paths) == 1:
        return db_paths[0]
    if graph_dataset:
        for p in db_paths:
            if p.stem == graph_dataset:
                return p
    # Multiple databases with no clear selection — do not guess.
    return None


def _sql_warning_details(
    failed_db_paths: list[str],
    db_errors: list[dict[str, str]],
) -> dict[str, Any] | None:
    if not failed_db_paths and not db_errors:
        return None
    return {
        "failed_db_paths": failed_db_paths,
        "db_errors": db_errors,
    }


def execute_sql_query(
    decision: RouteDecision,
    db_paths: list[Path],
    *,
    strict_sql: bool = False,
) -> dict[str, Any]:
    if decision.sql_request is None:
        return _sql_error(decision, "Router did not produce a SQL request.")
    if not db_paths:
        return _sql_error(
            decision, "No SQLite database found. Run rag-tag-jsonl-to-sql."
        )

    req = decision.sql_request
    # Canonical effective limit for this request; used as the global cap across DBs.
    effective_limit = req.limit or 50

    # Query every database and merge counts/items across all models.
    combined_count = 0
    combined_total = 0
    combined_items: list[Any] = []
    last_payload: dict[str, Any] = {}
    failed_db_paths: list[str] = []
    db_errors: list[dict[str, str]] = []

    for db_path in db_paths:
        try:
            envelope = query_ifc_sql(db_path, decision.sql_request)
        except SqlQueryError as exc:
            failed_db_paths.append(str(db_path))
            db_errors.append({"db_path": str(db_path), "error": str(exc)})
            if strict_sql:
                return _sql_error(
                    decision,
                    (
                        "Strict SQL mode aborted due to database query failure: "
                        f"{db_path}: {exc}"
                    ),
                    warning=_sql_warning_details(failed_db_paths, db_errors),
                )
            continue
        if envelope["status"] != "ok":
            failed_db_paths.append(str(db_path))
            db_error = envelope.get("error")
            db_errors.append(
                {"db_path": str(db_path), "error": str(db_error or "Unknown SQL error")}
            )
            if strict_sql:
                return _sql_error(
                    decision,
                    (
                        "Strict SQL mode aborted due to database query failure: "
                        f"{db_path}: {db_error or 'Unknown SQL error'}"
                    ),
                    warning=_sql_warning_details(failed_db_paths, db_errors),
                )
            continue
        payload = envelope["data"]
        last_payload = payload
        combined_count += payload.get("count", 0)
        combined_total += payload.get("total_count", 0)
        combined_items.extend(payload.get("items") or [])

    if not last_payload:
        return _sql_error(
            decision,
            "All database queries failed.",
            warning=_sql_warning_details(failed_db_paths, db_errors),
        )

    # Enforce global list limit: merged items across all DBs must not exceed the
    # requested limit.  Each per-DB query already applies the same LIMIT clause,
    # so the only way to exceed it is when results come from multiple databases.
    if req.intent == "list":
        combined_items = combined_items[:effective_limit]

    # Rebuild summary string with the combined count.
    label = req.ifc_class or "elements"
    if req.intent == "count":
        if req.level_like:
            summary = (
                f"Found {combined_count} {label} matching level '{req.level_like}'."
            )
        else:
            summary = f"Found {combined_count} {label}."
        result_count = combined_count
    else:
        # Use actual item count post-cap so summary matches displayed rows.
        shown = len(combined_items)
        if req.level_like:
            summary = (
                f"Found {combined_total} {label} matching level"
                f" '{req.level_like}', showing {shown}."
            )
        else:
            summary = f"Found {combined_total} {label}, showing {shown}."
        result_count = shown

    result = {
        "route": "sql",
        "decision": decision.reason,
        "db_paths": [str(p) for p in db_paths],
        "answer": summary,
        "data": {
            "intent": last_payload.get("intent"),
            "filters": last_payload.get("filters"),
            "count": result_count,
            "total_count": combined_total,
            # Return the canonical effective limit, not a per-DB artifact.
            "limit": effective_limit,
            "items": combined_items,
        },
        "sql": last_payload.get("sql"),
    }
    warning = _sql_warning_details(failed_db_paths, db_errors)
    if warning is not None:
        result["warning"] = warning
    return result


def _sql_error(
    decision: RouteDecision,
    message: str,
    *,
    warning: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a SQL error result payload."""
    result: dict[str, Any] = {
        "route": "sql",
        "decision": decision.reason,
        "error": message,
    }
    if warning is not None:
        result["warning"] = warning
    return result


def execute_graph_query(
    question: str,
    runtime: GraphRuntime,
    agent: GraphAgent,
    decision: RouteDecision,
    *,
    max_steps: int = 20,
) -> dict[str, Any]:
    """Execute graph query via agent.

    Args:
        question: User question
        runtime: Graph runtime
        agent: Graph agent instance
        decision: Routing decision

    Returns:
        Result dict with answer, data, or error
    """
    agent_result = agent.run(question, runtime, max_steps=max_steps)
    return {
        "route": "graph",
        "decision": decision.reason,
        **agent_result,
    }


def execute_query(
    question: str,
    db_paths: list[Path],
    runtime: GraphRuntime | nx.DiGraph | nx.MultiDiGraph | None,
    agent: GraphAgent | None,
    *,
    decision: RouteDecision | None = None,
    debug_llm_io: bool = False,
    graph_dataset: str | None = None,
    context_db: Path | None = None,
    payload_mode: str | None = None,
    strict_sql: bool = False,
    graph_max_steps: int = 20,
    graph: GraphRuntime | nx.DiGraph | nx.MultiDiGraph | None = None,
) -> dict[str, Any]:
    """Execute a query through the full pipeline (routing + execution).

    Args:
        question: User question
        db_paths: All SQLite databases to query
        runtime: Graph runtime or raw NetworkX graph (or None, will be loaded if
            needed)
        agent: Graph agent (or None, will be created if needed)
        decision: Optional precomputed routing decision
        debug_llm_io: Enable debug printing
        graph_dataset: JSONL stem to load (e.g. "Building-Architecture").
            When None, all .jsonl files in output/ are used.
        context_db: Explicit DB path to wire into the graph context for
            ``get_element_properties`` lookups.  When None, it is inferred
            from *db_paths* and *graph_dataset* via
            ``_resolve_context_db_path``.  Callers that already know the
            selected DB (e.g. when ``--db`` was passed explicitly) should
            supply it here for clarity and correctness.
        payload_mode: Optional graph payload mode override (``"full"`` or
            ``"minimal"``).  When None, graph construction uses the
            ``GRAPH_PAYLOAD_MODE`` env var defaulting to ``"full"``.
        graph: Legacy alias for ``runtime`` preserved for compatibility with
            older callers.

    Returns:
        Result dict with answer, route, decision, data, or error.
        Also returns updated runtime and agent if they were loaded/created.
        The returned bundle includes both ``runtime`` and legacy ``graph`` keys.
    """
    if runtime is not None and graph is not None and runtime is not graph:
        raise ValueError("Pass either runtime or graph, not both.")
    if runtime is None and graph is not None:
        runtime = graph

    try:
        if decision is None:
            decision = route_question(question, debug_llm_io=debug_llm_io)

        if decision.route == "sql":
            result = execute_sql_query(decision, db_paths, strict_sql=strict_sql)
            return {
                "result": result,
                "runtime": runtime,
                "graph": runtime,
                "agent": agent,
            }

        _require_explicit_graph_dataset(runtime, graph_dataset)

        # Resolve the DB path for graph context if not provided by the caller.
        resolved_context_db = context_db or _resolve_context_db_path(
            db_paths, graph_dataset
        )

        runtime, agent = _ensure_graph_context(
            runtime,
            agent,
            debug_llm_io,
            graph_dataset,
            resolved_context_db,
            payload_mode=payload_mode,
        )
        result = execute_graph_query(
            question,
            runtime,
            agent,
            decision,
            max_steps=graph_max_steps,
        )
        return {
            "result": result,
            "runtime": runtime,
            "graph": runtime,
            "agent": agent,
        }

    except Exception as exc:
        error_result = _routing_error(decision, str(exc))
        return {
            "result": error_result,
            "runtime": runtime,
            "graph": runtime,
            "agent": agent,
        }


def _ensure_graph_context(
    runtime: GraphRuntime | nx.DiGraph | nx.MultiDiGraph | None,
    agent: GraphAgent | None,
    debug_llm_io: bool,
    graph_dataset: str | None = None,
    db_path: Path | None = None,
    payload_mode: str | None = None,
) -> tuple[GraphRuntime, GraphAgent]:
    """Load runtime and agent instances when missing; wire DB path into runtime context.

    Args:
        runtime: Existing runtime or None to trigger loading.
        agent: Existing agent or None to trigger creation.
        debug_llm_io: Passed through to GraphAgent constructor.
        graph_dataset: JSONL stem for graph loading.
        db_path: DB path stored on the runtime for DB-backed lookups.
        payload_mode: Optional graph payload mode override for graph loading.
    """
    runtime = ensure_graph_runtime(
        runtime,
        graph_dataset=graph_dataset,
        context_db_path=db_path,
        payload_mode=payload_mode,
    )
    if agent is None:
        agent = GraphAgent(debug_llm_io=debug_llm_io)
    return runtime, agent


def _routing_error(decision: RouteDecision | None, message: str) -> dict[str, Any]:
    """Build a routing error result payload."""
    error_result = {"error": message, "route": "?", "decision": "routing failed"}
    if decision is not None:
        error_result["route"] = decision.route
        error_result["decision"] = decision.reason
    return error_result
