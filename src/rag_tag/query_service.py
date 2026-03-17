"""Shared query execution service for CLI and TUI."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import networkx as nx

from rag_tag.agent import GraphAgent
from rag_tag.graph import GraphRuntime, ensure_graph_runtime, load_graph_runtime
from rag_tag.graph_contract import merge_evidence_items
from rag_tag.ifc_sql_tool import SqlQueryError, query_ifc_sql
from rag_tag.paths import find_project_root
from rag_tag.router import RouteDecision, SqlRequest, route_question


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


def load_graph(
    dataset: str | None = None,
    payload_mode: str | None = None,
) -> nx.DiGraph:
    """Build the NetworkX graph from JSONL output files.

    Args:
        dataset: JSONL stem to load (e.g. ``"Building-Architecture"``).
            When ``None``, all ``.jsonl`` files in ``output/`` are used.
        payload_mode: Graph payload mode (``"full"`` or ``"minimal"``).
            When ``None``, the ``GRAPH_PAYLOAD_MODE`` env var is used,
            defaulting to ``"full"`` if unset.
    """
    from rag_tag.parser.jsonl_to_graph import build_graph  # noqa: PLC0415

    return build_graph(dataset=dataset, payload_mode=payload_mode)


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


def _require_explicit_graph_dataset(
    runtime: GraphRuntime | nx.DiGraph | nx.MultiDiGraph | None,
    graph_dataset: str | None,
) -> None:
    """Ensure graph_dataset is explicit when multiple datasets are available.

    Raises:
        ValueError: When multiple datasets are present but graph_dataset is None.
    """
    if graph_dataset is not None:
        # Explicit dataset provided — no check needed.
        return

    datasets = None
    if runtime is None:
        project_root = find_project_root(Path(__file__).resolve().parent)
        if project_root is not None:
            output_dir = project_root / "output"
            if output_dir.is_dir():
                datasets = sorted({path.stem for path in output_dir.glob("*.jsonl")})
    elif isinstance(runtime, GraphRuntime):
        datasets = runtime.get_networkx_graph().graph.get("datasets")
    elif isinstance(runtime, (nx.DiGraph, nx.MultiDiGraph)):
        datasets = runtime.graph.get("datasets")

    if datasets and len(datasets) > 1:
        raise ValueError(
            "Multiple graph datasets are available in the model. "
            "Please provide an explicit --graph-dataset to select which "
            f"dataset to query. Available datasets: {', '.join(datasets)}"
        )


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
    effective_limit = req.limit or 50

    successful_payloads: list[dict[str, Any]] = []
    combined_evidence: list[dict[str, Any]] = []
    last_payload: dict[str, Any] = {}
    failed_db_paths: list[str] = []
    db_errors: list[dict[str, str]] = []

    for db_path in db_paths:
        try:
            envelope = query_ifc_sql(db_path, req)
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
        successful_payloads.append(payload)
        combined_evidence = merge_evidence_items(
            combined_evidence,
            payload.get("evidence"),
        )

    if not last_payload:
        return _sql_error(
            decision,
            "All database queries failed.",
            warning=_sql_warning_details(failed_db_paths, db_errors),
        )

    merged_payload = _merge_sql_payloads(successful_payloads, req, effective_limit)
    result = {
        "route": "sql",
        "decision": decision.reason,
        "db_paths": [str(p) for p in db_paths],
        "answer": merged_payload["summary"],
        "data": {
            **merged_payload,
            "intent": last_payload.get("intent"),
            "filters": last_payload.get("filters"),
            "limit": effective_limit,
            "evidence": combined_evidence,
        },
        "sql": last_payload.get("sql"),
    }
    warning = _sql_warning_details(failed_db_paths, db_errors)
    if warning is not None:
        result["warning"] = warning
    return result


def _merge_sql_payloads(
    payloads: list[dict[str, Any]],
    req: SqlRequest,
    effective_limit: int,
) -> dict[str, Any]:
    label = req.ifc_class or "elements"

    if req.intent == "count":
        combined_count = sum(int(payload.get("count", 0)) for payload in payloads)
        if req.level_like:
            summary = (
                f"Found {combined_count} {label} matching level '{req.level_like}'."
            )
        else:
            summary = f"Found {combined_count} {label}."
        return {"count": combined_count, "summary": summary}

    if req.intent == "list":
        combined_total = sum(int(payload.get("total_count", 0)) for payload in payloads)
        combined_items: list[dict[str, Any]] = []
        for payload in payloads:
            merge_state = payload.get("merge_state") or {}
            full_items = merge_state.get("items") or payload.get("items") or []
            combined_items.extend(item for item in full_items if isinstance(item, dict))
        combined_items.sort(key=_sql_list_item_sort_key)
        combined_items = combined_items[:effective_limit]
        shown = len(combined_items)
        if req.level_like:
            summary = (
                f"Found {combined_total} {label} matching level '{req.level_like}', "
                f"showing {shown}."
            )
        else:
            summary = f"Found {combined_total} {label}, showing {shown}."
        return {
            "count": shown,
            "total_count": combined_total,
            "items": combined_items,
            "summary": summary,
        }

    if req.intent == "aggregate":
        return _merge_aggregate_payloads(payloads, req)

    if req.intent == "group":
        return _merge_group_payloads(payloads, req, effective_limit)

    raise ValueError(f"Unsupported SQL intent: {req.intent}")


def _merge_aggregate_payloads(
    payloads: list[dict[str, Any]],
    req: SqlRequest,
) -> dict[str, Any]:
    aggregate_op = req.aggregate_op
    label = req.ifc_class or "elements"
    field_label = req.aggregate_field.field if req.aggregate_field else label
    total_elements = sum(int(payload.get("total_elements", 0)) for payload in payloads)
    matched_value_count = sum(
        int(payload.get("matched_value_count", 0)) for payload in payloads
    )
    missing_value_count = sum(
        int(payload.get("missing_value_count", 0)) for payload in payloads
    )

    if aggregate_op == "count":
        aggregate_value = sum(
            int(payload.get("aggregate_value", 0) or 0) for payload in payloads
        )
    elif aggregate_op == "sum":
        aggregate_value = sum(
            float(payload.get("aggregate_value", 0) or 0) for payload in payloads
        )
    elif aggregate_op == "avg":
        total_sum = 0.0
        total_count = 0
        for payload in payloads:
            merge_state = payload.get("merge_state") or {}
            total_sum += float(merge_state.get("sum", 0) or 0)
            total_count += int(merge_state.get("matched_value_count", 0) or 0)
        aggregate_value = total_sum / total_count if total_count else None
    elif aggregate_op == "min":
        values = [
            float(payload["aggregate_value"])
            for payload in payloads
            if payload.get("aggregate_value") is not None
        ]
        aggregate_value = min(values) if values else None
    elif aggregate_op == "max":
        values = [
            float(payload["aggregate_value"])
            for payload in payloads
            if payload.get("aggregate_value") is not None
        ]
        aggregate_value = max(values) if values else None
    else:
        raise ValueError(f"Unsupported aggregate op: {aggregate_op}")

    if aggregate_op == "count" and req.aggregate_field is None:
        if req.level_like:
            summary = (
                f"Found {aggregate_value} {label} matching level '{req.level_like}'."
            )
        else:
            summary = f"Found {aggregate_value} {label}."
    elif req.level_like:
        summary = (
            f"Computed {aggregate_op} of {field_label} for {label} matching "
            f"level '{req.level_like}': {aggregate_value}."
        )
    else:
        summary = (
            f"Computed {aggregate_op} of {field_label} for {label}: {aggregate_value}."
        )

    return {
        "aggregate_op": aggregate_op,
        "aggregate_field": (
            {"source": req.aggregate_field.source, "field": req.aggregate_field.field}
            if req.aggregate_field is not None
            else None
        ),
        "aggregate_value": aggregate_value,
        "matched_value_count": matched_value_count,
        "missing_value_count": missing_value_count,
        "total_elements": total_elements,
        "count": matched_value_count
        if req.aggregate_field is not None
        else int(aggregate_value or 0),
        "summary": summary,
    }


def _sql_list_item_sort_key(item: dict[str, Any]) -> tuple[int, str, str, str, int]:
    name = item.get("name")
    express_id = item.get("express_id")
    if isinstance(express_id, int):
        normalized_express_id = express_id
    elif isinstance(express_id, str):
        try:
            normalized_express_id = int(express_id)
        except ValueError:
            normalized_express_id = -1
    else:
        normalized_express_id = -1
    return (
        0 if name is None else 1,
        "" if name is None else str(name),
        str(item.get("ifc_class") or ""),
        str(item.get("global_id") or ""),
        normalized_express_id,
    )


def _merge_group_payloads(
    payloads: list[dict[str, Any]],
    req: SqlRequest,
    effective_limit: int,
) -> dict[str, Any]:
    grouped_counts: dict[Any, int] = {}
    total_elements = 0
    matched_element_count = 0
    missing_value_count = 0

    for payload in payloads:
        total_elements += int(payload.get("total_elements", 0))
        matched_element_count += int(payload.get("matched_element_count", 0))
        missing_value_count += int(payload.get("missing_value_count", 0))
        merge_state = payload.get("merge_state") or {}
        full_groups = merge_state.get("groups") or payload.get("groups") or []
        for group in full_groups:
            key = group.get("group")
            grouped_counts[key] = grouped_counts.get(key, 0) + int(
                group.get("count", 0)
            )

    groups = [
        {"group": key, "count": count}
        for key, count in sorted(
            grouped_counts.items(),
            key=lambda item: (-item[1], "" if item[0] is None else str(item[0])),
        )
    ][:effective_limit]

    label = req.ifc_class or "elements"
    field_label = req.group_by.field if req.group_by is not None else "field"
    shown = len(groups)
    if req.level_like:
        summary = (
            f"Grouped {label} matching level '{req.level_like}' by {field_label}, "
            f"showing {shown} groups."
        )
    else:
        summary = f"Grouped {label} by {field_label}, showing {shown} groups."

    return {
        "group_by": (
            {"source": req.group_by.source, "field": req.group_by.field}
            if req.group_by is not None
            else None
        ),
        "groups": groups,
        "matched_element_count": matched_element_count,
        "missing_value_count": missing_value_count,
        "total_elements": total_elements,
        "count": shown,
        "summary": summary,
    }


def _sql_error(
    decision: RouteDecision,
    message: str,
    warning: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a SQL error result payload."""
    result = {
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
    """Execute graph query via agent."""
    agent_result = agent.run(question, runtime, max_steps=max_steps)
    return {
        "route": "graph",
        "decision": decision.reason,
        "runtime": runtime.backend_name,
        **agent_result,
    }


def execute_query(
    question: str,
    db_paths: list[Path],
    runtime: GraphRuntime | nx.DiGraph | nx.MultiDiGraph | None = None,
    agent: GraphAgent | None = None,
    graph: GraphRuntime | nx.DiGraph | nx.MultiDiGraph | None = None,
    *,
    decision: RouteDecision | None = None,
    debug_llm_io: bool = False,
    graph_dataset: str | None = None,
    context_db: Path | None = None,
    payload_mode: str | None = None,
    strict_sql: bool = False,
    graph_max_steps: int = 20,
) -> dict[str, Any]:
    """Execute a query through the full pipeline (routing + execution).

    Args:
        question: User question
        db_paths: All SQLite databases to query
        graph: NetworkX graph (or None, will be loaded if needed)
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

    Returns:
        Result dict with answer, route, decision, data, or error.
        Also returns updated graph and agent if they were loaded/created.
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
                "graph": graph,
                "agent": agent,
                "runtime": runtime,
            }

        _require_explicit_graph_dataset(runtime, graph_dataset)

        resolved_context_db = context_db or _resolve_context_db_path(
            db_paths, graph_dataset
        )
        runtime, agent = _ensure_graph_context(
            runtime or graph,
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
            "graph": runtime,
            "agent": agent,
            "runtime": runtime,
        }

    except Exception as exc:
        error_result = _routing_error(decision, str(exc))
        return {
            "result": error_result,
            "graph": runtime,
            "agent": agent,
            "runtime": runtime,
        }


def _normalize_db_path(raw_path: Path | str | None) -> str | None:
    if raw_path is None:
        return None
    return str(Path(raw_path).expanduser().resolve())


def _clear_graph_db_caches(graph: nx.DiGraph) -> None:
    """Clear graph-scoped DB caches after context DB changes."""
    graph.graph.pop("_property_cache", None)
    graph.graph.pop("_property_key_cache", None)

    cached_conn = graph.graph.pop("_db_lookup_conn", None)
    if cached_conn is not None:
        try:
            cached_conn.close()
        except Exception:  # noqa: BLE001
            pass


def _ensure_graph_context(
    graph: GraphRuntime | nx.DiGraph | nx.MultiDiGraph | None,
    agent: GraphAgent | None,
    debug_llm_io: bool,
    graph_dataset: str | None = None,
    db_path: Path | None = None,
    payload_mode: str | None = None,
) -> tuple[GraphRuntime, GraphAgent]:
    """Load graph and agent instances when missing; wire DB path into graph context.

    Args:
        graph: Existing graph or None to trigger loading. Can be a
            GraphRuntime or NetworkX graph.
        agent: Existing agent or None to trigger creation.
        debug_llm_io: Passed through to GraphAgent constructor.
        graph_dataset: JSONL stem for ``build_graph`` (None = all datasets).
        db_path: DB path to store on ``graph.graph["_db_path"]`` so that
            ``get_element_properties`` can perform DB-backed lookups.
            When None, any previously wired context is preserved.
        payload_mode: Optional graph payload mode override for graph loading.

    Returns:
        Tuple of (GraphRuntime, GraphAgent) ready for agent execution.
    """
    # Extract the underlying NetworkX graph through the public runtime API.
    existing_runtime = graph if isinstance(graph, GraphRuntime) else None
    nx_graph = (
        existing_runtime.get_networkx_graph() if existing_runtime is not None else graph
    )

    if nx_graph is None:
        nx_graph = load_graph(graph_dataset, payload_mode=payload_mode)

    if db_path is not None:
        # Wire the active DB path into the graph for tool-level property lookup.
        # When the DB context changes on an existing graph instance, clear
        # graph-scoped property caches to avoid stale cross-DB reads.
        resolved_db_path = db_path.expanduser().resolve()
        previous_db_path = _normalize_db_path(nx_graph.graph.get("_db_path"))
        current_db_path = str(resolved_db_path)
        if previous_db_path != current_db_path:
            _clear_graph_db_caches(nx_graph)
        nx_graph.graph["_db_path"] = resolved_db_path
        if existing_runtime is not None:
            existing_runtime.set_context_db_path(resolved_db_path)

    if agent is None:
        agent = GraphAgent(debug_llm_io=debug_llm_io)

    if existing_runtime is not None:
        return existing_runtime, agent

    selected_datasets: list[str] | None = None
    if graph_dataset is not None:
        selected_datasets = [graph_dataset]
    else:
        datasets = nx_graph.graph.get("datasets")
        if isinstance(datasets, list) and all(
            isinstance(item, str) for item in datasets
        ):
            selected_datasets = sorted(datasets)

    # Create the runtime through the configured backend factory so checked-in
    # config defaults and one-off GRAPH_BACKEND overrides are both honored.
    runtime = GraphRuntime.from_env(
        graph=nx_graph,
        db_path=db_path,
        selected_datasets=selected_datasets,
    )
    return runtime, agent


def _routing_error(decision: RouteDecision | None, message: str) -> dict[str, Any]:
    """Build a routing error result payload."""
    error_result = {"error": message, "route": "?", "decision": "routing failed"}
    if decision is not None:
        error_result["route"] = decision.route
        error_result["decision"] = decision.reason
    return error_result
