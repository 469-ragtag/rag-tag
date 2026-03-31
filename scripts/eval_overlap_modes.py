from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from rag_tag.config import (
    CONFIG_PATH_ENV_VAR,
    DEFAULT_OVERLAP_XY_MIN_RATIO,
    DEFAULT_OVERLAP_XY_TOP_K,
)
from rag_tag.graph import close_runtime, wrap_networkx_graph
from rag_tag.parser.jsonl_to_graph import (
    build_graph,
    plot_interactive_graph_overlap_modes,
)
from rag_tag.query_service import execute_query, find_sqlite_dbs
from rag_tag.router import RouteDecision

DEFAULT_OVERLAP_MODE_QUESTIONS = [
    "What is above the mechanical room?",
    "Which elements are in Level 1?",
    "Which rooms are adjacent to the kitchen?",
    "What type is the main entrance door?",
    "Is there a tree outside the building?",
]
DEFAULT_OVERLAP_MODES = ["full", "threshold", "top_k", "none"]
FORCED_GRAPH_DECISION = RouteDecision(
    route="graph",
    reason="forced graph overlap-mode evaluation",
    sql_request=None,
)
_SCRIPT_DIR = Path(__file__).resolve().parent


def _parse_overlap_mode(value: str) -> str:
    mode = value.strip().lower()
    if mode not in {"full", "threshold", "top_k", "none"}:
        raise argparse.ArgumentTypeError(
            "Mode must be one of: full, threshold, top_k, none."
        )
    return mode


def _parse_positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected an integer value.") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("Value must be greater than zero.")
    return parsed


def _parse_ratio(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected a numeric ratio.") from exc
    if not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("Ratio must be between 0.0 and 1.0.")
    return parsed


def _resolve_config_override_path(config_path: Path | None) -> str | None:
    if config_path is None:
        return None
    candidate = config_path.expanduser().resolve()
    if not candidate.is_file():
        raise FileNotFoundError(f"Config file not found: {candidate}")
    return str(candidate)


def _resolve_jsonl_paths(jsonl_paths: Sequence[Path]) -> list[Path]:
    if not jsonl_paths:
        raise ValueError("At least one JSONL path must be provided.")
    resolved: list[Path] = []
    for path in jsonl_paths:
        candidate = path.expanduser().resolve()
        if not candidate.is_file():
            raise FileNotFoundError(f"JSONL file not found: {candidate}")
        resolved.append(candidate)
    return resolved


def _resolve_db_paths(db_path: Path | None) -> tuple[list[Path], Path | None]:
    if db_path is None:
        db_paths = find_sqlite_dbs()
        return db_paths, db_paths[0] if len(db_paths) == 1 else None

    candidate = db_path.expanduser().resolve()
    if not candidate.is_file():
        raise FileNotFoundError(f"SQLite database not found: {candidate}")
    return [candidate], candidate


def _resolve_graph_dataset(
    graph_dataset: str | None,
    jsonl_paths: Sequence[Path],
    db_path: Path | None,
) -> str | None:
    if graph_dataset:
        return graph_dataset
    if db_path is not None:
        return db_path.stem
    if len(jsonl_paths) == 1:
        return jsonl_paths[0].stem
    return None


def load_questions(questions_file: Path | None) -> list[str]:
    if questions_file is None:
        return list(DEFAULT_OVERLAP_MODE_QUESTIONS)

    candidate = questions_file.expanduser().resolve()
    if not candidate.is_file():
        raise FileNotFoundError(f"Questions file not found: {candidate}")

    raw_text = candidate.read_text(encoding="utf-8")
    if candidate.suffix.lower() == ".json":
        return _load_json_questions(json.loads(raw_text), source=str(candidate))

    questions = [
        line.strip()
        for line in raw_text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not questions:
        raise ValueError(f"Questions file {candidate} did not contain any questions.")
    return questions


def _load_json_questions(payload: object, *, source: str) -> list[str]:
    if isinstance(payload, dict):
        payload = payload.get("questions")

    if not isinstance(payload, list):
        raise ValueError(
            f"Questions file {source} must contain a JSON list of strings "
            "or an object with a 'questions' list."
        )

    questions: list[str] = []
    for item in payload:
        if isinstance(item, str) and item.strip():
            questions.append(item.strip())
            continue
        if isinstance(item, dict):
            question = item.get("question")
            if isinstance(question, str) and question.strip():
                questions.append(question.strip())
                continue
        raise ValueError(
            f"Questions file {source} contains an unsupported entry: {item!r}"
        )

    if not questions:
        raise ValueError(f"Questions file {source} did not contain any questions.")
    return questions


def resolve_overlap_mode_specs(
    modes: Sequence[str] | None = None,
    *,
    threshold_min_ratio: float = DEFAULT_OVERLAP_XY_MIN_RATIO,
    top_k: int = DEFAULT_OVERLAP_XY_TOP_K,
) -> list[dict[str, Any]]:
    selected = list(modes) if modes else list(DEFAULT_OVERLAP_MODES)
    seen: set[str] = set()
    ordered_modes: list[str] = []
    for value in selected:
        mode = _parse_overlap_mode(value)
        if mode in seen:
            continue
        seen.add(mode)
        ordered_modes.append(mode)

    return [
        {
            "mode": mode,
            "min_ratio": float(threshold_min_ratio),
            "top_k": int(top_k),
        }
        for mode in ordered_modes
    ]


@contextmanager
def temporary_config_override(config_path: str | None = None) -> Iterator[None]:
    previous = os.environ.get(CONFIG_PATH_ENV_VAR)
    try:
        if config_path is not None:
            os.environ[CONFIG_PATH_ENV_VAR] = config_path
        yield
    finally:
        if previous is None:
            os.environ.pop(CONFIG_PATH_ENV_VAR, None)
        else:
            os.environ[CONFIG_PATH_ENV_VAR] = previous


def summarize_graph(graph) -> dict[str, Any]:
    relation_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    for _source, _target, attrs in graph.edges(data=True):
        relation_counts[str(attrs.get("relation", "related_to"))] += 1
        source_counts[str(attrs.get("source"))] += 1

    topology_relations = set(graph.graph.get("edge_categories", {}).get("topology", []))
    topology_relation_counts = {
        name: relation_counts[name]
        for name in sorted(topology_relations)
        if relation_counts[name] > 0
    }

    top_degree_nodes = []
    for node_id, degree in sorted(
        graph.degree(),
        key=lambda item: (-item[1], str(item[0])),
    )[:10]:
        node = graph.nodes[node_id]
        top_degree_nodes.append(
            {
                "id": node_id,
                "label": node.get("label"),
                "class_": node.get("class_"),
                "degree": int(degree),
            }
        )

    return {
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "relation_counts": dict(sorted(relation_counts.items())),
        "topology_relation_counts": topology_relation_counts,
        "source_counts": dict(sorted(source_counts.items())),
        "top_relations": [
            {"relation": relation, "count": count}
            for relation, count in relation_counts.most_common(10)
        ],
        "top_degree_nodes": top_degree_nodes,
        "graph_build": graph.graph.get("graph_build", {}),
    }


def build_overlap_mode_graphs(
    *,
    jsonl_paths: Sequence[Path],
    mode_specs: Sequence[dict[str, Any]],
    config_path: str | None,
    payload_mode: str | None = None,
) -> dict[str, Any]:
    resolved_jsonl_paths = _resolve_jsonl_paths(jsonl_paths)
    graphs: dict[str, Any] = {}
    with temporary_config_override(config_path):
        for spec in mode_specs:
            graphs[spec["mode"]] = build_graph(
                list(resolved_jsonl_paths),
                payload_mode=payload_mode,
                overlap_xy_mode=spec["mode"],
                overlap_xy_min_ratio=spec["min_ratio"],
                overlap_xy_top_k=spec["top_k"],
            )
    return graphs


def evaluate_overlap_modes(
    *,
    jsonl_paths: Sequence[Path],
    mode_specs: Sequence[dict[str, Any]],
    questions: Sequence[str] | None,
    db_paths: list[Path],
    config_path: str | None,
    graph_dataset: str | None,
    context_db: Path | None,
    max_steps: int,
    payload_mode: str | None = None,
    graphs_by_mode: dict[str, Any] | None = None,
) -> dict[str, Any]:
    resolved_jsonl_paths = _resolve_jsonl_paths(jsonl_paths)
    built_graphs = graphs_by_mode or build_overlap_mode_graphs(
        jsonl_paths=resolved_jsonl_paths,
        mode_specs=mode_specs,
        config_path=config_path,
        payload_mode=payload_mode,
    )
    mode_reports: list[dict[str, Any]] = []

    with temporary_config_override(config_path):
        for spec in mode_specs:
            graph = built_graphs[spec["mode"]]
            runtime = wrap_networkx_graph(
                graph,
                context_db_path=context_db,
                payload_mode=payload_mode,
            )
            agent = None
            question_results: list[dict[str, Any]] = []
            try:
                for question in questions or []:
                    question_started_at = time.perf_counter()
                    bundle = execute_query(
                        question,
                        db_paths,
                        runtime,
                        agent,
                        decision=FORCED_GRAPH_DECISION,
                        graph_dataset=graph_dataset,
                        context_db=context_db,
                        graph_max_steps=max_steps,
                        payload_mode=payload_mode,
                    )
                    duration_ms = round(
                        (time.perf_counter() - question_started_at) * 1000, 3
                    )
                    result = bundle["result"]
                    runtime = bundle.get("runtime") or runtime
                    agent = bundle.get("agent") or agent
                    question_results.append(
                        {
                            "question": question,
                            "answer": result.get("answer"),
                            "warning": result.get("warning"),
                            "error": result.get("error"),
                            "duration_ms": duration_ms,
                            "had_error": result.get("error") is not None,
                            "had_warning": result.get("warning") is not None,
                        }
                    )
            finally:
                close_runtime(runtime)

            mode_reports.append(
                {
                    "mode": spec["mode"],
                    "min_ratio": spec["min_ratio"],
                    "top_k": spec["top_k"],
                    "build_duration_ms": round(
                        float(
                            graph.graph.get("graph_build", {})
                            .get("overlap_xy", {})
                            .get("build_duration_ms", 0.0)
                        ),
                        3,
                    ),
                    "graph": summarize_graph(graph),
                    "questions": question_results,
                }
            )

    return {
        "config_path": config_path,
        "jsonl_paths": [str(path) for path in resolved_jsonl_paths],
        "db_paths": [str(path) for path in db_paths],
        "graph_dataset": graph_dataset,
        "context_db": str(context_db) if context_db is not None else None,
        "max_steps": max_steps,
        "payload_mode": payload_mode,
        "questions": list(questions or []),
        "modes": mode_reports,
    }


def write_json_report(report: dict[str, Any], output_path: Path) -> Path:
    candidate = output_path.expanduser().resolve()
    candidate.parent.mkdir(parents=True, exist_ok=True)
    candidate.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return candidate


def print_report(report: dict[str, Any]) -> None:
    print(
        "Evaluated "
        f"{len(report['modes'])} overlap mode(s) on "
        f"{len(report['jsonl_paths'])} JSONL file(s)."
    )
    if report.get("graph_dataset"):
        print(f"Graph dataset: {report['graph_dataset']}")
    if report["questions"]:
        print(f"Questions per mode: {len(report['questions'])}")

    print(
        "mode       edges     overlaps_xy  above     below     build_ms  "
        "question_errors"
    )
    for mode_report in report["modes"]:
        graph = mode_report["graph"]
        relation_counts = graph["relation_counts"]
        question_errors = sum(
            1 for item in mode_report["questions"] if item["had_error"]
        )
        print(
            f"{mode_report['mode']:<10}"
            f"{graph['edges']:<10}"
            f"{relation_counts.get('overlaps_xy', 0):<13}"
            f"{relation_counts.get('above', 0):<10}"
            f"{relation_counts.get('below', 0):<10}"
            f"{mode_report['build_duration_ms']:<10}"
            f"{question_errors}"
        )


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Compare graph overlap-emission modes on graph size and queries."
    )
    ap.add_argument(
        "--jsonl",
        type=Path,
        nargs="+",
        required=True,
        help="One or more JSONL files to build into a graph for comparison.",
    )
    ap.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional config override path used while building and querying.",
    )
    ap.add_argument(
        "--db",
        type=Path,
        default=None,
        help="Optional SQLite database path used for graph context lookups.",
    )
    ap.add_argument(
        "--graph-dataset",
        type=str,
        default=None,
        help="Dataset stem for graph runtime execution.",
    )
    ap.add_argument(
        "--questions-file",
        type=Path,
        default=None,
        help="Optional plain-text or JSON file of graph questions to evaluate.",
    )
    ap.add_argument(
        "--skip-questions",
        action="store_true",
        default=False,
        help="Only compare graph statistics without running graph questions.",
    )
    ap.add_argument(
        "--modes",
        nargs="+",
        type=_parse_overlap_mode,
        default=None,
        help="Subset of overlap modes to compare.",
    )
    ap.add_argument(
        "--threshold-min-ratio",
        type=_parse_ratio,
        default=DEFAULT_OVERLAP_XY_MIN_RATIO,
        help="Minimum overlap ratio used in threshold mode.",
    )
    ap.add_argument(
        "--top-k",
        type=_parse_positive_int,
        default=DEFAULT_OVERLAP_XY_TOP_K,
        help="Top-K retention used in top_k mode.",
    )
    ap.add_argument(
        "--max-steps",
        type=_parse_positive_int,
        default=20,
        help="Maximum graph-agent step budget per question.",
    )
    ap.add_argument(
        "--payload-mode",
        type=str,
        default=None,
        help="Optional graph payload mode override (full or minimal).",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON report output path.",
    )
    ap.add_argument(
        "--html",
        type=Path,
        default=None,
        help="Optional interactive HTML output path with overlap-mode toggles.",
    )
    args = ap.parse_args()

    try:
        config_override_path = _resolve_config_override_path(args.config)
        resolved_jsonl_paths = _resolve_jsonl_paths(args.jsonl)
        db_paths, resolved_db_path = _resolve_db_paths(args.db)
        graph_dataset = _resolve_graph_dataset(
            args.graph_dataset,
            resolved_jsonl_paths,
            resolved_db_path,
        )
        questions = [] if args.skip_questions else load_questions(args.questions_file)
        mode_specs = resolve_overlap_mode_specs(
            args.modes,
            threshold_min_ratio=args.threshold_min_ratio,
            top_k=args.top_k,
        )
        started_at = time.perf_counter()
        graphs_by_mode = build_overlap_mode_graphs(
            jsonl_paths=resolved_jsonl_paths,
            mode_specs=mode_specs,
            config_path=config_override_path,
            payload_mode=args.payload_mode,
        )
        build_duration_ms = round((time.perf_counter() - started_at) * 1000, 3)
        for mode_name, graph in graphs_by_mode.items():
            graph.graph.setdefault("graph_build", {}).setdefault("overlap_xy", {})[
                "build_duration_ms"
            ] = build_duration_ms
        report = evaluate_overlap_modes(
            jsonl_paths=resolved_jsonl_paths,
            mode_specs=mode_specs,
            questions=questions,
            db_paths=db_paths,
            config_path=config_override_path,
            graph_dataset=graph_dataset,
            context_db=resolved_db_path,
            max_steps=args.max_steps,
            payload_mode=args.payload_mode,
            graphs_by_mode=graphs_by_mode,
        )
    except (FileNotFoundError, ValueError, json.JSONDecodeError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print_report(report)
    if args.output is not None:
        written_path = write_json_report(report, args.output)
        print(f"JSON report written to {written_path}")
    if args.html is not None:
        plot_interactive_graph_overlap_modes(graphs_by_mode, args.html)
        print(f"HTML visualization written to {args.html.expanduser().resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
