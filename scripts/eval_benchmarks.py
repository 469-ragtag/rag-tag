from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rag_tag.config import AppConfig, load_project_config
from rag_tag.evals.benchmark import build_benchmark_cli_config, run_benchmark_suite
from rag_tag.query_service import find_sqlite_dbs


def _parse_profile_name(value: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise argparse.ArgumentTypeError("Profile name cannot be empty.")
    return cleaned


def _parse_strategy_name(value: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise argparse.ArgumentTypeError("Prompt strategy cannot be empty.")
    return cleaned


def _parse_tag(value: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise argparse.ArgumentTypeError("Tag cannot be empty.")
    return cleaned


def _parse_positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected an integer value.") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("Value must be greater than zero.")
    return parsed


def _resolve_config_override_path(config_path: Path | None) -> str | None:
    if config_path is None:
        return None
    candidate = config_path.expanduser().resolve()
    if not candidate.is_file():
        raise FileNotFoundError(f"Config file not found: {candidate}")
    return str(candidate)


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
    db_path: Path | None,
) -> str | None:
    if graph_dataset:
        return graph_dataset.strip()
    if db_path is not None:
        return db_path.stem
    return None


def _load_app_config(config_override_path: str | None) -> tuple[AppConfig, str | None]:
    loaded = load_project_config(
        Path(__file__).resolve().parent,
        config_path=config_override_path,
    )
    resolved_config_path = (
        str(loaded.config_path)
        if loaded.config_path is not None
        else config_override_path
    )
    return (
        loaded.config,
        resolved_config_path,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run end-to-end rag-tag benchmarks over the YAML case set."
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--experiment", type=str, default=None)
    parser.add_argument("--questions-file", type=Path, default=None)
    parser.add_argument(
        "--router-profiles",
        nargs="*",
        type=_parse_profile_name,
        default=None,
    )
    parser.add_argument(
        "--agent-profiles",
        nargs="*",
        type=_parse_profile_name,
        default=None,
    )
    parser.add_argument(
        "--prompt-strategies",
        nargs="*",
        type=_parse_strategy_name,
        default=None,
    )
    parser.add_argument("--repeat", type=_parse_positive_int, default=None)
    parser.add_argument("--max-concurrency", type=_parse_positive_int, default=None)
    parser.add_argument("--db", type=Path, default=None)
    parser.add_argument("--graph-dataset", type=str, default=None)
    parser.add_argument("--trace", action="store_true", default=False)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--tags", nargs="*", type=_parse_tag, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    try:
        config_override_path = _resolve_config_override_path(args.config)
        db_paths, context_db = _resolve_db_paths(args.db)
        if not db_paths:
            raise FileNotFoundError(
                "No SQLite database found. Pass --db or generate output/*.db first."
            )

        graph_dataset = _resolve_graph_dataset(args.graph_dataset, context_db)
        app_config, loaded_config_path = _load_app_config(config_override_path)

        cli_config = build_benchmark_cli_config(
            config=app_config,
            experiment_name=args.experiment,
            questions_file=args.questions_file,
            router_profiles=args.router_profiles,
            agent_profiles=args.agent_profiles,
            prompt_strategies=args.prompt_strategies,
            tags=args.tags,
            repeat=args.repeat,
            max_concurrency=args.max_concurrency,
            db_paths=db_paths,
            graph_dataset=graph_dataset,
            context_db=context_db,
            config_path=loaded_config_path,
            trace=args.trace,
            output_dir=args.output_dir,
        )
        result = run_benchmark_suite(cli_config)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(f"Benchmark run: {result.experiment_name}")
    print(f"Dataset: {result.dataset_name}")
    print(f"Combinations: {len(result.entries)}")
    print(f"Trace requested: {'yes' if result.trace_requested else 'no'}")
    print(
        f"Logfire: {'enabled' if result.logfire_status.enabled else 'disabled'}"
        + (
            " (cloud sync)"
            if result.logfire_status.cloud_sync
            else " (local only)"
            if result.trace_requested
            else ""
        )
    )
    print(f"Artifacts: {result.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
