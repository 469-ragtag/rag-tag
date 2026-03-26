from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rag_tag.config import AppConfig, load_project_config
from rag_tag.evals.benchmark import build_benchmark_cli_config, run_benchmark_suite
from rag_tag.evals.reporting import top_leaderboard_rows


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


def _parse_orchestrator_name(value: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise argparse.ArgumentTypeError("Orchestrator name cannot be empty.")
    return cleaned


def _parse_tag(value: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise argparse.ArgumentTypeError("Tag cannot be empty.")
    return cleaned


def _parse_model_name(value: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise argparse.ArgumentTypeError("Model name cannot be empty.")
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


def _resolve_db_paths(db_path: Path | None) -> tuple[list[Path] | None, Path | None]:
    if db_path is None:
        return None, None

    candidate = db_path.expanduser().resolve()
    if not candidate.is_file():
        raise FileNotFoundError(f"SQLite database not found: {candidate}")
    return [candidate], candidate


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


def _format_metric(value: object) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.3f}".rstrip("0").rstrip(".")
    return str(value)


def _format_token_triplet(row: dict[str, object]) -> str:
    return "/".join(
        _format_metric(row.get(field_name))
        for field_name in ("avg_input_tokens", "avg_output_tokens", "avg_total_tokens")
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run end-to-end rag-tag benchmarks over the YAML case set. "
            "Prefer --preset with an optional --target override."
        )
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--preset", type=str, default=None)
    parser.add_argument("--target", type=str, default=None)
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
    parser.add_argument(
        "--orchestrators",
        nargs="*",
        type=_parse_orchestrator_name,
        default=None,
    )
    parser.add_argument("--repeat", type=_parse_positive_int, default=None)
    parser.add_argument("--max-concurrency", type=_parse_positive_int, default=None)
    parser.add_argument("--db", type=Path, default=None)
    parser.add_argument("--graph-dataset", type=str, default=None)
    parser.add_argument("--trace", action="store_true", default=False)
    parser.add_argument("--answer-judge-model", type=_parse_model_name, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--tags", nargs="*", type=_parse_tag, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    try:
        if (
            args.preset is None
            and args.target is None
            and args.experiment is None
            and args.questions_file is None
        ):
            raise ValueError(
                "Pass --preset, --target, --experiment, or --questions-file "
                "to select a benchmark dataset."
            )
        config_override_path = _resolve_config_override_path(args.config)
        app_config, loaded_config_path = _load_app_config(config_override_path)
        db_paths, context_db = _resolve_db_paths(args.db)

        cli_config = build_benchmark_cli_config(
            config=app_config,
            experiment_name=args.experiment,
            preset_name=args.preset,
            target_name=args.target,
            questions_file=args.questions_file,
            router_profiles=args.router_profiles,
            agent_profiles=args.agent_profiles,
            prompt_strategies=args.prompt_strategies,
            orchestrators=args.orchestrators,
            tags=args.tags,
            repeat=args.repeat,
            max_concurrency=args.max_concurrency,
            db_paths=db_paths,
            graph_dataset=args.graph_dataset,
            context_db=context_db,
            config_path=loaded_config_path,
            trace=args.trace,
            answer_judge_model=args.answer_judge_model,
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
    leaderboard_rows = []
    report_path = getattr(result, "report_path", None)
    if isinstance(report_path, Path) and report_path.is_file():
        try:
            import json

            payload = json.loads(report_path.read_text(encoding="utf-8"))
            raw_rows = payload.get("leaderboard")
            if isinstance(raw_rows, list):
                leaderboard_rows = [row for row in raw_rows if isinstance(row, dict)]
        except (OSError, ValueError):
            leaderboard_rows = []

    for index, row in enumerate(top_leaderboard_rows(leaderboard_rows), start=1):
        combo = (
            f"{row.get('router_profile') or 'default-router'} / "
            f"{row.get('agent_profile') or 'default-agent'} / "
            f"{row.get('prompt_strategy') or 'baseline'} / "
            f"{row.get('graph_orchestrator') or 'pydanticai'}"
        )
        answer_correct = row.get("answer_correct_rate")
        route_correct = row.get("route_correct_rate")
        avg_duration = row.get("avg_duration_ms")
        print(
            f"{index}. {combo} | "
            f"answer_correct={_format_metric(answer_correct)} | "
            f"route_correct={_format_metric(route_correct)} | "
            f"avg_tokens(in/out/total)={_format_token_triplet(row)} | "
            f"avg_ms={_format_metric(avg_duration)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
