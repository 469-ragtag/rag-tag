from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from rag_tag.config import (
    AGENT_PROFILE_ENV_VAR,
    CONFIG_PATH_ENV_VAR,
    AppConfig,
    load_project_config,
)
from rag_tag.graph import close_runtime
from rag_tag.query_service import execute_query, find_sqlite_dbs
from rag_tag.router import RouteDecision

DEFAULT_GRAPH_QUESTIONS = [
    "Which rooms are adjacent to the kitchen?",
    "Find doors near the stair core.",
    "Which spaces are connected to the lobby?",
    "Find the path from the lobby to the server room.",
    "What is above the mechanical room?",
]

FORCED_GRAPH_DECISION = RouteDecision(
    route="graph",
    reason="forced graph evaluation",
    sql_request=None,
)

_SCRIPT_DIR = Path(__file__).resolve().parent
_MISSING = object()


def _parse_profile_name(value: str) -> str:
    profile_name = value.strip()
    if not profile_name:
        raise argparse.ArgumentTypeError("Profile name cannot be empty.")
    return profile_name


def _parse_experiment_name(value: str) -> str:
    experiment_name = value.strip()
    if not experiment_name:
        raise argparse.ArgumentTypeError("Experiment name cannot be empty.")
    return experiment_name


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
        return graph_dataset
    if db_path is not None:
        return db_path.stem
    return None


def _dedupe_preserving_order(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def resolve_agent_profiles(
    config: AppConfig,
    *,
    cli_profiles: Sequence[str] | None = None,
    experiment: str | None = None,
) -> list[str]:
    selected: list[str] = []

    if cli_profiles:
        selected = [profile.strip() for profile in cli_profiles if profile.strip()]
    elif experiment is not None:
        try:
            experiment_config = config.experiments[experiment]
        except KeyError as exc:
            raise ValueError(
                f"Experiment '{experiment}' was not found in project config."
            ) from exc
        selected.extend(profile.strip() for profile in experiment_config.profiles)
        if experiment_config.agent_profile:
            selected.append(experiment_config.agent_profile.strip())
    elif config.defaults.agent_profile:
        selected.append(config.defaults.agent_profile.strip())
    elif "agent" in config.profiles:
        selected.append("agent")

    selected = _dedupe_preserving_order([value for value in selected if value])
    if not selected:
        raise ValueError(
            "No agent profiles selected. Pass --profiles, use --experiment, "
            "or configure defaults.agent_profile."
        )

    missing_profiles = [name for name in selected if name not in config.profiles]
    if missing_profiles:
        raise ValueError(
            "Unknown agent profile(s): "
            f"{', '.join(missing_profiles)}. "
            f"Available profiles: {', '.join(sorted(config.profiles)) or '(none)'}"
        )

    return selected


def load_questions(questions_file: Path | None) -> list[str]:
    if questions_file is None:
        return list(DEFAULT_GRAPH_QUESTIONS)

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


@contextmanager
def temporary_profile_overrides(
    *,
    config_path: str | None = None,
    agent_profile: str | None = None,
) -> Iterator[None]:
    previous_values = {
        CONFIG_PATH_ENV_VAR: os.environ.get(CONFIG_PATH_ENV_VAR, _MISSING),
        AGENT_PROFILE_ENV_VAR: os.environ.get(AGENT_PROFILE_ENV_VAR, _MISSING),
    }

    try:
        if config_path is not None:
            os.environ[CONFIG_PATH_ENV_VAR] = config_path
        if agent_profile is not None:
            os.environ[AGENT_PROFILE_ENV_VAR] = agent_profile
        yield
    finally:
        for name, previous in previous_values.items():
            if previous is _MISSING:
                os.environ.pop(name, None)
            else:
                os.environ[name] = str(previous)


def build_profile_summary(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    question_count = len(results)
    total_duration = sum(float(item["duration_ms"]) for item in results)
    error_count = sum(1 for item in results if item["had_error"])
    warning_count = sum(1 for item in results if item["had_warning"])
    step_budget_warning_count = sum(
        1 for item in results if item["step_budget_warning"]
    )
    data_count = sum(1 for item in results if item["has_data"])

    return {
        "question_count": question_count,
        "error_count": error_count,
        "warning_count": warning_count,
        "step_budget_warning_count": step_budget_warning_count,
        "data_count": data_count,
        "average_duration_ms": round(total_duration / question_count, 3)
        if question_count
        else 0.0,
    }


def evaluate_graph_models(
    *,
    questions: Sequence[str],
    profile_names: Sequence[str],
    db_paths: list[Path],
    config_path: str | None,
    graph_dataset: str | None,
    context_db: Path | None,
    max_steps: int,
) -> dict[str, Any]:
    profile_reports: list[dict[str, Any]] = []

    for profile_name in profile_names:
        runtime = None
        agent = None
        results: list[dict[str, Any]] = []

        with temporary_profile_overrides(
            config_path=config_path,
            agent_profile=profile_name,
        ):
            try:
                for question in questions:
                    started_at = time.perf_counter()
                    bundle = execute_query(
                        question,
                        db_paths,
                        runtime,
                        agent,
                        decision=FORCED_GRAPH_DECISION,
                        graph_dataset=graph_dataset,
                        context_db=context_db,
                        graph_max_steps=max_steps,
                    )
                    duration_ms = round((time.perf_counter() - started_at) * 1000, 3)

                    result = bundle["result"]
                    runtime = bundle.get("runtime") or runtime
                    agent = bundle.get("agent") or agent
                    warning = result.get("warning")
                    error = result.get("error")
                    answer = result.get("answer")
                    has_data = result.get("data") is not None

                    results.append(
                        {
                            "profile_name": profile_name,
                            "question": question,
                            "answer": answer,
                            "warning": warning,
                            "error": error,
                            "duration_ms": duration_ms,
                            "had_error": error is not None,
                            "had_warning": warning is not None,
                            "step_budget_warning": _is_step_budget_warning(warning),
                            "has_data": has_data,
                        }
                    )
            finally:
                close_runtime(runtime)

        profile_reports.append(
            {
                "profile_name": profile_name,
                "summary": build_profile_summary(results),
                "results": results,
            }
        )

    return {
        "config_path": config_path,
        "db_paths": [str(path) for path in db_paths],
        "graph_dataset": graph_dataset,
        "context_db": str(context_db) if context_db is not None else None,
        "max_steps": max_steps,
        "forced_route": FORCED_GRAPH_DECISION.route,
        "questions": list(questions),
        "profiles": profile_reports,
    }


def _is_step_budget_warning(warning: object | None) -> bool:
    if warning is None:
        return False
    text = (
        json.dumps(warning, sort_keys=True) if not isinstance(warning, str) else warning
    )
    return "step budget exceeded" in text.lower()


def write_json_report(report: dict[str, Any], output_path: Path) -> Path:
    candidate = output_path.expanduser().resolve()
    candidate.parent.mkdir(parents=True, exist_ok=True)
    candidate.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return candidate


def print_report(report: dict[str, Any]) -> None:
    print(
        "Evaluated "
        f"{len(report['questions'])} question(s) across "
        f"{len(report['profiles'])} profile(s)."
    )
    if report.get("graph_dataset"):
        print(f"Graph dataset: {report['graph_dataset']}")

    for profile_report in report["profiles"]:
        summary = profile_report["summary"]
        print("-")
        print(f"Profile: {profile_report['profile_name']}")
        print(
            "Summary: "
            f"questions={summary['question_count']}, "
            f"errors={summary['error_count']}, "
            f"warnings={summary['warning_count']}, "
            f"avg_duration_ms={summary['average_duration_ms']}"
        )
        for result in profile_report["results"]:
            status = "ok"
            if result["had_error"]:
                status = "error"
            elif result["had_warning"]:
                status = "warning"

            print(f"[{status}] {result['question']}")
            if result["error"] is not None:
                print(f"  error: {result['error']}")
            else:
                print(f"  answer: {result['answer']}")
            if result["warning"] is not None:
                print(f"  warning: {result['warning']}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Compare graph-agent outputs across configured model profiles."
    )
    ap.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to config file used for provider/profile resolution.",
    )
    selection_group = ap.add_mutually_exclusive_group()
    selection_group.add_argument(
        "--profiles",
        nargs="+",
        type=_parse_profile_name,
        default=None,
        help="One or more graph-agent profiles to evaluate.",
    )
    selection_group.add_argument(
        "--experiment",
        type=_parse_experiment_name,
        default=None,
        help="Config experiment name that expands to graph-agent profiles.",
    )
    ap.add_argument(
        "--questions-file",
        type=Path,
        default=None,
        help="Plain-text or JSON file containing graph questions to evaluate.",
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
        help=(
            "Dataset stem for graph runtime loading "
            "(defaults to DB stem when unambiguous)."
        ),
    )
    ap.add_argument(
        "--max-steps",
        type=_parse_positive_int,
        default=20,
        help="Maximum graph-agent step budget per question.",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON report output path.",
    )
    args = ap.parse_args()

    try:
        config_override_path = _resolve_config_override_path(args.config)
        loaded = load_project_config(_SCRIPT_DIR, config_path=config_override_path)
        profile_names = resolve_agent_profiles(
            loaded.config,
            cli_profiles=args.profiles,
            experiment=args.experiment,
        )
        questions = load_questions(args.questions_file)
        db_paths, resolved_db_path = _resolve_db_paths(args.db)
        graph_dataset = _resolve_graph_dataset(args.graph_dataset, resolved_db_path)
        report = evaluate_graph_models(
            questions=questions,
            profile_names=profile_names,
            db_paths=db_paths,
            config_path=config_override_path,
            graph_dataset=graph_dataset,
            context_db=resolved_db_path,
            max_steps=args.max_steps,
        )
    except (FileNotFoundError, ValueError, json.JSONDecodeError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print_report(report)

    if args.output is not None:
        written_path = write_json_report(report, args.output)
        print(f"JSON report written to {written_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
