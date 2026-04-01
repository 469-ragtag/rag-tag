"""Benchmark suite execution and artifact writing helpers."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic_evals.reporting import EvaluationReport

from rag_tag.config import (
    AppConfig,
    BenchmarkPresetConfig,
    BenchmarkTargetConfig,
    ExperimentConfig,
)
from rag_tag.observability import LogfireStatus, setup_logfire
from rag_tag.query_service import find_sqlite_dbs

from .dataset import BenchmarkCase, BenchmarkDataset, load_benchmark_dataset
from .evaluators import DEFAULT_ANSWER_JUDGE_MODEL
from .reporting import (
    build_case_groups_rows,
    build_leaderboard_rows,
    build_runs_rows,
    write_csv_rows,
)
from .runner import BenchmarkExperimentConfig, evaluate_benchmark_dataset_async
from .strategies import resolve_benchmark_strategy

DEFAULT_BENCHMARK_GRAPH_ORCHESTRATOR = "pydanticai"


@dataclass(frozen=True)
class BenchmarkCombination:
    """One router x agent x strategy x orchestrator benchmark condition."""

    router_profile: str | None
    agent_profile: str | None
    prompt_strategy: str
    graph_orchestrator: str | None

    def display_name(self) -> str:
        router = self.router_profile or "default-router"
        agent = self.agent_profile or "default-agent"
        strategy = self.prompt_strategy or "baseline"
        orchestrator = self.graph_orchestrator or DEFAULT_BENCHMARK_GRAPH_ORCHESTRATOR
        return f"{router}__{agent}__{strategy}__{orchestrator}"


@dataclass(frozen=True)
class BenchmarkSuiteEntry:
    """One evaluated benchmark combination and its report."""

    combination: BenchmarkCombination
    report: EvaluationReport[Any, Any, Any]


@dataclass(frozen=True)
class BenchmarkSuiteResult:
    """In-memory result of a benchmark suite execution."""

    run_id: str
    experiment_name: str
    dataset_name: str
    dataset_path: str
    output_dir: Path
    trace_requested: bool
    logfire_status: LogfireStatus
    entries: list[BenchmarkSuiteEntry]
    manifest_path: Path
    report_path: Path


@dataclass(frozen=True)
class BenchmarkCliConfig:
    """Resolved CLI inputs for one benchmark suite run."""

    experiment_name: str
    dataset_path: Path
    db_paths: list[Path]
    config_path: str | None = None
    graph_dataset: str | None = None
    context_db: Path | None = None
    payload_mode: str | None = None
    strict_sql: bool = False
    graph_max_steps: int | None = None
    max_concurrency: int | None = None
    repeat: int = 1
    progress: bool = True
    include_answer_judge: bool = True
    answer_judge_model: str | None = DEFAULT_ANSWER_JUDGE_MODEL
    debug_llm_io: bool = False
    trace: bool = False
    tags: list[str] = field(default_factory=list)
    case_id: str | None = None
    output_dir: Path | None = None
    combinations: list[BenchmarkCombination] = field(default_factory=list)


def build_benchmark_cli_config(
    *,
    config: AppConfig,
    experiment_name: str | None,
    preset_name: str | None,
    target_name: str | None,
    questions_file: Path | None,
    router_profiles: list[str] | None,
    agent_profiles: list[str] | None,
    prompt_strategies: list[str] | None,
    orchestrators: list[str] | None,
    tags: list[str] | None,
    repeat: int | None,
    max_concurrency: int | None,
    db_paths: list[Path] | None,
    graph_dataset: str | None,
    context_db: Path | None,
    config_path: str | None,
    payload_mode: str | None = None,
    strict_sql: bool = False,
    graph_max_steps: int | None = None,
    progress: bool = True,
    include_answer_judge: bool = True,
    answer_judge_model: str | None = None,
    debug_llm_io: bool = False,
    trace: bool = False,
    case_id: str | None = None,
    output_dir: Path | None = None,
) -> BenchmarkCliConfig:
    """Resolve CLI/config inputs into one executable benchmark suite config."""

    experiment = _resolve_experiment_config(config, experiment_name)
    preset = _resolve_benchmark_preset_config(config, preset_name)
    target = _resolve_benchmark_target_config(
        config,
        target_name if target_name is not None else preset.target if preset else None,
    )
    dataset_path = _resolve_dataset_path(
        questions_file,
        target,
        experiment,
        config_path=config_path,
    )
    resolved_db_paths = _resolve_db_paths(
        db_paths,
        target,
        config_path=config_path,
    )
    resolved_context_db = _resolve_context_db(
        explicit_context_db=context_db,
        db_paths=resolved_db_paths,
        target=target,
        config_path=config_path,
    )
    resolved_graph_dataset = _resolve_graph_dataset(
        graph_dataset,
        context_db=resolved_context_db,
        target=target,
    )
    if preset_name is not None:
        resolved_experiment_name = preset_name
    elif experiment_name is not None:
        resolved_experiment_name = experiment_name
    elif target_name is not None:
        resolved_experiment_name = target_name
    elif preset is not None:
        resolved_experiment_name = preset.target
    else:
        resolved_experiment_name = dataset_path.stem

    combinations = expand_benchmark_matrix(
        router_profiles=router_profiles
        or _list_or_empty(preset, "router_profiles")
        or _list_or_empty(experiment, "router_profiles"),
        agent_profiles=agent_profiles
        or _list_or_empty(preset, "agent_profiles")
        or _list_or_empty(experiment, "agent_profiles"),
        prompt_strategies=prompt_strategies
        or _list_or_empty(preset, "prompt_strategies")
        or _list_or_empty(experiment, "prompt_strategies"),
        graph_orchestrators=orchestrators
        or _list_or_empty(preset, "graph_orchestrators")
        or _list_or_empty(experiment, "graph_orchestrators"),
        fallback_router_profile=(
            experiment.router_profile
            if experiment is not None and experiment.router_profile
            else config.defaults.router_profile
        ),
        fallback_agent_profile=(
            experiment.agent_profile
            if experiment is not None and experiment.agent_profile
            else config.defaults.agent_profile
        ),
        fallback_graph_orchestrator=(
            experiment.graph_orchestrator
            if experiment is not None and experiment.graph_orchestrator
            else config.defaults.graph_orchestrator
            or DEFAULT_BENCHMARK_GRAPH_ORCHESTRATOR
        ),
    )

    resolved_tags = list(
        tags
        or (preset.tags if preset is not None and preset.tags else [])
        or (experiment.tags if experiment is not None and experiment.tags else [])
    )
    resolved_case_id = case_id.strip() if case_id is not None else None
    if resolved_case_id == "":
        resolved_case_id = None
    resolved_repeat = (
        repeat
        if repeat is not None
        else preset.repeat
        if preset is not None and preset.repeat
        else experiment.repeat
        if experiment is not None and experiment.repeat
        else 1
    )
    resolved_max_concurrency = (
        max_concurrency
        if max_concurrency is not None
        else preset.max_concurrency
        if preset is not None and preset.max_concurrency is not None
        else experiment.max_concurrency
        if experiment is not None
        else None
    )
    resolved_answer_judge_model = (
        answer_judge_model
        if answer_judge_model is not None
        else preset.answer_judge_model
        if preset is not None and preset.answer_judge_model
        else experiment.answer_judge_model
        if experiment is not None and experiment.answer_judge_model
        else DEFAULT_ANSWER_JUDGE_MODEL
    )
    resolved_runtime_config_path = config_path
    if (
        resolved_runtime_config_path is None
        and target is not None
        and target.config_path
    ):
        resolved_runtime_config_path = str(
            _resolve_path_like(target.config_path, config_path=config_path)
        )

    return BenchmarkCliConfig(
        experiment_name=resolved_experiment_name,
        dataset_path=dataset_path,
        db_paths=resolved_db_paths,
        config_path=resolved_runtime_config_path,
        graph_dataset=resolved_graph_dataset,
        context_db=resolved_context_db,
        payload_mode=payload_mode,
        strict_sql=strict_sql,
        graph_max_steps=graph_max_steps,
        max_concurrency=resolved_max_concurrency,
        repeat=resolved_repeat,
        progress=progress,
        include_answer_judge=include_answer_judge,
        answer_judge_model=resolved_answer_judge_model,
        debug_llm_io=debug_llm_io,
        trace=trace,
        tags=resolved_tags,
        case_id=resolved_case_id,
        output_dir=output_dir,
        combinations=combinations,
    )


def _resolve_benchmark_preset_config(
    config: AppConfig,
    preset_name: str | None,
) -> BenchmarkPresetConfig | None:
    if preset_name is None:
        return None
    try:
        return config.benchmark_presets[preset_name]
    except KeyError as exc:
        raise ValueError(
            f"Benchmark preset '{preset_name}' was not found in project config."
        ) from exc


def _resolve_benchmark_target_config(
    config: AppConfig,
    target_name: str | None,
) -> BenchmarkTargetConfig | None:
    if target_name is None:
        return None
    try:
        return config.benchmark_targets[target_name]
    except KeyError as exc:
        raise ValueError(
            f"Benchmark target '{target_name}' was not found in project config."
        ) from exc


def expand_benchmark_matrix(
    *,
    router_profiles: list[str],
    agent_profiles: list[str],
    prompt_strategies: list[str],
    graph_orchestrators: list[str],
    fallback_router_profile: str | None,
    fallback_agent_profile: str | None,
    fallback_graph_orchestrator: str | None,
) -> list[BenchmarkCombination]:
    """Expand the configured benchmark matrix into ordered combinations."""

    resolved_router_profiles = router_profiles or [fallback_router_profile]
    resolved_agent_profiles = agent_profiles or [fallback_agent_profile]
    resolved_prompt_strategies = prompt_strategies or ["baseline"]
    resolved_graph_orchestrators = graph_orchestrators or [fallback_graph_orchestrator]

    combinations: list[BenchmarkCombination] = []
    seen: set[tuple[str | None, str | None, str, str | None]] = set()
    for router_profile in resolved_router_profiles:
        for agent_profile in resolved_agent_profiles:
            for prompt_strategy in resolved_prompt_strategies:
                normalized_strategy = resolve_benchmark_strategy(
                    prompt_strategy.strip() or "baseline"
                ).name
                for graph_orchestrator in resolved_graph_orchestrators:
                    normalized_orchestrator = _normalize_graph_orchestrator_name(
                        graph_orchestrator,
                        fallback=fallback_graph_orchestrator,
                    )
                    key = (
                        router_profile,
                        agent_profile,
                        normalized_strategy,
                        normalized_orchestrator,
                    )
                    if key in seen:
                        continue
                    seen.add(key)
                    combinations.append(
                        BenchmarkCombination(
                            router_profile=router_profile.strip()
                            if router_profile
                            else None,
                            agent_profile=(
                                agent_profile.strip() if agent_profile else None
                            ),
                            prompt_strategy=normalized_strategy,
                            graph_orchestrator=normalized_orchestrator,
                        )
                    )

    if not combinations:
        raise ValueError("Benchmark matrix expansion produced no combinations.")

    return combinations


def _normalize_graph_orchestrator_name(
    orchestrator: str | None,
    *,
    fallback: str | None,
) -> str | None:
    candidate = orchestrator or fallback
    if candidate is None:
        return None
    normalized = candidate.strip().lower()
    if normalized in {"pydanticai", "langgraph"}:
        return normalized
    raise ValueError(
        "Unsupported benchmark graph orchestrator "
        f"{candidate!r}. Allowed values: langgraph, pydanticai."
    )


def load_benchmark_dataset_with_filters(
    dataset_path: Path,
    *,
    tags: list[str] | None = None,
    case_id: str | None = None,
) -> BenchmarkDataset:
    """Load a benchmark dataset and optionally filter it by cutoff and tags."""

    dataset = load_benchmark_dataset(dataset_path)
    filtered_cases = list(dataset.cases)

    normalized_case_id = case_id.strip() if case_id is not None else None
    if normalized_case_id == "":
        normalized_case_id = None
    if normalized_case_id is not None:
        matched_index = next(
            (
                index
                for index, case in enumerate(filtered_cases)
                if case.id == normalized_case_id
            ),
            None,
        )
        if matched_index is None:
            raise ValueError(
                f"Benchmark dataset case-id cutoff was not found: {normalized_case_id}"
            )
        filtered_cases = filtered_cases[: matched_index + 1]

    if not tags:
        return BenchmarkDataset(
            schema_version=dataset.schema_version,
            dataset_name=dataset.dataset_name,
            cases=filtered_cases,
        )

    normalized_tags = {tag.strip() for tag in tags if tag.strip()}
    if not normalized_tags:
        return BenchmarkDataset(
            schema_version=dataset.schema_version,
            dataset_name=dataset.dataset_name,
            cases=filtered_cases,
        )

    filtered_cases = [
        case for case in filtered_cases if normalized_tags.intersection(case.tags)
    ]
    if not filtered_cases:
        within_cutoff = (
            f" within case-id cutoff {normalized_case_id}"
            if normalized_case_id is not None
            else ""
        )
        raise ValueError(
            "Benchmark dataset tag filter matched no cases: "
            f"{', '.join(sorted(normalized_tags))}{within_cutoff}"
        )

    return BenchmarkDataset(
        schema_version=dataset.schema_version,
        dataset_name=dataset.dataset_name,
        cases=filtered_cases,
    )


def load_benchmark_dataset_with_tags(
    dataset_path: Path,
    *,
    tags: list[str] | None = None,
) -> BenchmarkDataset:
    """Backward-compatible tag-only dataset loader."""

    return load_benchmark_dataset_with_filters(dataset_path, tags=tags)


def run_benchmark_suite(config: BenchmarkCliConfig) -> BenchmarkSuiteResult:
    """Run the benchmark matrix, optionally with Logfire tracing."""

    return asyncio.run(run_benchmark_suite_async(config))


async def run_benchmark_suite_async(
    config: BenchmarkCliConfig,
) -> BenchmarkSuiteResult:
    """Run the benchmark matrix, optionally with Logfire tracing."""

    logfire_status = setup_logfire(enabled=config.trace, console=True)
    dataset = load_benchmark_dataset_with_filters(
        config.dataset_path,
        tags=config.tags,
        case_id=config.case_id,
    )
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_dir = _resolve_output_dir(config.output_dir, run_id)
    output_dir.mkdir(parents=True, exist_ok=True)

    entries: list[BenchmarkSuiteEntry] = []
    for combination in config.combinations:
        experiment_config = BenchmarkExperimentConfig(
            db_paths=config.db_paths,
            config_path=config.config_path,
            router_profile=combination.router_profile,
            agent_profile=combination.agent_profile,
            prompt_strategy=combination.prompt_strategy,
            graph_orchestrator=combination.graph_orchestrator,
            graph_dataset=config.graph_dataset,
            context_db=config.context_db,
            payload_mode=config.payload_mode,
            strict_sql=config.strict_sql,
            graph_max_steps=config.graph_max_steps,
            max_concurrency=config.max_concurrency,
            progress=config.progress,
            repeat=config.repeat,
            answer_judge_model=config.answer_judge_model,
            include_answer_judge=config.include_answer_judge,
            debug_llm_io=config.debug_llm_io,
            report_metadata=_build_entry_report_metadata(
                config=config,
                dataset=dataset,
                logfire_status=logfire_status,
            ),
        )
        report = await evaluate_benchmark_dataset_async(
            dataset,
            experiment=experiment_config,
            experiment_name=f"{config.experiment_name}__{combination.display_name()}",
        )
        entries.append(BenchmarkSuiteEntry(combination=combination, report=report))

    runs_rows = build_runs_rows(entries)
    case_group_rows = build_case_groups_rows(entries)
    leaderboard_rows = build_leaderboard_rows(entries, repeat=config.repeat)
    report_path = output_dir / "report.json"
    manifest_path = output_dir / "run_manifest.json"
    runs_path = output_dir / "runs.csv"
    case_groups_path = output_dir / "case_groups.csv"
    leaderboard_path = output_dir / "leaderboard.csv"
    write_csv_rows(runs_path, runs_rows, kind="runs")
    write_csv_rows(case_groups_path, case_group_rows, kind="case_groups")
    write_csv_rows(leaderboard_path, leaderboard_rows, kind="leaderboard")
    _write_json(
        report_path,
        _serialize_suite_report(
            config,
            dataset,
            run_id,
            logfire_status,
            entries,
            runs_rows=runs_rows,
            case_group_rows=case_group_rows,
            leaderboard_rows=leaderboard_rows,
        ),
    )
    _write_json(
        manifest_path,
        _build_run_manifest(
            config=config,
            dataset=dataset,
            run_id=run_id,
            logfire_status=logfire_status,
            entries=entries,
            report_path=report_path,
            runs_path=runs_path,
            case_groups_path=case_groups_path,
            leaderboard_path=leaderboard_path,
        ),
    )

    return BenchmarkSuiteResult(
        run_id=run_id,
        experiment_name=config.experiment_name,
        dataset_name=dataset.dataset_name,
        dataset_path=str(config.dataset_path),
        output_dir=output_dir,
        trace_requested=config.trace,
        logfire_status=logfire_status,
        entries=entries,
        manifest_path=manifest_path,
        report_path=report_path,
    )


def _resolve_experiment_config(
    config: AppConfig,
    experiment_name: str | None,
) -> ExperimentConfig | None:
    if experiment_name is None:
        return None
    try:
        return config.experiments[experiment_name]
    except KeyError as exc:
        raise ValueError(
            f"Experiment '{experiment_name}' was not found in project config."
        ) from exc


def _resolve_dataset_path(
    questions_file: Path | None,
    target: BenchmarkTargetConfig | None,
    experiment: ExperimentConfig | None,
    *,
    config_path: str | None,
) -> Path:
    if questions_file is not None:
        return _resolve_existing_file(
            questions_file,
            config_path=config_path,
            kind="Benchmark dataset file",
        )
    if target is not None:
        return _resolve_existing_file(
            target.questions_file,
            config_path=config_path,
            kind="Benchmark dataset file",
        )
    if experiment is not None and experiment.questions_file:
        return _resolve_existing_file(
            experiment.questions_file,
            config_path=config_path,
            kind="Benchmark dataset file",
        )
    raise ValueError(
        "No benchmark questions file selected. Pass --questions-file, --target, "
        "--preset, or use an experiment with questions_file configured."
    )


def _resolve_db_paths(
    explicit_db_paths: list[Path] | None,
    target: BenchmarkTargetConfig | None,
    *,
    config_path: str | None,
) -> list[Path]:
    if explicit_db_paths:
        return [path.expanduser().resolve() for path in explicit_db_paths]

    configured_paths: list[str] = []
    if target is not None:
        if target.db_path:
            configured_paths.append(target.db_path)
        configured_paths.extend(target.db_paths)
    if configured_paths:
        return [
            _resolve_existing_file(
                path,
                config_path=config_path,
                kind="SQLite database",
            )
            for path in configured_paths
        ]

    discovered_paths = find_sqlite_dbs()
    if discovered_paths:
        return discovered_paths
    raise FileNotFoundError(
        "No SQLite database found. Pass --db, use a benchmark target with db_paths, "
        "or generate output/*.db first."
    )


def _resolve_context_db(
    *,
    explicit_context_db: Path | None,
    db_paths: list[Path],
    target: BenchmarkTargetConfig | None,
    config_path: str | None,
) -> Path | None:
    if explicit_context_db is not None:
        return explicit_context_db.expanduser().resolve()
    if target is not None and target.context_db:
        return _resolve_existing_file(
            target.context_db,
            config_path=config_path,
            kind="Benchmark context DB",
        )
    if len(db_paths) == 1:
        return db_paths[0]
    return None


def _resolve_graph_dataset(
    explicit_graph_dataset: str | None,
    *,
    context_db: Path | None,
    target: BenchmarkTargetConfig | None,
) -> str | None:
    if explicit_graph_dataset and explicit_graph_dataset.strip():
        return explicit_graph_dataset.strip()
    if target is not None and target.graph_dataset and target.graph_dataset.strip():
        return target.graph_dataset.strip()
    if context_db is not None:
        return context_db.stem
    return None


def _resolve_existing_file(
    value: str | Path,
    *,
    config_path: str | None,
    kind: str,
) -> Path:
    candidate = _resolve_path_like(value, config_path=config_path)
    if not candidate.is_file():
        raise FileNotFoundError(f"{kind} not found: {candidate}")
    return candidate


def _resolve_path_like(
    value: str | Path,
    *,
    config_path: str | None,
) -> Path:
    candidate = Path(value).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    if config_path is not None:
        return (Path(config_path).expanduser().resolve().parent / candidate).resolve()
    return candidate.resolve()


def _list_or_empty(
    source: BenchmarkPresetConfig | ExperimentConfig | None,
    field_name: str,
) -> list[str]:
    if source is None:
        return []
    value = getattr(source, field_name, [])
    if not value:
        return []
    return [item.strip() for item in value if item and item.strip()]


def _resolve_output_dir(output_dir: Path | None, run_id: str) -> Path:
    if output_dir is not None:
        return output_dir.expanduser().resolve()
    return (Path("output") / "benchmarks" / run_id).resolve()


def _serialize_suite_report(
    config: BenchmarkCliConfig,
    dataset: BenchmarkDataset,
    run_id: str,
    logfire_status: LogfireStatus,
    entries: list[BenchmarkSuiteEntry],
    runs_rows: list[dict[str, Any]],
    case_group_rows: list[dict[str, Any]],
    leaderboard_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "experiment_name": config.experiment_name,
        "dataset_name": dataset.dataset_name,
        "dataset_path": str(config.dataset_path),
        "benchmark_metadata": _build_suite_metadata(config, dataset),
        "trace_requested": config.trace,
        "logfire": {
            "enabled": logfire_status.enabled,
            "cloud_sync": logfire_status.cloud_sync,
            "url": logfire_status.url or None,
        },
        "runs": runs_rows,
        "case_groups": case_group_rows,
        "leaderboard": leaderboard_rows,
        "entries": [
            {
                "combination": asdict(entry.combination),
                "report": _to_jsonable(entry.report),
            }
            for entry in entries
        ],
    }


def _build_run_manifest(
    *,
    config: BenchmarkCliConfig,
    dataset: BenchmarkDataset,
    run_id: str,
    logfire_status: LogfireStatus,
    entries: list[BenchmarkSuiteEntry],
    report_path: Path,
    runs_path: Path,
    case_groups_path: Path,
    leaderboard_path: Path,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "experiment_name": config.experiment_name,
        "dataset_name": dataset.dataset_name,
        "dataset_path": str(config.dataset_path),
        "benchmark_metadata": _build_suite_metadata(config, dataset),
        "trace_requested": config.trace,
        "logfire": {
            "enabled": logfire_status.enabled,
            "cloud_sync": logfire_status.cloud_sync,
            "url": logfire_status.url or None,
        },
        "combination_count": len(entries),
        "case_count": len(dataset.cases),
        "repeat": config.repeat,
        "max_concurrency": config.max_concurrency,
        "report_path": str(report_path),
        "runs_path": str(runs_path),
        "case_groups_path": str(case_groups_path),
        "leaderboard_path": str(leaderboard_path),
        "reports": [
            {
                "router_profile": entry.combination.router_profile,
                "agent_profile": entry.combination.agent_profile,
                "prompt_strategy": entry.combination.prompt_strategy,
                "graph_orchestrator": entry.combination.graph_orchestrator,
                "report_name": entry.report.name,
                "trace_id": entry.report.trace_id,
                "span_id": entry.report.span_id,
                "case_count": len(entry.report.cases),
                "failure_count": len(entry.report.failures),
            }
            for entry in entries
        ],
    }


def _build_entry_report_metadata(
    *,
    config: BenchmarkCliConfig,
    dataset: BenchmarkDataset,
    logfire_status: LogfireStatus,
) -> dict[str, Any]:
    return {
        "benchmark_experiment_name": config.experiment_name,
        "dataset_name": dataset.dataset_name,
        "dataset_path": str(config.dataset_path),
        "answer_judge_model": config.answer_judge_model,
        "tag_filter": list(config.tags),
        "case_id_cutoff": config.case_id,
        "trace_requested": config.trace,
        "logfire_enabled": logfire_status.enabled,
        "logfire_cloud_sync": logfire_status.cloud_sync,
        "logfire_url": logfire_status.url or None,
    }


def _build_suite_metadata(
    config: BenchmarkCliConfig,
    dataset: BenchmarkDataset,
) -> dict[str, Any]:
    return {
        "dataset_name": dataset.dataset_name,
        "dataset_path": str(config.dataset_path),
        "db_paths": [str(path) for path in config.db_paths],
        "graph_dataset": config.graph_dataset,
        "context_db": str(config.context_db) if config.context_db is not None else None,
        "config_path": config.config_path,
        "router_profiles": _unique_preserving_order(
            combination.router_profile for combination in config.combinations
        ),
        "agent_profiles": _unique_preserving_order(
            combination.agent_profile for combination in config.combinations
        ),
        "prompt_strategies": _unique_preserving_order(
            combination.prompt_strategy for combination in config.combinations
        ),
        "graph_orchestrators": _unique_preserving_order(
            combination.graph_orchestrator for combination in config.combinations
        ),
        "answer_judge_model": config.answer_judge_model,
        "tag_filter": list(config.tags),
        "case_id_cutoff": config.case_id,
        "repeat": config.repeat,
        "max_concurrency": config.max_concurrency,
    }


def _unique_preserving_order(values: Iterable[str | None]) -> list[str | None]:
    ordered: list[str | None] = []
    for value in values:
        if value not in ordered:
            ordered.append(value)
    return ordered


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _to_jsonable(value: object) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, BenchmarkCase):
        return value.model_dump()
    if hasattr(value, "model_dump") and callable(value.model_dump):
        return _to_jsonable(value.model_dump())
    if is_dataclass(value):
        return {
            field_info.name: _to_jsonable(getattr(value, field_info.name))
            for field_info in fields(value)
        }
    report_like_fields = (
        "name",
        "cases",
        "failures",
        "analyses",
        "report_evaluator_failures",
        "experiment_metadata",
        "trace_id",
        "span_id",
    )
    if any(hasattr(value, field_name) for field_name in report_like_fields):
        return {
            field_name: _to_jsonable(getattr(value, field_name))
            for field_name in report_like_fields
            if hasattr(value, field_name)
        }
    if hasattr(value, "__dict__"):
        return {
            str(key): _to_jsonable(item)
            for key, item in vars(value).items()
            if not str(key).startswith("_")
        }
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_jsonable(item) for item in value]
    return str(value)
