from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from rag_tag.config import AppConfig
from rag_tag.evals.benchmark import (
    BenchmarkCliConfig,
    BenchmarkCombination,
    build_benchmark_cli_config,
    expand_benchmark_matrix,
    load_benchmark_dataset_with_filters,
    run_benchmark_suite,
)
from rag_tag.evals.evaluators import DEFAULT_ANSWER_JUDGE_MODEL
from rag_tag.observability import LogfireStatus


def test_expand_benchmark_matrix_uses_fallbacks_when_lists_are_empty() -> None:
    combinations = expand_benchmark_matrix(
        router_profiles=[],
        agent_profiles=[],
        prompt_strategies=[],
        graph_orchestrators=[],
        fallback_router_profile="router-default",
        fallback_agent_profile="agent-default",
        fallback_graph_orchestrator="pydanticai",
    )

    assert combinations == [
        BenchmarkCombination(
            router_profile="router-default",
            agent_profile="agent-default",
            prompt_strategy="baseline",
            graph_orchestrator="pydanticai",
        )
    ]


def test_expand_benchmark_matrix_includes_orchestrator_dimension() -> None:
    combinations = expand_benchmark_matrix(
        router_profiles=["router-a"],
        agent_profiles=["agent-a"],
        prompt_strategies=["baseline", "strict-grounded"],
        graph_orchestrators=["pydanticai", "langgraph"],
        fallback_router_profile=None,
        fallback_agent_profile=None,
        fallback_graph_orchestrator="pydanticai",
    )

    assert combinations == [
        BenchmarkCombination(
            router_profile="router-a",
            agent_profile="agent-a",
            prompt_strategy="baseline",
            graph_orchestrator="pydanticai",
        ),
        BenchmarkCombination(
            router_profile="router-a",
            agent_profile="agent-a",
            prompt_strategy="baseline",
            graph_orchestrator="langgraph",
        ),
        BenchmarkCombination(
            router_profile="router-a",
            agent_profile="agent-a",
            prompt_strategy="strict-grounded",
            graph_orchestrator="pydanticai",
        ),
        BenchmarkCombination(
            router_profile="router-a",
            agent_profile="agent-a",
            prompt_strategy="strict-grounded",
            graph_orchestrator="langgraph",
        ),
    ]


def test_expand_benchmark_matrix_rejects_removed_decompose_strategy() -> None:
    with pytest.raises(ValueError, match="Unsupported benchmark prompt strategy"):
        expand_benchmark_matrix(
            router_profiles=["router-a"],
            agent_profiles=["agent-a"],
            prompt_strategies=["decompose"],
            graph_orchestrators=["pydanticai"],
            fallback_router_profile=None,
            fallback_agent_profile=None,
            fallback_graph_orchestrator="pydanticai",
        )


def test_run_benchmark_suite_skips_logfire_setup_when_trace_is_disabled(
    tmp_path: Path,
    monkeypatch,
) -> None:
    dataset_path = tmp_path / "benchmark.yaml"
    dataset_path.write_text(
        "dataset_name: benchmark_cases_v1\n"
        "cases:\n"
        "  - id: q001\n"
        "    question: How many doors are there?\n"
        "    expected_route: sql\n",
        encoding="utf-8",
    )

    captured_enabled: list[bool] = []

    def fake_setup_logfire(*, enabled: bool, console: bool) -> LogfireStatus:
        del console
        captured_enabled.append(enabled)
        return LogfireStatus(enabled=False, cloud_sync=False, url="")

    monkeypatch.setattr("rag_tag.evals.benchmark.setup_logfire", fake_setup_logfire)

    async def fake_evaluate_benchmark_dataset_async(
        dataset, *, experiment, experiment_name
    ):
        return type(
            "Report",
            (),
            {
                "name": experiment_name,
                "cases": [],
                "failures": [],
                "analyses": [],
                "report_evaluator_failures": [],
                "experiment_metadata": experiment.report_metadata,
                "trace_id": None,
                "span_id": None,
            },
        )()

    monkeypatch.setattr(
        "rag_tag.evals.benchmark.evaluate_benchmark_dataset_async",
        fake_evaluate_benchmark_dataset_async,
    )

    result = run_benchmark_suite(
        BenchmarkCliConfig(
            experiment_name="benchmark-e2e-v1",
            dataset_path=dataset_path,
            db_paths=[tmp_path / "model.db"],
            output_dir=tmp_path / "artifacts",
            combinations=[
                BenchmarkCombination(
                    router_profile="router-a",
                    agent_profile="agent-a",
                    prompt_strategy="baseline",
                    graph_orchestrator="pydanticai",
                )
            ],
            trace=False,
        )
    )

    assert captured_enabled == [False]
    assert result.logfire_status.enabled is False
    assert result.report_path.is_file()
    assert result.manifest_path.is_file()
    assert (result.output_dir / "runs.csv").is_file()
    assert (result.output_dir / "case_groups.csv").is_file()
    assert (result.output_dir / "leaderboard.csv").is_file()


def test_run_benchmark_suite_preserves_trace_metadata_when_enabled(
    tmp_path: Path,
    monkeypatch,
) -> None:
    dataset_path = tmp_path / "benchmark.yaml"
    dataset_path.write_text(
        "dataset_name: benchmark_cases_v1\n"
        "cases:\n"
        "  - id: q001\n"
        "    question: Which rooms are adjacent to the kitchen?\n"
        "    expected_route: graph\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "rag_tag.evals.benchmark.setup_logfire",
        lambda *, enabled, console: LogfireStatus(
            enabled=enabled,
            cloud_sync=True,
            url="https://logfire.pydantic.dev",
        ),
    )

    async def fake_evaluate_benchmark_dataset_async(
        dataset, *, experiment, experiment_name
    ):
        return type(
            "Report",
            (),
            {
                "name": experiment_name,
                "cases": [],
                "failures": [],
                "analyses": [],
                "report_evaluator_failures": [],
                "experiment_metadata": {
                    **(experiment.report_metadata or {}),
                    "router_profile": experiment.router_profile,
                    "agent_profile": experiment.agent_profile,
                    "prompt_strategy": experiment.prompt_strategy,
                    "graph_orchestrator": experiment.graph_orchestrator,
                },
                "trace_id": "trace-123",
                "span_id": "span-456",
            },
        )()

    monkeypatch.setattr(
        "rag_tag.evals.benchmark.evaluate_benchmark_dataset_async",
        fake_evaluate_benchmark_dataset_async,
    )

    result = run_benchmark_suite(
        BenchmarkCliConfig(
            experiment_name="benchmark-e2e-v1",
            dataset_path=dataset_path,
            db_paths=[tmp_path / "model.db"],
            output_dir=tmp_path / "artifacts",
            combinations=[
                BenchmarkCombination(
                    router_profile="router-a",
                    agent_profile="agent-a",
                    prompt_strategy="strict-grounded",
                    graph_orchestrator="langgraph",
                )
            ],
            trace=True,
        )
    )

    payload = json.loads(result.report_path.read_text(encoding="utf-8"))
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))

    assert payload["trace_requested"] is True
    assert payload["logfire"]["enabled"] is True
    assert payload["benchmark_metadata"]["db_paths"] == [str(tmp_path / "model.db")]
    assert payload["benchmark_metadata"]["graph_dataset"] is None
    assert payload["benchmark_metadata"]["router_profiles"] == ["router-a"]
    assert payload["benchmark_metadata"]["agent_profiles"] == ["agent-a"]
    assert payload["benchmark_metadata"]["prompt_strategies"] == ["strict-grounded"]
    assert payload["benchmark_metadata"]["graph_orchestrators"] == ["langgraph"]
    assert payload["entries"][0]["report"]["trace_id"] == "trace-123"
    assert payload["entries"][0]["report"]["span_id"] == "span-456"
    assert payload["entries"][0]["report"]["experiment_metadata"][
        "dataset_path"
    ] == str(dataset_path)
    assert (
        payload["entries"][0]["report"]["experiment_metadata"]["answer_judge_model"]
        == DEFAULT_ANSWER_JUDGE_MODEL
    )
    assert payload["entries"][0]["report"]["experiment_metadata"]["tag_filter"] == []
    assert payload["entries"][0]["combination"]["graph_orchestrator"] == "langgraph"
    assert isinstance(payload["leaderboard"], list)
    assert isinstance(payload["case_groups"], list)
    assert isinstance(payload["runs"], list)
    assert manifest["benchmark_metadata"]["repeat"] == 1
    assert manifest["benchmark_metadata"]["max_concurrency"] is None
    assert manifest["reports"][0]["graph_orchestrator"] == "langgraph"
    assert manifest["reports"][0]["trace_id"] == "trace-123"
    assert manifest["reports"][0]["span_id"] == "span-456"
    assert manifest["leaderboard_path"].endswith("leaderboard.csv")
    assert manifest["case_groups_path"].endswith("case_groups.csv")
    assert manifest["runs_path"].endswith("runs.csv")


def test_build_benchmark_cli_config_uses_experiment_defaults_and_tag_filter(
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "benchmark.yaml"
    dataset_path.write_text(
        "dataset_name: benchmark_cases_v1\n"
        "cases:\n"
        "  - id: q001\n"
        "    question: How many doors?\n"
        "    expected_route: sql\n",
        encoding="utf-8",
    )

    config = AppConfig.model_validate(
        {
            "profiles": {
                "router-a": {"model": "google-gla:gemini-2.5-flash"},
                "agent-a": {"model": "cohere:command-a-03-2025"},
            },
            "experiments": {
                "benchmark-e2e-v1": {
                    "description": "trace benchmark",
                    "questions_file": str(dataset_path),
                    "router_profile": "router-a",
                    "agent_profile": "agent-a",
                    "prompt_strategies": ["baseline"],
                    "graph_orchestrators": ["langgraph"],
                    "repeat": 2,
                    "max_concurrency": 1,
                    "answer_judge_model": "google-gla:gemini-2.5-flash",
                    "tags": ["sql"],
                }
            },
        }
    )

    cli_config = build_benchmark_cli_config(
        config=config,
        experiment_name="benchmark-e2e-v1",
        preset_name=None,
        target_name=None,
        questions_file=None,
        router_profiles=None,
        agent_profiles=None,
        prompt_strategies=None,
        orchestrators=None,
        tags=None,
        repeat=None,
        max_concurrency=None,
        db_paths=[tmp_path / "model.db"],
        graph_dataset="model",
        context_db=tmp_path / "model.db",
        config_path=None,
        trace=True,
    )

    assert cli_config.repeat == 2
    assert cli_config.max_concurrency == 1
    assert cli_config.answer_judge_model == "google-gla:gemini-2.5-flash"
    assert cli_config.tags == ["sql"]
    assert cli_config.trace is True
    assert cli_config.combinations == [
        BenchmarkCombination(
            router_profile="router-a",
            agent_profile="agent-a",
            prompt_strategy="baseline",
            graph_orchestrator="langgraph",
        )
    ]


def test_load_benchmark_dataset_with_filters_applies_case_id_before_tags(
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "benchmark.yaml"
    dataset_path.write_text(
        "schema_version: 2\n"
        "dataset_name: benchmark_cases_v2\n"
        "cases:\n"
        "  - id: q001\n"
        "    question: How many walls?\n"
        "    expected_route: sql\n"
        "    answer:\n"
        "      canonical: There are 4 walls.\n"
        "    tags: [sql]\n"
        "  - id: q002\n"
        "    question: Which rooms are adjacent?\n"
        "    expected_route: graph\n"
        "    answer:\n"
        "      canonical: The living room is adjacent to the entry hall.\n"
        "    tags: [graph]\n"
        "  - id: q003\n"
        "    question: Which storey contains the chimney?\n"
        "    expected_route: graph\n"
        "    answer:\n"
        "      canonical: The chimney is on the groundfloor.\n"
        "    tags: [graph]\n",
        encoding="utf-8",
    )

    filtered = load_benchmark_dataset_with_filters(
        dataset_path,
        tags=["graph"],
        case_id="q002",
    )

    assert filtered.schema_version == 2
    assert filtered.dataset_name == "benchmark_cases_v2"
    assert [case.id for case in filtered.cases] == ["q002"]


def test_load_benchmark_dataset_with_filters_rejects_unknown_case_id(
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "benchmark.yaml"
    dataset_path.write_text(
        "schema_version: 2\n"
        "dataset_name: benchmark_cases_v2\n"
        "cases:\n"
        "  - id: q001\n"
        "    question: How many walls?\n"
        "    expected_route: sql\n"
        "    answer:\n"
        "      canonical: There are 4 walls.\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="case-id cutoff was not found: q020"):
        load_benchmark_dataset_with_filters(dataset_path, case_id="q020")


def test_build_benchmark_cli_config_resolves_relative_dataset_from_config_path(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    dataset_path = project_root / "evals" / "benchmark_cases_v1.yaml"
    dataset_path.parent.mkdir()
    dataset_path.write_text(
        "dataset_name: benchmark_cases_v1\n"
        "cases:\n"
        "  - id: q001\n"
        "    question: How many doors?\n"
        "    expected_route: sql\n",
        encoding="utf-8",
    )
    config_path = project_root / "config.yaml"
    config_path.write_text("experiments: {}\n", encoding="utf-8")
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    monkeypatch.chdir(outside_dir)

    config = AppConfig.model_validate(
        {
            "experiments": {
                "benchmark-e2e-v1": {
                    "questions_file": "evals/benchmark_cases_v1.yaml",
                    "prompt_strategies": ["baseline"],
                    "graph_orchestrators": ["pydanticai"],
                }
            }
        }
    )

    cli_config = build_benchmark_cli_config(
        config=config,
        experiment_name="benchmark-e2e-v1",
        preset_name=None,
        target_name=None,
        questions_file=None,
        router_profiles=None,
        agent_profiles=None,
        prompt_strategies=None,
        orchestrators=None,
        tags=None,
        repeat=None,
        max_concurrency=None,
        db_paths=[tmp_path / "model.db"],
        graph_dataset="model",
        context_db=tmp_path / "model.db",
        config_path=str(config_path),
    )

    assert cli_config.dataset_path == dataset_path.resolve()
    assert cli_config.answer_judge_model == DEFAULT_ANSWER_JUDGE_MODEL
    assert cli_config.combinations[0].graph_orchestrator == "pydanticai"


def test_build_benchmark_cli_config_resolves_preset_target_bundle(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project_root = tmp_path / "project"
    evals_dir = project_root / "evals"
    output_dir = project_root / "output"
    evals_dir.mkdir(parents=True)
    output_dir.mkdir()
    dataset_path = evals_dir / "benchmark_cases_v1.yaml"
    dataset_path.write_text(
        "dataset_name: benchmark_cases_v1\n"
        "cases:\n"
        "  - id: q001\n"
        "    question: How many doors?\n"
        "    expected_route: sql\n",
        encoding="utf-8",
    )
    db_path = output_dir / "Building-Architecture.db"
    db_path.write_text("", encoding="utf-8")
    config_path = project_root / "config.yaml"
    config_path.write_text("benchmark_targets: {}\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    config = AppConfig.model_validate(
        {
            "benchmark_targets": {
                "building-architecture": {
                    "questions_file": "evals/benchmark_cases_v1.yaml",
                    "db_paths": ["output/Building-Architecture.db"],
                    "graph_dataset": "Building-Architecture",
                }
            },
            "benchmark_presets": {
                "smoke": {
                    "target": "building-architecture",
                    "router_profiles": ["router-a"],
                    "agent_profiles": ["agent-a"],
                    "prompt_strategies": ["baseline"],
                    "graph_orchestrators": ["langgraph"],
                    "tags": ["sql"],
                    "repeat": 2,
                    "max_concurrency": 1,
                    "answer_judge_model": "judge-model",
                }
            },
        }
    )

    cli_config = build_benchmark_cli_config(
        config=config,
        experiment_name=None,
        preset_name="smoke",
        target_name=None,
        questions_file=None,
        router_profiles=None,
        agent_profiles=None,
        prompt_strategies=None,
        orchestrators=None,
        tags=None,
        repeat=None,
        max_concurrency=None,
        db_paths=None,
        graph_dataset=None,
        context_db=None,
        config_path=str(config_path),
    )

    assert cli_config.experiment_name == "smoke"
    assert cli_config.dataset_path == dataset_path.resolve()
    assert cli_config.db_paths == [db_path.resolve()]
    assert cli_config.context_db == db_path.resolve()
    assert cli_config.graph_dataset == "Building-Architecture"
    assert cli_config.tags == ["sql"]
    assert cli_config.repeat == 2
    assert cli_config.max_concurrency == 1
    assert cli_config.answer_judge_model == "judge-model"
    assert cli_config.combinations == [
        BenchmarkCombination(
            router_profile="router-a",
            agent_profile="agent-a",
            prompt_strategy="baseline",
            graph_orchestrator="langgraph",
        )
    ]


def test_build_benchmark_cli_config_cli_overrides_beat_preset_target_and_experiment(
    tmp_path: Path,
) -> None:
    dataset_from_cli = tmp_path / "cli.yaml"
    dataset_from_cli.write_text(
        "dataset_name: benchmark_cases_v1\n"
        "cases:\n"
        "  - id: q001\n"
        "    question: How many doors?\n"
        "    expected_route: sql\n",
        encoding="utf-8",
    )
    db_from_cli = tmp_path / "cli.db"
    db_from_cli.write_text("", encoding="utf-8")

    config = AppConfig.model_validate(
        {
            "defaults": {
                "router_profile": "router-default",
                "agent_profile": "agent-default",
                "graph_orchestrator": "pydanticai",
            },
            "benchmark_targets": {
                "target-a": {
                    "questions_file": str(tmp_path / "target-a.yaml"),
                    "db_paths": [str(tmp_path / "target-a.db")],
                    "graph_dataset": "target-a",
                },
                "target-b": {
                    "questions_file": str(tmp_path / "target-b.yaml"),
                    "db_paths": [str(tmp_path / "target-b.db")],
                    "graph_dataset": "target-b",
                },
            },
            "benchmark_presets": {
                "full": {
                    "target": "target-a",
                    "router_profiles": ["router-preset"],
                    "agent_profiles": ["agent-preset"],
                    "prompt_strategies": ["strict-grounded"],
                    "graph_orchestrators": ["langgraph"],
                    "tags": ["preset"],
                    "repeat": 2,
                    "max_concurrency": 3,
                    "answer_judge_model": "preset-judge",
                }
            },
            "experiments": {
                "benchmark-e2e-v1": {
                    "questions_file": str(tmp_path / "experiment.yaml"),
                    "router_profile": "router-experiment",
                    "agent_profile": "agent-experiment",
                    "prompt_strategies": ["baseline"],
                    "graph_orchestrators": ["pydanticai"],
                    "tags": ["experiment"],
                    "repeat": 5,
                    "max_concurrency": 4,
                    "answer_judge_model": "experiment-judge",
                }
            },
        }
    )

    for filename in ("target-a.yaml", "target-b.yaml", "experiment.yaml"):
        (tmp_path / filename).write_text(
            "dataset_name: benchmark_cases_v1\n"
            "cases:\n"
            "  - id: q001\n"
            "    question: How many doors?\n"
            "    expected_route: sql\n",
            encoding="utf-8",
        )
    for filename in ("target-a.db", "target-b.db"):
        (tmp_path / filename).write_text("", encoding="utf-8")

    cli_config = build_benchmark_cli_config(
        config=config,
        experiment_name="benchmark-e2e-v1",
        preset_name="full",
        target_name="target-b",
        questions_file=dataset_from_cli,
        router_profiles=["router-cli"],
        agent_profiles=["agent-cli"],
        prompt_strategies=["baseline"],
        orchestrators=["pydanticai"],
        tags=["cli"],
        case_id="q020",
        repeat=7,
        max_concurrency=8,
        db_paths=[db_from_cli],
        graph_dataset="cli-graph",
        context_db=db_from_cli,
        config_path=None,
        answer_judge_model="cli-judge",
    )

    assert cli_config.experiment_name == "full"
    assert cli_config.dataset_path == dataset_from_cli.resolve()
    assert cli_config.db_paths == [db_from_cli.resolve()]
    assert cli_config.context_db == db_from_cli.resolve()
    assert cli_config.graph_dataset == "cli-graph"
    assert cli_config.tags == ["cli"]
    assert cli_config.case_id == "q020"
    assert cli_config.repeat == 7
    assert cli_config.max_concurrency == 8
    assert cli_config.answer_judge_model == "cli-judge"
    assert cli_config.combinations == [
        BenchmarkCombination(
            router_profile="router-cli",
            agent_profile="agent-cli",
            prompt_strategy="baseline",
            graph_orchestrator="pydanticai",
        )
    ]


def test_eval_benchmarks_script_passes_answer_judge_and_orchestrator_overrides(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "eval_benchmarks.py"
    spec = importlib.util.spec_from_file_location("eval_benchmarks_script", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    db_path = tmp_path / "model.db"
    db_path.write_text("", encoding="utf-8")
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        module,
        "_load_app_config",
        lambda config_override_path: (
            __import__("rag_tag.config", fromlist=["AppConfig"]).AppConfig(),
            config_override_path,
        ),
    )

    def fake_build_benchmark_cli_config(**kwargs):
        captured.update(kwargs)
        return BenchmarkCliConfig(
            experiment_name="benchmark-e2e-v1",
            dataset_path=tmp_path / "benchmark.yaml",
            db_paths=[db_path],
            graph_dataset="model",
            context_db=db_path,
            answer_judge_model=kwargs["answer_judge_model"],
            output_dir=tmp_path / "artifacts",
            combinations=[
                BenchmarkCombination(
                    router_profile="router-a",
                    agent_profile="agent-a",
                    prompt_strategy="baseline",
                    graph_orchestrator="langgraph",
                )
            ],
        )

    monkeypatch.setattr(
        module, "build_benchmark_cli_config", fake_build_benchmark_cli_config
    )
    monkeypatch.setattr(
        module,
        "run_benchmark_suite",
        lambda config: _mock_suite_result(tmp_path, config),
    )

    exit_code = module.main(
        [
            "--preset",
            "smoke",
            "--target",
            "building-architecture",
            "--db",
            str(db_path),
            "--answer-judge-model",
            "google-gla:gemini-2.5-flash",
            "--case-id",
            "q020",
            "--orchestrators",
            "langgraph",
        ]
    )

    assert exit_code == 0
    assert captured["preset_name"] == "smoke"
    assert captured["target_name"] == "building-architecture"
    assert captured["answer_judge_model"] == "google-gla:gemini-2.5-flash"
    assert captured["case_id"] == "q020"
    assert captured["orchestrators"] == ["langgraph"]
    capsys.readouterr()


def test_eval_benchmarks_script_main_runs_and_prints_summary(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "eval_benchmarks.py"
    spec = importlib.util.spec_from_file_location("eval_benchmarks_script", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    db_path = tmp_path / "model.db"
    db_path.write_text("", encoding="utf-8")

    monkeypatch.setattr(
        module,
        "_load_app_config",
        lambda config_override_path: (
            __import__("rag_tag.config", fromlist=["AppConfig"]).AppConfig(),
            config_override_path,
        ),
    )
    monkeypatch.setattr(
        module,
        "build_benchmark_cli_config",
        lambda **kwargs: BenchmarkCliConfig(
            experiment_name="benchmark-e2e-v1",
            dataset_path=tmp_path / "benchmark.yaml",
            db_paths=[db_path],
            graph_dataset="model",
            context_db=db_path,
            trace=True,
            output_dir=tmp_path / "artifacts",
            combinations=[
                BenchmarkCombination(
                    router_profile="router-a",
                    agent_profile="agent-a",
                    prompt_strategy="baseline",
                    graph_orchestrator="langgraph",
                )
            ],
        ),
    )
    monkeypatch.setattr(
        module,
        "run_benchmark_suite",
        lambda config: _mock_suite_result(tmp_path, config),
    )

    exit_code = module.main(
        [
            "--questions-file",
            str(tmp_path / "benchmark.yaml"),
            "--db",
            str(db_path),
            "--trace",
        ]
    )

    assert exit_code == 0
    stdout = capsys.readouterr().out
    assert "Benchmark run: benchmark-e2e-v1" in stdout
    assert "Trace requested: yes" in stdout
    assert "1. router-a / agent-a / baseline / langgraph" in stdout
    assert "answer_correct=1" in stdout
    assert "route_correct=0.8" in stdout
    assert "avg_tokens(in/out/total)=30/10/40" in stdout


def _mock_suite_result(tmp_path: Path, config: BenchmarkCliConfig):
    report_path = tmp_path / "artifacts" / "report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(
            {
                "leaderboard": [
                    {
                        "router_profile": "router-a",
                        "agent_profile": "agent-a",
                        "prompt_strategy": "baseline",
                        "graph_orchestrator": "langgraph",
                        "answer_correct_rate": 1.0,
                        "route_correct_rate": 0.8,
                        "avg_input_tokens": 30.0,
                        "avg_output_tokens": 10.0,
                        "avg_total_tokens": 40.0,
                        "avg_duration_ms": 120.0,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    return type(
        "Result",
        (),
        {
            "experiment_name": config.experiment_name,
            "dataset_name": "benchmark_cases_v1",
            "entries": [object()],
            "trace_requested": config.trace,
            "logfire_status": LogfireStatus(
                enabled=True,
                cloud_sync=False,
                url="",
            ),
            "output_dir": config.output_dir,
            "report_path": report_path,
        },
    )()
