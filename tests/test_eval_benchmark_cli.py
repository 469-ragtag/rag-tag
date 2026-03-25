from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from rag_tag.evals.benchmark import (
    BenchmarkCliConfig,
    BenchmarkCombination,
    build_benchmark_cli_config,
    expand_benchmark_matrix,
    run_benchmark_suite,
)
from rag_tag.evals.evaluators import DEFAULT_ANSWER_JUDGE_MODEL
from rag_tag.observability import LogfireStatus


def test_expand_benchmark_matrix_uses_fallbacks_when_lists_are_empty() -> None:
    combinations = expand_benchmark_matrix(
        router_profiles=[],
        agent_profiles=[],
        prompt_strategies=[],
        fallback_router_profile="router-default",
        fallback_agent_profile="agent-default",
    )

    assert combinations == [
        BenchmarkCombination(
            router_profile="router-default",
            agent_profile="agent-default",
            prompt_strategy="baseline",
        )
    ]


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
                )
            ],
            trace=True,
        )
    )

    payload = json.loads(result.report_path.read_text(encoding="utf-8"))
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))

    assert payload["trace_requested"] is True
    assert payload["logfire"]["enabled"] is True
    assert payload["entries"][0]["report"]["trace_id"] == "trace-123"
    assert payload["entries"][0]["report"]["span_id"] == "span-456"
    assert isinstance(payload["leaderboard"], list)
    assert isinstance(payload["case_groups"], list)
    assert isinstance(payload["runs"], list)
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

    from rag_tag.config import AppConfig

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
        questions_file=None,
        router_profiles=None,
        agent_profiles=None,
        prompt_strategies=None,
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
        )
    ]


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

    from rag_tag.config import AppConfig

    config = AppConfig.model_validate(
        {
            "experiments": {
                "benchmark-e2e-v1": {
                    "questions_file": "evals/benchmark_cases_v1.yaml",
                    "prompt_strategies": ["baseline"],
                }
            }
        }
    )

    cli_config = build_benchmark_cli_config(
        config=config,
        experiment_name="benchmark-e2e-v1",
        questions_file=None,
        router_profiles=None,
        agent_profiles=None,
        prompt_strategies=None,
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


def test_eval_benchmarks_script_passes_answer_judge_model_override(
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
            "--questions-file",
            str(tmp_path / "benchmark.yaml"),
            "--db",
            str(db_path),
            "--answer-judge-model",
            "google-gla:gemini-2.5-flash",
        ]
    )

    assert exit_code == 0
    assert captured["answer_judge_model"] == "google-gla:gemini-2.5-flash"
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
    assert "1. router-a / agent-a / baseline" in stdout


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
                        "route_accuracy": 1.0,
                        "answer_score_avg": 0.9,
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
