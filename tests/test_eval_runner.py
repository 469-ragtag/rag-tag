from __future__ import annotations

from pathlib import Path

from pydantic_evals.evaluators import LLMJudge, MaxDuration

from rag_tag.evals import (
    DEFAULT_ANSWER_JUDGE_MODEL,
    DEFAULT_ANSWER_JUDGE_RUBRIC,
    BenchmarkAnswer,
    BenchmarkCase,
    BenchmarkDataset,
    BenchmarkExperimentConfig,
    BenchmarkTaskResult,
    build_eval_dataset,
    evaluate_benchmark_dataset,
)
from rag_tag.evals.task_runner import BenchmarkUsage


def test_build_eval_dataset_adds_case_and_dataset_evaluators() -> None:
    dataset = BenchmarkDataset(
        dataset_name="benchmark_cases_v1",
        cases=[
            BenchmarkCase(
                id="q001",
                question="How many walls are in the model?",
                expected_route="sql",
                answer=BenchmarkAnswer(
                    canonical="There are 4 walls in the model.",
                    judge_notes=["returns a deterministic count"],
                ),
                tags=["sql", "count"],
                max_duration_s=10,
            )
        ],
    )

    eval_dataset = build_eval_dataset(
        dataset,
        include_answer_judge=True,
        answer_judge_model="openai:gpt-5.2",
    )

    assert eval_dataset.name == "benchmark_cases_v1"
    assert len(eval_dataset.cases) == 1
    assert eval_dataset.cases[0].name == "q001"
    assert eval_dataset.cases[0].metadata == {
        "case_id": "q001",
        "expected_route": "sql",
        "answer_judge_target": {
            "case_id": "q001",
            "question": "How many walls are in the model?",
            "expected_route": "sql",
            "answer": {
                "canonical": "There are 4 walls in the model.",
                "acceptable": [],
                "judge_notes": ["returns a deterministic count"],
            },
        },
        "answer": {
            "canonical": "There are 4 walls in the model.",
            "acceptable": [],
            "judge_notes": ["returns a deterministic count"],
        },
        "expected_answer": "There are 4 walls in the model.",
        "reference_points": ["returns a deterministic count"],
        "tags": ["sql", "count"],
        "max_duration_s": 10,
    }
    assert eval_dataset.cases[0].expected_output == {
        "case_id": "q001",
        "question": "How many walls are in the model?",
        "expected_route": "sql",
        "answer": {
            "canonical": "There are 4 walls in the model.",
            "acceptable": [],
            "judge_notes": ["returns a deterministic count"],
        },
    }
    assert len(eval_dataset.cases[0].evaluators) == 1
    assert isinstance(eval_dataset.cases[0].evaluators[0], MaxDuration)
    assert any(
        type(evaluator).__name__ == "RouteMatchesExpected"
        for evaluator in eval_dataset.evaluators
    )
    assert any(
        type(evaluator).__name__ == "NoExecutionError"
        for evaluator in eval_dataset.evaluators
    )
    assert any(isinstance(evaluator, LLMJudge) for evaluator in eval_dataset.evaluators)
    judge = next(
        evaluator
        for evaluator in eval_dataset.evaluators
        if isinstance(evaluator, LLMJudge)
    )
    assert judge.model == "openai:gpt-5.2"
    assert judge.include_expected_output is True
    assert judge.assertion == {
        "evaluation_name": "answer_correct",
        "include_reason": True,
    }
    assert judge.score == {
        "evaluation_name": "judge_score",
        "include_reason": True,
    }
    assert "output.answer" in DEFAULT_ANSWER_JUDGE_RUBRIC
    assert "expected_output.answer.canonical" in DEFAULT_ANSWER_JUDGE_RUBRIC


def test_build_eval_dataset_defaults_answer_judge_to_repo_model() -> None:
    dataset = BenchmarkDataset(
        dataset_name="benchmark_cases_v1",
        cases=[
            BenchmarkCase(
                id="q001",
                question="How many walls are in the model?",
                expected_route="sql",
            )
        ],
    )

    eval_dataset = build_eval_dataset(dataset, include_answer_judge=True)

    judge = next(
        evaluator
        for evaluator in eval_dataset.evaluators
        if isinstance(evaluator, LLMJudge)
    )
    assert judge.model == DEFAULT_ANSWER_JUDGE_MODEL
    assert judge.include_expected_output is True
    assert judge.assertion == {
        "evaluation_name": "answer_correct",
        "include_reason": True,
    }
    assert judge.score == {
        "evaluation_name": "judge_score",
        "include_reason": True,
    }


def test_build_eval_dataset_uses_loop_safe_answer_judge() -> None:
    dataset = BenchmarkDataset(
        dataset_name="benchmark_cases_v1",
        cases=[
            BenchmarkCase(
                id="q001",
                question="How many walls are in the model?",
                expected_route="sql",
            )
        ],
    )

    eval_dataset = build_eval_dataset(dataset, include_answer_judge=True)

    judge = next(
        evaluator
        for evaluator in eval_dataset.evaluators
        if isinstance(evaluator, LLMJudge)
    )
    assert type(judge).__name__ == "LoopSafeLLMJudge"


def test_evaluate_benchmark_dataset_runs_cases_and_repeats(
    monkeypatch,
) -> None:
    dataset = BenchmarkDataset(
        dataset_name="benchmark_cases_v1",
        cases=[
            BenchmarkCase(
                id="q001",
                question="How many walls are in the model?",
                expected_route="sql",
                max_duration_s=10,
            ),
            BenchmarkCase(
                id="q002",
                question="What building storey contains the chimney?",
                expected_route="graph",
                max_duration_s=20,
            ),
        ],
    )

    calls: list[tuple[str, object | None, object | None]] = []

    def fake_run_benchmark_case(
        case: BenchmarkCase,
        *,
        db_paths: list[Path],
        runtime: object | None,
        agent: object | None,
        config_path: str | None = None,
        router_profile: str | None = None,
        agent_profile: str | None = None,
        prompt_strategy: str = "baseline",
        graph_orchestrator: str | None = None,
        graph_dataset: str | None = None,
        context_db: Path | None = None,
        payload_mode: str | None = None,
        strict_sql: bool = False,
        graph_max_steps: int | None = None,
        debug_llm_io: bool = False,
    ):
        del (
            db_paths,
            config_path,
            router_profile,
            agent_profile,
            prompt_strategy,
            graph_orchestrator,
            graph_dataset,
            context_db,
            payload_mode,
            strict_sql,
            graph_max_steps,
            debug_llm_io,
        )
        calls.append((case.id, runtime, agent))
        next_runtime = runtime or object()
        next_agent = agent or object()
        result = BenchmarkTaskResult(
            case_id=case.id,
            question=case.question,
            expected_route=case.expected_route,
            selected_route=case.expected_route,
            decision_reason="ok",
            router_profile="router-alpha",
            agent_profile="agent-beta",
            prompt_strategy="baseline",
            graph_orchestrator=None,
            config_path=None,
            graph_dataset="model",
            context_db=str(Path("/tmp/model.db")),
            db_paths=[str(Path("/tmp/model.db"))],
            answer=f"answer for {case.id}",
            warning=None,
            error=None,
            data={"case_id": case.id},
            sql=None,
            duration_ms=5.0,
            had_error=False,
            had_warning=False,
            has_data=True,
            usage=BenchmarkUsage(),
        )

        class Bundle:
            def __init__(self):
                self.result = result
                self.runtime = next_runtime
                self.agent = next_agent

        return Bundle()

    closed_runtimes: list[object | None] = []

    monkeypatch.setattr(
        "rag_tag.evals.runner.run_benchmark_case",
        fake_run_benchmark_case,
    )
    monkeypatch.setattr(
        "rag_tag.evals.runner.close_runtime",
        lambda runtime: closed_runtimes.append(runtime),
    )
    sleep_calls: list[float] = []

    async def fake_sleep(delay_seconds: float = 20.0) -> None:
        sleep_calls.append(delay_seconds)

    monkeypatch.setattr(
        "rag_tag.evals.runner._sleep_between_benchmark_cases",
        fake_sleep,
    )

    report = evaluate_benchmark_dataset(
        dataset,
        experiment=BenchmarkExperimentConfig(
            db_paths=[Path("/tmp/model.db")],
            include_answer_judge=False,
            repeat=2,
            progress=False,
        ),
        experiment_name="batch3-smoke",
    )

    assert report.name == "batch3-smoke"
    assert len(report.cases) == 4
    assert [case.name for case in report.cases] == [
        "q001 [1/2]",
        "q001 [2/2]",
        "q002 [1/2]",
        "q002 [2/2]",
    ]
    assert calls[0][0] == "q001"
    assert calls[1][0] == "q001"
    assert calls[2][0] == "q002"
    assert calls[3][0] == "q002"
    assert closed_runtimes == [calls[-1][1]]
    assert sleep_calls == [20.0, 20.0, 20.0, 20.0]


def test_evaluate_benchmark_dataset_disables_state_reuse_when_concurrent(
    monkeypatch,
) -> None:
    dataset = BenchmarkDataset(
        dataset_name="benchmark_cases_v1",
        cases=[
            BenchmarkCase(
                id="q001",
                question="How many walls are in the model?",
                expected_route="sql",
            ),
            BenchmarkCase(
                id="q002",
                question="What building storey contains the chimney?",
                expected_route="graph",
            ),
        ],
    )

    calls: list[tuple[str, object | None, object | None]] = []
    closed_runtimes: list[object | None] = []

    def fake_run_benchmark_case(
        case: BenchmarkCase,
        *,
        db_paths: list[Path],
        runtime: object | None,
        agent: object | None,
        config_path: str | None = None,
        router_profile: str | None = None,
        agent_profile: str | None = None,
        prompt_strategy: str = "baseline",
        graph_orchestrator: str | None = None,
        graph_dataset: str | None = None,
        context_db: Path | None = None,
        payload_mode: str | None = None,
        strict_sql: bool = False,
        graph_max_steps: int | None = None,
        debug_llm_io: bool = False,
    ):
        del (
            db_paths,
            config_path,
            router_profile,
            agent_profile,
            prompt_strategy,
            graph_orchestrator,
            graph_dataset,
            context_db,
            payload_mode,
            strict_sql,
            graph_max_steps,
            debug_llm_io,
        )
        calls.append((case.id, runtime, agent))
        result = BenchmarkTaskResult(
            case_id=case.id,
            question=case.question,
            expected_route=case.expected_route,
            selected_route=case.expected_route,
            decision_reason="ok",
            router_profile="router-alpha",
            agent_profile="agent-beta",
            prompt_strategy="baseline",
            graph_orchestrator=None,
            config_path=None,
            graph_dataset="model",
            context_db=str(Path("/tmp/model.db")),
            db_paths=[str(Path("/tmp/model.db"))],
            answer=f"answer for {case.id}",
            warning=None,
            error=None,
            data={"case_id": case.id},
            sql=None,
            duration_ms=5.0,
            had_error=False,
            had_warning=False,
            has_data=True,
            usage=BenchmarkUsage(),
        )

        class Bundle:
            def __init__(self) -> None:
                self.result = result
                self.runtime = object()
                self.agent = object()

        return Bundle()

    monkeypatch.setattr(
        "rag_tag.evals.runner.run_benchmark_case",
        fake_run_benchmark_case,
    )
    monkeypatch.setattr(
        "rag_tag.evals.runner.close_runtime",
        lambda runtime: closed_runtimes.append(runtime),
    )
    sleep_calls: list[float] = []

    async def fake_sleep(delay_seconds: float = 20.0) -> None:
        sleep_calls.append(delay_seconds)

    monkeypatch.setattr(
        "rag_tag.evals.runner._sleep_between_benchmark_cases",
        fake_sleep,
    )

    report = evaluate_benchmark_dataset(
        dataset,
        experiment=BenchmarkExperimentConfig(
            db_paths=[Path("/tmp/model.db")],
            include_answer_judge=False,
            repeat=1,
            progress=False,
            max_concurrency=2,
        ),
    )

    assert report.experiment_metadata is not None
    assert report.experiment_metadata["max_concurrency"] == 1
    assert report.experiment_metadata["requested_max_concurrency"] == 2
    assert report.experiment_metadata["effective_max_concurrency"] == 1
    assert report.experiment_metadata["state_reuse_enabled"] is True
    assert calls[0] == ("q001", None, None)
    assert calls[1][0] == "q002"
    assert calls[1][1] is not None
    assert calls[1][2] is not None
    assert len(closed_runtimes) == 1
    assert closed_runtimes[0] is not None
    assert closed_runtimes[0] is not calls[1][1]
    assert sleep_calls == [20.0, 20.0]
