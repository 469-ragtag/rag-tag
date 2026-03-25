from __future__ import annotations

from dataclasses import dataclass

from rag_tag.evals.benchmark import BenchmarkCombination, BenchmarkSuiteEntry
from rag_tag.evals.reporting import (
    build_case_groups_rows,
    build_leaderboard_rows,
    build_runs_rows,
)
from rag_tag.evals.task_runner import BenchmarkTaskResult, BenchmarkUsage


@dataclass(frozen=True)
class _EvalResult:
    value: object
    reason: str | None = None


@dataclass(frozen=True)
class _Case:
    name: str
    inputs: object
    metadata: dict[str, object] | None
    expected_output: object | None
    output: BenchmarkTaskResult
    metrics: dict[str, float | int]
    attributes: dict[str, object]
    scores: dict[str, _EvalResult]
    labels: dict[str, _EvalResult]
    assertions: dict[str, _EvalResult]
    task_duration: float
    total_duration: float
    source_case_name: str | None = None
    trace_id: str | None = None
    span_id: str | None = None
    evaluator_failures: list[object] | None = None


@dataclass(frozen=True)
class _Failure:
    name: str
    inputs: object
    metadata: dict[str, object] | None
    expected_output: object | None
    error_message: str
    error_stacktrace: str
    source_case_name: str | None = None
    trace_id: str | None = None
    span_id: str | None = None


class _Report:
    def __init__(
        self, *, name: str, cases: list[_Case], failures: list[_Failure]
    ) -> None:
        self.name = name
        self.cases = cases
        self.failures = failures
        self.analyses = []
        self.report_evaluator_failures = []
        self.experiment_metadata = {
            "benchmark_experiment_name": "benchmark-e2e-v1",
            "dataset_name": "benchmark_cases_v1",
        }
        self.trace_id = "report-trace"
        self.span_id = "report-span"


def _task_result(
    *,
    case_id: str,
    question: str,
    selected_route: str,
    duration_ms: float,
    usage: BenchmarkUsage,
) -> BenchmarkTaskResult:
    return BenchmarkTaskResult(
        case_id=case_id,
        question=question,
        expected_route="graph" if "adjacent" in question.lower() else "sql",
        selected_route=selected_route,
        decision_reason="ok",
        router_profile="router-a",
        agent_profile="agent-a",
        prompt_strategy="baseline",
        graph_orchestrator=None,
        config_path=None,
        graph_dataset="model",
        context_db="/tmp/model.db",
        db_paths=["/tmp/model.db"],
        answer=f"answer for {case_id}",
        warning=None,
        error=None,
        data={"case_id": case_id},
        sql=None,
        duration_ms=duration_ms,
        had_error=False,
        had_warning=False,
        has_data=True,
        usage=usage,
    )


def test_build_runs_rows_includes_case_and_failure_records() -> None:
    report = _Report(
        name="benchmark-e2e-v1__router-a__agent-a__baseline",
        cases=[
            _Case(
                name="q001 [1/2]",
                inputs=type("Inputs", (), {"question": "Which rooms are adjacent?"})(),
                metadata={"case_id": "q001", "expected_route": "graph"},
                expected_output=None,
                output=_task_result(
                    case_id="q001",
                    question="Which rooms are adjacent?",
                    selected_route="graph",
                    duration_ms=100.0,
                    usage=BenchmarkUsage(
                        input_tokens=10,
                        output_tokens=5,
                        total_tokens=15,
                        usage_available=True,
                    ),
                ),
                metrics={},
                attributes={},
                scores={"LLMJudge": _EvalResult(0.8, "grounded")},
                labels={},
                assertions={
                    "RouteMatchesExpected": _EvalResult(True, None),
                    "NoExecutionError": _EvalResult(True, None),
                    "MaxDuration": _EvalResult(True, None),
                },
                task_duration=0.1,
                total_duration=0.1,
                source_case_name="q001",
                trace_id="case-trace-1",
                span_id="case-span-1",
            )
        ],
        failures=[
            _Failure(
                name="q002 [1/2]",
                inputs=type("Inputs", (), {"question": "How many doors?"})(),
                metadata={"case_id": "q002", "expected_route": "sql"},
                expected_output=None,
                error_message="task crashed",
                error_stacktrace="traceback",
                source_case_name="q002",
                trace_id="failure-trace-1",
                span_id="failure-span-1",
            )
        ],
    )
    entry = BenchmarkSuiteEntry(
        combination=BenchmarkCombination(
            router_profile="router-a",
            agent_profile="agent-a",
            prompt_strategy="baseline",
        ),
        report=report,
    )

    rows = build_runs_rows([entry])

    assert len(rows) == 2
    assert rows[0]["case_id"] == "q001"
    assert rows[0]["repeat_index"] == 1
    assert rows[0]["repeat_total"] == 2
    assert rows[0]["judge_score"] == 0.8
    assert rows[0]["trace_id"] == "case-trace-1"
    assert rows[1]["case_id"] == "q002"
    assert rows[1]["error"] == "task crashed"
    assert rows[1]["no_error_assertion"] is False


def test_build_case_groups_and_leaderboard_rows_aggregate_metrics() -> None:
    report = _Report(
        name="benchmark-e2e-v1__router-a__agent-a__baseline",
        cases=[
            _Case(
                name="q001 [1/2]",
                inputs=type("Inputs", (), {"question": "Which rooms are adjacent?"})(),
                metadata={"case_id": "q001", "expected_route": "graph"},
                expected_output=None,
                output=_task_result(
                    case_id="q001",
                    question="Which rooms are adjacent?",
                    selected_route="graph",
                    duration_ms=100.0,
                    usage=BenchmarkUsage(
                        input_tokens=10,
                        output_tokens=5,
                        total_tokens=15,
                        usage_available=True,
                    ),
                ),
                metrics={},
                attributes={},
                scores={"LLMJudge": _EvalResult(0.8, "good")},
                labels={},
                assertions={
                    "RouteMatchesExpected": _EvalResult(True, None),
                    "NoExecutionError": _EvalResult(True, None),
                    "MaxDuration": _EvalResult(True, None),
                },
                task_duration=0.1,
                total_duration=0.1,
                source_case_name="q001",
            ),
            _Case(
                name="q001 [2/2]",
                inputs=type("Inputs", (), {"question": "Which rooms are adjacent?"})(),
                metadata={"case_id": "q001", "expected_route": "graph"},
                expected_output=None,
                output=_task_result(
                    case_id="q001",
                    question="Which rooms are adjacent?",
                    selected_route="sql",
                    duration_ms=180.0,
                    usage=BenchmarkUsage(),
                ),
                metrics={},
                attributes={},
                scores={"LLMJudge": _EvalResult(0.4, "weak")},
                labels={},
                assertions={
                    "RouteMatchesExpected": _EvalResult(False, None),
                    "NoExecutionError": _EvalResult(True, None),
                    "MaxDuration": _EvalResult(False, None),
                },
                task_duration=0.18,
                total_duration=0.18,
                source_case_name="q001",
            ),
        ],
        failures=[],
    )
    entry = BenchmarkSuiteEntry(
        combination=BenchmarkCombination(
            router_profile="router-a",
            agent_profile="agent-a",
            prompt_strategy="baseline",
        ),
        report=report,
    )

    case_group_rows = build_case_groups_rows([entry])
    leaderboard_rows = build_leaderboard_rows([entry], repeat=2)

    assert len(case_group_rows) == 1
    assert case_group_rows[0]["run_count"] == 2
    assert case_group_rows[0]["route_accuracy"] == 0.5
    assert case_group_rows[0]["duration_pass_rate"] == 0.5
    assert case_group_rows[0]["answer_score_avg"] == 0.6
    assert case_group_rows[0]["avg_duration_ms"] == 140.0
    assert case_group_rows[0]["p95_duration_ms"] == 180.0
    assert case_group_rows[0]["sum_total_tokens"] == 15
    assert case_group_rows[0]["token_coverage_rate"] == 0.5

    assert len(leaderboard_rows) == 1
    assert leaderboard_rows[0]["case_count"] == 2
    assert leaderboard_rows[0]["route_accuracy"] == 0.5
    assert leaderboard_rows[0]["answer_score_avg"] == 0.6
    assert leaderboard_rows[0]["sum_total_tokens"] == 15
