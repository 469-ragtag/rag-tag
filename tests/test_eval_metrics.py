from __future__ import annotations

from rag_tag.evals.metrics import aggregate_benchmark_usage
from rag_tag.evals.task_runner import BenchmarkTaskResult, BenchmarkUsage


def test_aggregate_benchmark_usage_handles_partial_coverage() -> None:
    aggregate = aggregate_benchmark_usage(
        [
            BenchmarkUsage(
                input_tokens=10,
                output_tokens=4,
                total_tokens=14,
                usage_available=True,
            ),
            BenchmarkUsage(),
            BenchmarkUsage(
                input_tokens=6,
                output_tokens=3,
                total_tokens=9,
                usage_available=True,
            ),
        ]
    )

    assert aggregate.sum_input_tokens == 16
    assert aggregate.sum_output_tokens == 7
    assert aggregate.sum_total_tokens == 23
    assert aggregate.avg_input_tokens == 8.0
    assert aggregate.avg_output_tokens == 3.5
    assert aggregate.avg_total_tokens == 11.5
    assert aggregate.token_coverage_rate == 2 / 3


def test_aggregate_benchmark_usage_accepts_task_results() -> None:
    result = BenchmarkTaskResult(
        case_id="q001",
        question="How many walls?",
        expected_route="sql",
        selected_route="sql",
        decision_reason="count",
        router_profile="router-a",
        agent_profile="agent-a",
        prompt_strategy="baseline",
        graph_orchestrator=None,
        config_path=None,
        graph_dataset="model",
        context_db=None,
        db_paths=["/tmp/model.db"],
        answer="Found 12 walls.",
        warning=None,
        error=None,
        data={"count": 12},
        sql="select count(*)",
        duration_ms=12.5,
        had_error=False,
        had_warning=False,
        has_data=True,
        usage=BenchmarkUsage(
            input_tokens=21,
            output_tokens=5,
            total_tokens=26,
            usage_available=True,
        ),
    )

    aggregate = aggregate_benchmark_usage([result])

    assert aggregate.sum_total_tokens == 26
    assert aggregate.avg_total_tokens == 26.0
    assert aggregate.token_coverage_rate == 1.0


def test_aggregate_benchmark_usage_returns_empty_metrics_without_usage() -> None:
    aggregate = aggregate_benchmark_usage([BenchmarkUsage(), BenchmarkUsage()])

    assert aggregate.avg_input_tokens is None
    assert aggregate.avg_output_tokens is None
    assert aggregate.avg_total_tokens is None
    assert aggregate.sum_input_tokens is None
    assert aggregate.sum_output_tokens is None
    assert aggregate.sum_total_tokens is None
    assert aggregate.token_coverage_rate == 0.0
