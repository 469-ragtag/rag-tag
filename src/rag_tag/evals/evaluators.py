"""Custom evaluators for rag-tag benchmark experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic_evals.evaluators import (
    EvaluationReason,
    Evaluator,
    EvaluatorContext,
    LLMJudge,
)

from .dataset import BenchmarkCase
from .task_runner import BenchmarkTaskResult

BenchmarkCaseMetadata = dict[str, Any]
DEFAULT_ANSWER_JUDGE_MODEL = "google-gla:gemini-2.5-flash"

DEFAULT_ANSWER_JUDGE_RUBRIC = """
You are evaluating the quality of an IFC/BIM benchmark answer.

Judge the answer against the benchmark input and output objects provided.

Pass only when:
- the answer addresses the user question directly
- the selected route is appropriate for the benchmark case
- the answer stays grounded in the available evidence or clearly signals uncertainty
- the answer does not fabricate entities, relationships, properties, or counts
- the answer aligns with the case reference points when they are present
""".strip()


def _clear_pydantic_ai_async_http_client_cache() -> None:
    """Clear PydanticAI's shared async HTTP client cache.

    The benchmark task path still uses sync `run_sync(...)` calls for router and
    graph execution, while `LLMJudge` runs asynchronously. PydanticAI's Google
    provider reuses a process-wide cached async HTTP client, which can become
    bound to the event loop created by the sync path. Clearing the cache before
    judge requests forces a fresh async client on the current eval loop.
    """

    try:
        from pydantic_ai._ssrf import cached_async_http_client
    except Exception:
        return

    cached_factory = getattr(
        cached_async_http_client,
        "__globals__",
        {},
    ).get("_cached_async_http_client")
    cache_clear = getattr(cached_factory, "cache_clear", None)
    if callable(cache_clear):
        cache_clear()


@dataclass
class RouteMatchesExpected(
    Evaluator[BenchmarkCase, BenchmarkTaskResult, BenchmarkCaseMetadata]
):
    """Assert that the selected route matches the benchmark expectation."""

    def evaluate(
        self,
        ctx: EvaluatorContext[
            BenchmarkCase, BenchmarkTaskResult, BenchmarkCaseMetadata
        ],
    ) -> bool | EvaluationReason:
        expected_route = ctx.inputs.expected_route
        selected_route = ctx.output.selected_route
        if selected_route == expected_route:
            return True
        return EvaluationReason(
            False,
            (f"Expected route '{expected_route}' but got '{selected_route}'."),
        )


@dataclass
class NoExecutionError(
    Evaluator[BenchmarkCase, BenchmarkTaskResult, BenchmarkCaseMetadata]
):
    """Assert that the benchmark task completed without an execution error."""

    def evaluate(
        self,
        ctx: EvaluatorContext[
            BenchmarkCase, BenchmarkTaskResult, BenchmarkCaseMetadata
        ],
    ) -> bool | EvaluationReason:
        error = ctx.output.error
        if error is None:
            return True
        return EvaluationReason(False, f"Task returned error: {error}")


class LoopSafeLLMJudge(LLMJudge):
    """LLM judge that avoids reusing async clients bound to another loop."""

    async def evaluate(
        self,
        ctx: EvaluatorContext[
            BenchmarkCase, BenchmarkTaskResult, BenchmarkCaseMetadata
        ],
    ) -> Any:
        _clear_pydantic_ai_async_http_client_cache()
        return await super().evaluate(ctx)


def build_default_answer_judge(model: str | None = None) -> LLMJudge:
    """Build the default answer-quality evaluator for benchmark experiments."""

    return LoopSafeLLMJudge(
        rubric=DEFAULT_ANSWER_JUDGE_RUBRIC,
        model=model or DEFAULT_ANSWER_JUDGE_MODEL,
        include_input=True,
        assertion={"include_reason": True},
        score={"include_reason": True},
    )
