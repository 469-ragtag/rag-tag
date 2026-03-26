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
You are evaluating whether a rag-tag benchmark answer is close enough to the
expected IFC/BIM answer.

Treat answer correctness as the primary task. Route correctness is scored
separately, so a route mismatch alone should not fail an otherwise correct,
grounded answer.

Compare all of the following when they are available:
- the input question
- expected_output.answer.canonical
- expected_output.answer.acceptable
- expected_output.answer.judge_notes
- expected_output.expected_route
- output.selected_route
- output.answer (the final answer text shown to the user)
- output.data (structured payload and supporting evidence)
- output.warning or output.error

Judging rules:
- Pass when the final answer is materially correct or acceptably equivalent to
  the canonical or acceptable answers.
- Use output.data as important evidence for counts, entities, relationships,
  and other structured facts. The final answer text is still the main answer,
  but the structured payload can confirm whether it is grounded.
- Give credit for concise wording when the meaning is still correct and the
  structured payload supports it.
- Fail when the answer is missing, materially incorrect, contradicted by the
  structured payload, unsupported, or fabricated.
- If the system clearly states uncertainty because the evidence is incomplete,
  that can still be acceptable when it matches the benchmark expectations.

Score from 0.0 to 1.0, where 1.0 means clearly correct and grounded, 0.0 means
incorrect or unsupported, and intermediate scores reflect partial correctness.
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
        include_expected_output=True,
        assertion={
            "evaluation_name": "answer_correct",
            "include_reason": True,
        },
        score={
            "evaluation_name": "judge_score",
            "include_reason": True,
        },
    )
