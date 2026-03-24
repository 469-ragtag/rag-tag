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


def build_default_answer_judge(model: str | None = None) -> LLMJudge:
    """Build the default answer-quality evaluator for benchmark experiments."""

    return LLMJudge(
        rubric=DEFAULT_ANSWER_JUDGE_RUBRIC,
        model=model,
        include_input=True,
        assertion=True,
        score=True,
    )
