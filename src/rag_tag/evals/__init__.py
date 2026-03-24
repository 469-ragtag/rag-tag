"""Benchmark dataset loading utilities for Pydantic Evals integration."""

from .dataset import (
    BenchmarkCase,
    BenchmarkDataset,
    BenchmarkRoute,
    load_benchmark_dataset,
)
from .evaluators import (
    DEFAULT_ANSWER_JUDGE_RUBRIC,
    NoExecutionError,
    RouteMatchesExpected,
    build_default_answer_judge,
)
from .runner import (
    BenchmarkExperimentConfig,
    build_eval_dataset,
    evaluate_benchmark_dataset,
)
from .runtime import temporary_runtime_overrides
from .strategies import (
    BenchmarkPromptStrategy,
    BenchmarkStrategySettings,
    resolve_benchmark_strategy,
)
from .task_runner import (
    BenchmarkTaskBundle,
    BenchmarkTaskResult,
    BenchmarkUsage,
    run_benchmark_case,
)

__all__ = [
    "BenchmarkCase",
    "BenchmarkDataset",
    "BenchmarkExperimentConfig",
    "BenchmarkPromptStrategy",
    "BenchmarkRoute",
    "BenchmarkStrategySettings",
    "BenchmarkTaskBundle",
    "BenchmarkTaskResult",
    "BenchmarkUsage",
    "DEFAULT_ANSWER_JUDGE_RUBRIC",
    "NoExecutionError",
    "RouteMatchesExpected",
    "build_default_answer_judge",
    "build_eval_dataset",
    "evaluate_benchmark_dataset",
    "load_benchmark_dataset",
    "resolve_benchmark_strategy",
    "run_benchmark_case",
    "temporary_runtime_overrides",
]
