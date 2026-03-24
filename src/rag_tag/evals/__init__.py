"""Benchmark dataset loading utilities for Pydantic Evals integration."""

from .benchmark import (
    BenchmarkCliConfig,
    BenchmarkCombination,
    BenchmarkSuiteEntry,
    BenchmarkSuiteResult,
    build_benchmark_cli_config,
    expand_benchmark_matrix,
    load_benchmark_dataset_with_tags,
    run_benchmark_suite,
)
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
from .metrics import BenchmarkUsageAggregate, aggregate_benchmark_usage
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
    "BenchmarkCliConfig",
    "BenchmarkCombination",
    "BenchmarkDataset",
    "BenchmarkExperimentConfig",
    "BenchmarkPromptStrategy",
    "BenchmarkRoute",
    "BenchmarkStrategySettings",
    "BenchmarkSuiteEntry",
    "BenchmarkSuiteResult",
    "BenchmarkTaskBundle",
    "BenchmarkTaskResult",
    "BenchmarkUsage",
    "BenchmarkUsageAggregate",
    "DEFAULT_ANSWER_JUDGE_RUBRIC",
    "NoExecutionError",
    "RouteMatchesExpected",
    "aggregate_benchmark_usage",
    "build_benchmark_cli_config",
    "build_default_answer_judge",
    "build_eval_dataset",
    "evaluate_benchmark_dataset",
    "expand_benchmark_matrix",
    "load_benchmark_dataset_with_tags",
    "load_benchmark_dataset",
    "resolve_benchmark_strategy",
    "run_benchmark_suite",
    "run_benchmark_case",
    "temporary_runtime_overrides",
]
