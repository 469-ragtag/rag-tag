"""Benchmark-only prompt and orchestration strategy definitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

BenchmarkPromptStrategy = Literal["baseline", "strict-grounded", "decompose"]

_STRICT_GROUNDED_APPENDIX = """
Benchmark strategy: strict-grounded

Additional rules:
- Keep the final answer concise and evidence-bound.
- If evidence is partial or ambiguous, say so plainly instead of inferring.
- Prefer explicit uncertainty over speculative completion.
- Do not imply measurements, materials, relationships, or counts unless a tool
  returned them.
- When comparing or summarizing, only use grounded values already established
  in the tool evidence.
""".strip()


@dataclass(frozen=True)
class BenchmarkStrategySettings:
    """Resolved benchmark-only runtime settings for a prompt strategy."""

    name: BenchmarkPromptStrategy
    graph_orchestrator_override: str | None = None
    graph_prompt_append: str | None = None


def resolve_benchmark_strategy(
    strategy: str,
) -> BenchmarkStrategySettings:
    """Normalize a benchmark strategy name into concrete runtime settings."""

    normalized = strategy.strip().lower()
    if normalized == "baseline":
        return BenchmarkStrategySettings(name="baseline")
    if normalized == "strict-grounded":
        return BenchmarkStrategySettings(
            name="strict-grounded",
            graph_prompt_append=_STRICT_GROUNDED_APPENDIX,
        )
    if normalized == "decompose":
        return BenchmarkStrategySettings(
            name="decompose",
            graph_orchestrator_override="langgraph",
        )

    raise ValueError(
        "Unsupported benchmark prompt strategy "
        f"{strategy!r}. Allowed values: baseline, strict-grounded, decompose."
    )
