"""Runtime override helpers for benchmark evaluation flows."""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager

from rag_tag.config import (
    AGENT_PROFILE_ENV_VAR,
    CONFIG_PATH_ENV_VAR,
    ROUTER_PROFILE_ENV_VAR,
)

_MISSING = object()
_AGENT_MODEL_ENV_VAR = "AGENT_MODEL"
_ROUTER_MODEL_ENV_VAR = "ROUTER_MODEL"
BENCHMARK_GRAPH_ORCHESTRATOR_ENV_VAR = "RAG_TAG_BENCHMARK_GRAPH_ORCHESTRATOR"
BENCHMARK_GRAPH_PROMPT_APPEND_ENV_VAR = "RAG_TAG_BENCHMARK_GRAPH_PROMPT_APPEND"


@contextmanager
def temporary_runtime_overrides(
    *,
    config_path: str | None = None,
    router_profile: str | None = None,
    agent_profile: str | None = None,
    graph_orchestrator_override: str | None = None,
    graph_prompt_append: str | None = None,
) -> Iterator[None]:
    """Temporarily apply config/profile overrides for benchmark runs.

    Profile overrides clear the corresponding direct model env vars so that
    profile-based resolution remains authoritative for the duration of the run.
    """

    previous_values = {
        CONFIG_PATH_ENV_VAR: os.environ.get(CONFIG_PATH_ENV_VAR, _MISSING),
        ROUTER_PROFILE_ENV_VAR: os.environ.get(ROUTER_PROFILE_ENV_VAR, _MISSING),
        AGENT_PROFILE_ENV_VAR: os.environ.get(AGENT_PROFILE_ENV_VAR, _MISSING),
        _ROUTER_MODEL_ENV_VAR: os.environ.get(_ROUTER_MODEL_ENV_VAR, _MISSING),
        _AGENT_MODEL_ENV_VAR: os.environ.get(_AGENT_MODEL_ENV_VAR, _MISSING),
        BENCHMARK_GRAPH_ORCHESTRATOR_ENV_VAR: os.environ.get(
            BENCHMARK_GRAPH_ORCHESTRATOR_ENV_VAR,
            _MISSING,
        ),
        BENCHMARK_GRAPH_PROMPT_APPEND_ENV_VAR: os.environ.get(
            BENCHMARK_GRAPH_PROMPT_APPEND_ENV_VAR,
            _MISSING,
        ),
    }

    try:
        if config_path is not None:
            os.environ[CONFIG_PATH_ENV_VAR] = config_path
        if router_profile is not None:
            os.environ[ROUTER_PROFILE_ENV_VAR] = router_profile
            os.environ.pop(_ROUTER_MODEL_ENV_VAR, None)
        if agent_profile is not None:
            os.environ[AGENT_PROFILE_ENV_VAR] = agent_profile
            os.environ.pop(_AGENT_MODEL_ENV_VAR, None)
        if graph_orchestrator_override is not None:
            os.environ[BENCHMARK_GRAPH_ORCHESTRATOR_ENV_VAR] = (
                graph_orchestrator_override
            )
        if graph_prompt_append is not None:
            os.environ[BENCHMARK_GRAPH_PROMPT_APPEND_ENV_VAR] = graph_prompt_append
        yield
    finally:
        for name, previous in previous_values.items():
            if previous is _MISSING:
                os.environ.pop(name, None)
            else:
                os.environ[name] = str(previous)
