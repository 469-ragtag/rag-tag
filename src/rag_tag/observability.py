"""Logfire observability setup for agent tracing."""

from __future__ import annotations

import os
import sys


def setup_logfire(*, enabled: bool = True) -> None:
    """Configure Logfire instrumentation for PydanticAI.

    Args:
        enabled: Whether to enable Logfire. If False, does nothing.
    """
    if not enabled:
        return

    try:
        import logfire
    except ImportError:
        print(
            "Warning: logfire not installed. Run 'uv sync' to enable tracing.",
            file=sys.stderr,
        )
        return

    token = os.getenv("LOGFIRE_TOKEN")
    if token:
        try:
            logfire.configure(token=token)
        except Exception as exc:
            print(
                f"Warning: Failed to configure Logfire with token: {exc}",
                file=sys.stderr,
            )
            return
    else:
        try:
            logfire.configure(send_to_logfire=False)
        except Exception as exc:
            print(
                f"Warning: Failed to configure Logfire in local mode: {exc}",
                file=sys.stderr,
            )
            return

    try:
        logfire.instrument_pydantic_ai()
    except Exception as exc:
        print(
            f"Warning: Failed to instrument PydanticAI: {exc}",
            file=sys.stderr,
        )
