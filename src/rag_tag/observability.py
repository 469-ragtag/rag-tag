"""Observability setup with Logfire integration for PydanticAI.

Provides instrumentation for PydanticAI agents and tools, tracking:
- Request/response times
- Token usage and costs
- Tool calls and results
- Errors and retries
"""

from __future__ import annotations

import os
import warnings


def setup_logfire(enabled: bool = False) -> None:
    """Configure Logfire instrumentation for PydanticAI.

    Args:
        enabled: If True, configure and enable Logfire instrumentation.
                 If False, Logfire is not configured (no-op).

    Note:
        Logfire requires a token for authentication. Set LOGFIRE_TOKEN
        environment variable or run `logfire auth` CLI command.

        If Logfire is not installed, this function is a no-op.
    """
    if not enabled:
        return

    try:
        import logfire
    except ModuleNotFoundError:
        warnings.warn(
            "Logfire not installed. Install with: pip install logfire",
            stacklevel=2,
        )
        return

    # Check if token is configured
    token = os.getenv("LOGFIRE_TOKEN")
    if not token:
        warnings.warn(
            "LOGFIRE_TOKEN not set. Run 'logfire auth' or set token in environment.",
            stacklevel=2,
        )
        # Logfire will still work locally, but won't send data to cloud
        # Good for development/testing

    # Configure Logfire with minimal settings
    # This will auto-detect project from environment or create a new one
    try:
        logfire.configure(
            # Use environment variables for token and project_name
            # LOGFIRE_TOKEN, LOGFIRE_PROJECT_NAME
            send_to_logfire=token is not None,
        )
    except Exception as exc:
        warnings.warn(f"Failed to configure Logfire: {exc}", stacklevel=2)
        return

    # Instrument PydanticAI for automatic tracing
    try:
        logfire.instrument_pydantic_ai()
    except Exception as exc:
        warnings.warn(f"Failed to instrument PydanticAI: {exc}", stacklevel=2)
        return

    # Success message (only if token is set)
    if token:
        print("Logfire instrumentation enabled for PydanticAI")
    else:
        print("Logfire instrumentation enabled (local only, no cloud sync)")
