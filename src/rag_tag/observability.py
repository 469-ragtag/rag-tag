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
from pathlib import Path


def _load_dotenv() -> None:
    """Load .env from project root if it exists.

    Searches upward from this file to find .env in the project root.
    Similar to pydantic_ai._load_env() behavior.
    """
    try:
        from dotenv import load_dotenv
    except ModuleNotFoundError:
        # python-dotenv not installed, skip loading
        return

    # Search upward from this file to find project root with .env
    current = Path(__file__).resolve().parent
    for _ in range(10):  # Max 10 levels up
        env_file = current / ".env"
        if env_file.is_file():
            load_dotenv(env_file)
            return
        parent = current.parent
        if parent == current:  # Reached filesystem root
            break
        current = parent


def setup_logfire(enabled: bool = False) -> None:
    """Configure Logfire instrumentation for PydanticAI.

    Args:
        enabled: If True, configure and enable Logfire instrumentation.
                 If False, Logfire is not configured (no-op).

    Note:
        Logfire requires a WRITE token for sending traces to the cloud.
        Set LOGFIRE_TOKEN (write token) in .env or run `logfire auth` CLI.
        Read tokens are for the query API only and will cause 401 errors.

        If Logfire is not installed, this function is a no-op.

        This function will automatically load .env from the project root
        before checking for LOGFIRE_TOKEN.

        Optional: Set LOGFIRE_PROJECT_NAME to specify a project.
    """
    if not enabled:
        return

    # Load .env before checking for token
    _load_dotenv()

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
            "LOGFIRE_TOKEN not set. Set a WRITE token (not read token) in .env or "
            "run 'logfire auth' to authenticate. Read tokens are for query API only. "
            "Tracing will work locally without cloud sync.",
            stacklevel=2,
        )
        # Logfire will still work locally, but won't send data to cloud
        # Good for development/testing

    # Configure Logfire with minimal settings
    # This will auto-detect project from environment or create a new one
    try:
        logfire.configure(
            # Use environment variables for token and project_name
            # LOGFIRE_TOKEN (must be WRITE token), LOGFIRE_PROJECT_NAME
            # 'if-token-present' sends only if write token exists or local
            # credentials present
            send_to_logfire="if-token-present",
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
