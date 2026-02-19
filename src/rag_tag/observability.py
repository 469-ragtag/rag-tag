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
from dataclasses import dataclass
from pathlib import Path


@dataclass
class LogfireStatus:
    """Result returned by setup_logfire().

    Attributes:
        enabled: True when Logfire was successfully configured and
            PydanticAI instrumented.
        cloud_sync: True when a LOGFIRE_TOKEN write-token was present, meaning
            traces are forwarded to the Logfire cloud dashboard.
        url: The Logfire dashboard base URL when cloud_sync is True, otherwise
            an empty string.
    """

    enabled: bool = False
    cloud_sync: bool = False
    url: str = ""


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


def setup_logfire(
    enabled: bool = False,
    console: bool = True,
) -> LogfireStatus:
    """Configure Logfire instrumentation for PydanticAI.

    Args:
        enabled: If True, configure and enable Logfire instrumentation.
                 If False, Logfire is not configured (no-op).
        console: If False, suppress all Logfire console output (warnings and
                 print messages).  Set to False in TUI mode so that log lines
                 do not corrupt the Textual display.

    Returns:
        LogfireStatus with enabled/cloud_sync/url populated.

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
        return LogfireStatus()

    # Load .env before checking for token
    _load_dotenv()

    try:
        import logfire
    except ModuleNotFoundError:
        if console:
            warnings.warn(
                "Logfire not installed. Install with: pip install logfire",
                stacklevel=2,
            )
        return LogfireStatus()

    # Check if token is configured
    token = os.getenv("LOGFIRE_TOKEN")
    if not token and console:
        warnings.warn(
            "LOGFIRE_TOKEN not set. Set a WRITE token (not read token) in .env or "
            "run 'logfire auth' to authenticate. Read tokens are for query API only. "
            "Tracing will work locally without cloud sync.",
            stacklevel=2,
        )
        # Logfire will still work locally, but won't send data to cloud

    # Build configure kwargs.  When running inside the TUI we pass
    # console=False so logfire never writes to stderr/stdout — that output
    # would corrupt the Textual display.
    configure_kwargs: dict[str, object] = {
        "send_to_logfire": "if-token-present",
    }
    if not console:
        configure_kwargs["console"] = False

    try:
        logfire.configure(**configure_kwargs)  # type: ignore[arg-type]
    except Exception as exc:
        if console:
            warnings.warn(f"Failed to configure Logfire: {exc}", stacklevel=2)
        return LogfireStatus()

    # Instrument PydanticAI for automatic tracing
    try:
        logfire.instrument_pydantic_ai()
    except Exception as exc:
        if console:
            warnings.warn(f"Failed to instrument PydanticAI: {exc}", stacklevel=2)
        return LogfireStatus()

    if token:
        url = "https://logfire.pydantic.dev"
        if console:
            print("Logfire instrumentation enabled for PydanticAI")
        return LogfireStatus(enabled=True, cloud_sync=True, url=url)
    else:
        if console:
            print("Logfire instrumentation enabled (local only, no cloud sync)")
        return LogfireStatus(enabled=True, cloud_sync=False, url="")
