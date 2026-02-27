"""Logfire setup helpers for optional PydanticAI observability.

Configuration is best-effort and safely degrades when dependencies are missing.
"""

from __future__ import annotations

import logging
import os
import warnings
from dataclasses import dataclass
from pathlib import Path

_logger = logging.getLogger(__name__)


@dataclass
class LogfireStatus:
    """Status returned by ``setup_logfire``."""

    enabled: bool = False
    cloud_sync: bool = False
    url: str = ""


def _load_dotenv() -> None:
    """Load the nearest ``.env`` file by walking upward from this module."""
    try:
        from dotenv import load_dotenv
    except ModuleNotFoundError:
        return

    current = Path(__file__).resolve().parent
    for _ in range(10):
        env_file = current / ".env"
        if env_file.is_file():
            load_dotenv(env_file)
            return
        parent = current.parent
        if parent == current:
            break
        current = parent


def setup_logfire(
    enabled: bool = False,
    console: bool = True,
) -> LogfireStatus:
    """Configure Logfire and instrument PydanticAI when enabled."""
    if not enabled:
        return LogfireStatus()

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

    token = os.getenv("LOGFIRE_TOKEN")
    if not token and console:
        warnings.warn(
            "LOGFIRE_TOKEN not set. Set a WRITE token (not read token) in .env or "
            "run 'logfire auth' to authenticate. Read tokens are for query API only. "
            "Tracing will work locally without cloud sync.",
            stacklevel=2,
        )

    # NOTE: In TUI mode, disable logfire console output to avoid display noise.
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

    try:
        logfire.instrument_pydantic_ai()
    except Exception as exc:
        if console:
            warnings.warn(f"Failed to instrument PydanticAI: {exc}", stacklevel=2)
        return LogfireStatus()

    if token:
        url = "https://logfire.pydantic.dev"
        if console:
            _logger.debug("Logfire instrumentation enabled for PydanticAI")
        return LogfireStatus(enabled=True, cloud_sync=True, url=url)
    else:
        if console:
            _logger.debug("Logfire instrumentation enabled (local only, no cloud sync)")
        return LogfireStatus(enabled=True, cloud_sync=False, url="")
