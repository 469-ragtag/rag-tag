from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


class GraphBackend(Protocol):
    """Minimal runtime backend contract for graph query engines."""

    name: str

    def load(
        self,
        *,
        dataset: str | None = None,
        payload_mode: str | None = None,
    ) -> GraphRuntime: ...

    def query(
        self,
        runtime: GraphRuntime,
        action: str,
        params: dict[str, Any],
        payload_mode: str,
    ) -> dict[str, Any]: ...

    def close(self, runtime: GraphRuntime) -> None: ...


@dataclass(slots=True)
class GraphRuntime:
    """Backend-agnostic graph runtime state."""

    backend_name: str
    backend: GraphBackend
    selected_datasets: list[str] = field(default_factory=list)
    payload_mode: str = "full"
    context_db_path: Path | None = None
    backend_handle: Any = None
    caches: dict[str, Any] = field(default_factory=dict)
