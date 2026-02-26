"""Batch 4 smoke-tests: multi-DB list merge global limit + metadata coherence.

Run with:
    uv run python scripts/check_batch4.py
"""

from __future__ import annotations

import re
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Minimal stubs so we can import query_service without installing every dep.
# ---------------------------------------------------------------------------


# Stub rag_tag.router
@dataclass(frozen=True)
class SqlRequest:
    intent: str
    ifc_class: str | None
    level_like: str | None
    limit: int


@dataclass(frozen=True)
class RouteDecision:
    route: str
    reason: str
    sql_request: SqlRequest | None


def _make_db_envelope(items: list[dict], total: int, limit: int) -> dict[str, Any]:
    """Simulate what ifc_sql_tool.query_ifc_sql returns for a list intent."""
    return {
        "status": "ok",
        "data": {
            "intent": "list",
            "filters": {},
            "total_count": total,
            "limit": limit,  # per-DB limit — should NOT leak into merged result
            "items": items[:limit],  # SQL LIMIT applied per DB
            "summary": f"Found {total}, showing {min(total, limit)}.",
            "sql": {"query": "SELECT ...", "params": []},
        },
        "error": None,
    }


def _make_count_envelope(count: int) -> dict[str, Any]:
    return {
        "status": "ok",
        "data": {
            "intent": "count",
            "filters": {},
            "count": count,
            "summary": f"Found {count}.",
            "sql": {"query": "SELECT COUNT(*) ...", "params": []},
        },
        "error": None,
    }


# ---------------------------------------------------------------------------
# Import target under test using minimal mocks for heavy deps.
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# Provide lightweight stubs for packages that require real IFC / GPU resources.
_stubs = {
    "networkx": types.ModuleType("networkx"),
    "rag_tag.agent": types.ModuleType("rag_tag.agent"),
    "rag_tag.ifc_sql_tool": types.ModuleType("rag_tag.ifc_sql_tool"),
    "rag_tag.paths": types.ModuleType("rag_tag.paths"),
    "rag_tag.router": types.ModuleType("rag_tag.router"),
}
_stubs["networkx"].DiGraph = object  # type: ignore[attr-defined]
_stubs["rag_tag.agent"].GraphAgent = object  # type: ignore[attr-defined]


class _FakeSqlQueryError(RuntimeError):
    pass


_stubs["rag_tag.ifc_sql_tool"].SqlQueryError = _FakeSqlQueryError  # type: ignore[attr-defined]
_stubs[
    "rag_tag.ifc_sql_tool"
].query_ifc_sql = None  # replaced per test  # type: ignore[attr-defined]
_stubs["rag_tag.paths"].find_project_root = lambda *a: None  # type: ignore[attr-defined]
_stubs["rag_tag.router"].RouteDecision = RouteDecision  # type: ignore[attr-defined]
_stubs["rag_tag.router"].SqlRequest = SqlRequest  # type: ignore[attr-defined]
_stubs["rag_tag.router"].route_question = None  # type: ignore[attr-defined]

for name, stub in _stubs.items():
    sys.modules[name] = stub  # type: ignore[assignment]

from rag_tag import query_service  # noqa: E402  (must be after stubs)

# Re-point the module-level names query_service uses.
query_service.SqlQueryError = _FakeSqlQueryError  # type: ignore[attr-defined]
query_service.query_ifc_sql = None  # replaced per test


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FAIL = "\033[31mFAIL\033[0m"
PASS = "\033[32mPASS\033[0m"
_failures: list[str] = []


def check(condition: bool, name: str) -> None:
    if condition:
        print(f"  {PASS}  {name}")
    else:
        print(f"  {FAIL}  {name}")
        _failures.append(name)


# ---------------------------------------------------------------------------
# Test 1 — list merge with 2 DBs must not exceed requested limit
# ---------------------------------------------------------------------------
print("\n[Test 1] Multi-DB list merge: global limit enforced")

LIMIT = 10
items_per_db = [{"name": f"item-db1-{i}", "ifc_class": "IfcWall"} for i in range(LIMIT)]
items_db2 = [{"name": f"item-db2-{i}", "ifc_class": "IfcWall"} for i in range(LIMIT)]

envelopes = [
    _make_db_envelope(items_per_db, total=LIMIT, limit=LIMIT),
    _make_db_envelope(items_db2, total=LIMIT, limit=LIMIT),
]
call_count = 0


def _mock_query(db_path: Path, request: SqlRequest) -> dict[str, Any]:  # noqa: ARG001
    global call_count
    env = envelopes[call_count]
    call_count += 1
    return env


query_service.query_ifc_sql = _mock_query  # type: ignore[attr-defined]

db_paths = [Path("/fake/db1.db"), Path("/fake/db2.db")]
req = SqlRequest(intent="list", ifc_class="IfcWall", level_like=None, limit=LIMIT)
decision = RouteDecision(route="sql", reason="test", sql_request=req)

result = query_service.execute_sql_query(decision, db_paths)
data = result.get("data", {})
items = data.get("items", [])

check(len(items) <= LIMIT, f"items count {len(items)} <= limit {LIMIT}")
check(
    data.get("limit") == LIMIT,
    f"data.limit == requested limit ({data.get('limit')} == {LIMIT})",
)
check(
    len(items) == data.get("limit") or len(items) < data.get("limit", 0),  # type: ignore[operator]
    f"len(items)={len(items)} coherent with data.limit={data.get('limit')}",
)
# Summary should mention the actual shown count, not a stale per-DB number.
shown_in_summary = None
answer: str = result.get("answer", "")
for token in answer.split():
    try:
        shown_in_summary = int(token.rstrip("."))
        break
    except ValueError:
        continue
# The last number in "Found X ..., showing Y." is `shown`.

m = re.search(r"showing (\d+)", answer)
if m:
    shown_in_summary = int(m.group(1))
check(
    shown_in_summary == len(items),
    f"summary 'showing {shown_in_summary}' matches len(items)={len(items)}",
)


# ---------------------------------------------------------------------------
# Test 2 — list merge does NOT over-cap when total < limit
# ---------------------------------------------------------------------------
print("\n[Test 2] Multi-DB list merge: no over-cap when total < limit")

LIMIT2 = 50
items_s1 = [{"name": f"s1-{i}", "ifc_class": "IfcDoor"} for i in range(5)]
items_s2 = [{"name": f"s2-{i}", "ifc_class": "IfcDoor"} for i in range(3)]

envelopes2 = [
    _make_db_envelope(items_s1, total=5, limit=LIMIT2),
    _make_db_envelope(items_s2, total=3, limit=LIMIT2),
]
call_count2 = 0


def _mock_query2(db_path: Path, request: SqlRequest) -> dict[str, Any]:  # noqa: ARG001
    global call_count2
    env = envelopes2[call_count2]
    call_count2 += 1
    return env


query_service.query_ifc_sql = _mock_query2  # type: ignore[attr-defined]

req2 = SqlRequest(intent="list", ifc_class="IfcDoor", level_like=None, limit=LIMIT2)
decision2 = RouteDecision(route="sql", reason="test2", sql_request=req2)
result2 = query_service.execute_sql_query(decision2, db_paths)
data2 = result2.get("data", {})
items2 = data2.get("items", [])

check(len(items2) == 8, f"all 8 items returned when total < limit (got {len(items2)})")
check(
    data2.get("limit") == LIMIT2, f"data.limit == {LIMIT2} (got {data2.get('limit')})"
)
check(
    data2.get("total_count") == 8, f"total_count == 8 (got {data2.get('total_count')})"
)

m2 = re.search(r"showing (\d+)", result2.get("answer", ""))
shown2 = int(m2.group(1)) if m2 else None
check(shown2 == 8, f"summary 'showing {shown2}' == 8")


# ---------------------------------------------------------------------------
# Test 3 — count intent unaffected
# ---------------------------------------------------------------------------
print("\n[Test 3] Count intent: combined count unaffected by limit logic")

envelopes3 = [_make_count_envelope(42), _make_count_envelope(18)]
call_count3 = 0


def _mock_query3(db_path: Path, request: SqlRequest) -> dict[str, Any]:  # noqa: ARG001
    global call_count3
    env = envelopes3[call_count3]
    call_count3 += 1
    return env


query_service.query_ifc_sql = _mock_query3  # type: ignore[attr-defined]

req3 = SqlRequest(intent="count", ifc_class="IfcWall", level_like=None, limit=50)
decision3 = RouteDecision(route="sql", reason="test3", sql_request=req3)
result3 = query_service.execute_sql_query(decision3, db_paths)
data3 = result3.get("data", {})

check(data3.get("count") == 60, f"combined count == 60 (got {data3.get('count')})")
check(
    data3.get("items") == [], f"items empty for count intent (got {data3.get('items')})"
)
check(
    "60" in result3.get("answer", ""),
    f"answer mentions '60' (answer='{result3.get('answer')}')",
)


# ---------------------------------------------------------------------------
# Test 4 — stale guidance text no longer present
# ---------------------------------------------------------------------------
print("\n[Test 4] Stale CSV guidance text absent from user-facing messages")

src_qs = Path(__file__).resolve().parents[1] / "src/rag_tag/query_service.py"
src_tui = Path(__file__).resolve().parents[1] / "src/rag_tag/tui.py"

qs_text = src_qs.read_text()
tui_text = src_tui.read_text()

check(
    "csv_to_sql.py" not in qs_text,
    "query_service.py contains no reference to 'csv_to_sql.py'",
)
check(
    "csv_to_sql.py" not in tui_text,
    "tui.py contains no reference to 'csv_to_sql.py'",
)
check(
    "rag-tag-jsonl-to-sql" in qs_text,
    "query_service.py references new 'rag-tag-jsonl-to-sql' command",
)
check(
    "rag-tag-jsonl-to-sql" in tui_text,
    "tui.py references new 'rag-tag-jsonl-to-sql' command",
)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print()
if _failures:
    print(f"\033[31m{len(_failures)} check(s) FAILED:\033[0m")
    for f in _failures:
        print(f"  - {f}")
    sys.exit(1)
else:
    print("\033[32mAll checks passed.\033[0m")
