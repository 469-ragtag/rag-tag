"""Parity check: verify GRAPH_PAYLOAD_MODE=full vs minimal behaviour.

Builds synthetic graphs/DBs and checks:

  1. Node/property parity between full and minimal payload modes.
  2. DB-backed parity for dotted filters (string + numeric).
  3. DB-backed parity for flat-key fallback into PropertySets (numeric).
  4. Context DB switch on the same graph object does not return stale cache data.
  5. query_service DB context rewiring clears property caches safely.
  6. get_element_properties preserves typed/nested payload fidelity with DB merge.

Run with:
    uv run python scripts/check_payload_mode_parity.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

# ---------------------------------------------------------------------------
# Minimal JSONL fixture
# ---------------------------------------------------------------------------

_WALL_RECORD: dict = {
    "GlobalId": "WALL001",
    "ExpressId": 101,
    "IfcType": "IfcWall",
    "ClassRaw": "IfcWall",
    "Name": "Wall 1",
    "Description": "Exterior load-bearing wall",
    "ObjectType": None,
    "Tag": None,
    "TypeName": None,
    "PredefinedType": "STANDARD",
    "Materials": ["Concrete"],
    "Hierarchy": {"ParentId": "STOR001", "Level": "Level 1"},
    "Geometry": {
        "Centroid": [0.0, 0.0, 1.5],
        "BoundingBox": {
            "min": [-5.0, 0.0, 0.0],
            "max": [5.0, 0.2, 3.0],
        },
    },
    "PropertySets": {
        "Official": {
            "Pset_WallCommon": {
                "FireRating": "EI 90",
                "ThermalTransmittance": 0.28,
                "IsExternal": True,
                "NestedSpec": {"Layer": {"Code": "A1"}},
            }
        },
        "Custom": {},
    },
    "Quantities": {
        "Qto_WallBaseQuantities": {
            "Length": 10.0,
            "Height": 3.0,
        }
    },
    "Relationships": {},
}

_STOREY_RECORD: dict = {
    "GlobalId": "STOR001",
    "ExpressId": 50,
    "IfcType": "IfcBuildingStorey",
    "ClassRaw": "IfcBuildingStorey",
    "Name": "Level 1",
    "Hierarchy": {"ParentId": None, "Level": None},
    "Geometry": {
        "Centroid": [0.0, 0.0, 0.0],
        "BoundingBox": {
            "min": [-10.0, -10.0, 0.0],
            "max": [10.0, 10.0, 3.5],
        },
    },
    "PropertySets": {"Official": {}, "Custom": {}},
    "Quantities": {},
    "Relationships": {},
}

_WALL_RECORD_ALT: dict = deepcopy(_WALL_RECORD)
_WALL_RECORD_ALT["PropertySets"] = {
    "Official": {
        "Pset_WallCommon": {
            "FireRating": "EI 120",
            "ThermalTransmittance": 0.41,
            "IsExternal": True,
            "NestedSpec": {"Layer": {"Code": "B7"}},
        }
    },
    "Custom": {},
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"

_failures: list[str] = []


def check(condition: bool, label: str) -> None:
    if condition:
        print(f"  {PASS}  {label}")
    else:
        print(f"  {FAIL}  {label}")
        _failures.append(label)


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")


# ---------------------------------------------------------------------------
# Main check body
# ---------------------------------------------------------------------------


def main() -> int:
    # Import here so we can guarantee the src path is available.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from rag_tag.agent.graph_tools import register_graph_tools
    from rag_tag.ifc_graph_tool import query_ifc_graph
    from rag_tag.parser.jsonl_to_graph import build_graph_from_jsonl
    from rag_tag.parser.jsonl_to_sql import jsonl_to_sql
    from rag_tag.query_service import _ensure_graph_context

    with tempfile.NamedTemporaryFile(
        suffix=".jsonl", mode="w", delete=False, encoding="utf-8"
    ) as tmp:
        tmp_path = Path(tmp.name)
    tmp_db = tmp_path.with_suffix(".db")
    _write_jsonl(tmp_path, [_STOREY_RECORD, _WALL_RECORD])
    jsonl_to_sql(tmp_path, tmp_db)

    with tempfile.NamedTemporaryFile(
        suffix=".jsonl", mode="w", delete=False, encoding="utf-8"
    ) as tmp_alt:
        tmp_path_alt = Path(tmp_alt.name)
    tmp_db_alt = tmp_path_alt.with_suffix(".db")
    _write_jsonl(tmp_path_alt, [_STOREY_RECORD, _WALL_RECORD_ALT])
    jsonl_to_sql(tmp_path_alt, tmp_db_alt)

    class _ToolCollector:
        def __init__(self) -> None:
            self.tools: dict[str, Any] = {}

        def tool(self, fn):
            self.tools[fn.__name__] = fn
            return fn

    # -----------------------------------------------------------------------
    print("\n[Part 1] Build graphs in both modes")
    # -----------------------------------------------------------------------
    G_full = build_graph_from_jsonl([tmp_path], payload_mode="full")
    G_min = build_graph_from_jsonl([tmp_path], payload_mode="minimal")

    check(
        G_full.graph["_payload_mode"] == "full",
        "full G.graph['_payload_mode'] == 'full'",
    )
    check(
        G_min.graph["_payload_mode"] == "minimal",
        "minimal G.graph['_payload_mode'] == 'minimal'",
    )

    # -----------------------------------------------------------------------
    print("\n[Part 2] Node ID parity")
    # -----------------------------------------------------------------------
    full_ids = set(G_full.nodes)
    min_ids = set(G_min.nodes)
    check(full_ids == min_ids, f"Same node IDs in both modes ({len(full_ids)} nodes)")

    # -----------------------------------------------------------------------
    print("\n[Part 3] Flat properties parity")
    # -----------------------------------------------------------------------
    wall_nid = "Element::WALL001"
    check(wall_nid in G_full, f"{wall_nid} present in full graph")
    check(wall_nid in G_min, f"{wall_nid} present in minimal graph")

    full_props = G_full.nodes[wall_nid].get("properties") or {}
    min_props = G_min.nodes[wall_nid].get("properties") or {}
    check(full_props == min_props, "Flat properties are identical between modes")
    check(
        full_props.get("GlobalId") == "WALL001", "GlobalId present in flat properties"
    )
    check(full_props.get("Name") == "Wall 1", "Name present in flat properties")

    # -----------------------------------------------------------------------
    print("\n[Part 4] Payload content diverges as expected")
    # -----------------------------------------------------------------------
    full_payload = G_full.nodes[wall_nid].get("payload") or {}
    min_payload = G_min.nodes[wall_nid].get("payload") or {}

    check(
        "PropertySets" in full_payload,
        "full mode: PropertySets present in wall payload",
    )
    check(
        "PropertySets" not in min_payload,
        "minimal mode: PropertySets absent from wall payload",
    )
    check(
        "Quantities" in full_payload,
        "full mode: Quantities present in wall payload",
    )
    check(
        "Quantities" not in min_payload,
        "minimal mode: Quantities absent from wall payload",
    )

    # Minimal payload stub must contain the DB-lookup reference fields.
    for field in ("GlobalId", "ExpressId", "IfcType", "ClassRaw", "Name"):
        check(
            field in min_payload,
            f"minimal mode: '{field}' present in minimal payload stub",
        )

    # -----------------------------------------------------------------------
    print("\n[Part 5] find_nodes flat filter works in both modes")
    # -----------------------------------------------------------------------
    for mode_label, G in (("full", G_full), ("minimal", G_min)):
        res = query_ifc_graph(G, "find_nodes", {"property_filters": {"Name": "Wall 1"}})
        elements = (res.get("data") or {}).get("elements", [])
        check(
            len(elements) == 1,
            f"{mode_label} mode: find_nodes flat filter Name='Wall 1' returns 1 match",
        )

    # -----------------------------------------------------------------------
    print("\n[Part 6] DB-backed filter parity (dotted + flat numeric)")
    # -----------------------------------------------------------------------
    pset_filters = {"Pset_WallCommon.FireRating": "EI 90"}
    dotted_numeric_filters = {"Qto_WallBaseQuantities.Length": 10.0}
    flat_pset_numeric_filters = {"ThermalTransmittance": 0.28}

    res_full = query_ifc_graph(G_full, "find_nodes", {"property_filters": pset_filters})
    full_matches = (res_full.get("data") or {}).get("elements", [])
    check(len(full_matches) == 1, "full mode: dotted string pset filter returns 1")

    res_full_qto = query_ifc_graph(
        G_full,
        "find_nodes",
        {"property_filters": dotted_numeric_filters},
    )
    full_qto_matches = (res_full_qto.get("data") or {}).get("elements", [])
    check(
        len(full_qto_matches) == 1,
        "full mode: dotted numeric quantity filter returns 1",
    )

    res_full_flat_pset = query_ifc_graph(
        G_full,
        "find_nodes",
        {"property_filters": flat_pset_numeric_filters},
    )
    full_flat_pset_matches = (res_full_flat_pset.get("data") or {}).get("elements", [])
    check(
        len(full_flat_pset_matches) == 1,
        "full mode: flat pset fallback ThermalTransmittance=0.28 returns 1",
    )

    res_min = query_ifc_graph(G_min, "find_nodes", {"property_filters": pset_filters})
    min_matches = (res_min.get("data") or {}).get("elements", [])
    check(
        len(min_matches) == 0,
        "minimal mode (no DB): dotted string pset filter returns 0",
    )

    res_min_qto = query_ifc_graph(
        G_min,
        "find_nodes",
        {"property_filters": dotted_numeric_filters},
    )
    min_qto_matches = (res_min_qto.get("data") or {}).get("elements", [])
    check(
        len(min_qto_matches) == 0,
        "minimal mode (no DB): dotted numeric quantity filter returns 0",
    )

    res_min_flat_pset = query_ifc_graph(
        G_min,
        "find_nodes",
        {"property_filters": flat_pset_numeric_filters},
    )
    min_flat_pset_matches = (res_min_flat_pset.get("data") or {}).get("elements", [])
    check(
        len(min_flat_pset_matches) == 0,
        "minimal mode (no DB): flat pset fallback ThermalTransmittance=0.28 returns 0",
    )

    # Wire DB context and verify parity is restored in minimal mode.
    G_min.graph["_db_path"] = tmp_db
    res_min_db = query_ifc_graph(
        G_min, "find_nodes", {"property_filters": pset_filters}
    )
    min_db_matches = (res_min_db.get("data") or {}).get("elements", [])
    check(
        len(min_db_matches) == 1,
        "minimal mode (with DB): dotted string pset filter returns 1",
    )

    res_min_db_qto = query_ifc_graph(
        G_min,
        "find_nodes",
        {"property_filters": dotted_numeric_filters},
    )
    min_db_qto_matches = (res_min_db_qto.get("data") or {}).get("elements", [])
    check(
        len(min_db_qto_matches) == 1,
        "minimal mode (with DB): dotted numeric quantity filter returns 1",
    )

    res_min_db_flat_pset = query_ifc_graph(
        G_min,
        "find_nodes",
        {"property_filters": flat_pset_numeric_filters},
    )
    min_db_flat_pset_matches = (res_min_db_flat_pset.get("data") or {}).get(
        "elements", []
    )
    check(
        len(min_db_flat_pset_matches) == 1,
        "minimal mode (with DB): flat pset fallback "
        "ThermalTransmittance=0.28 returns 1",
    )

    # list_property_keys should also recover dotted keys in minimal mode with DB.
    collector = _ToolCollector()
    register_graph_tools(collector)
    list_property_keys = collector.tools["list_property_keys"]
    key_result = list_property_keys(
        SimpleNamespace(deps=G_min),
        class_=None,
        sample_values=False,
    )
    keys = set(((key_result.get("data") or {}).get("keys") or []))
    check(
        "Pset_WallCommon.FireRating" in keys,
        "minimal mode (with DB): list_property_keys includes dotted pset key",
    )
    check(
        "Qto_WallBaseQuantities.Length" in keys,
        "minimal mode (with DB): list_property_keys includes dotted quantity key",
    )

    # -----------------------------------------------------------------------
    print("\n[Part 7] Context DB switch on same graph does not return stale values")
    # -----------------------------------------------------------------------
    warm_028 = query_ifc_graph(
        G_min,
        "find_nodes",
        {"property_filters": {"ThermalTransmittance": 0.28}},
    )
    warm_028_matches = (warm_028.get("data") or {}).get("elements", [])
    check(
        len(warm_028_matches) == 1,
        "warm cache on DB-A: ThermalTransmittance=0.28 returns 1",
    )

    G_min.graph["_db_path"] = tmp_db_alt
    switched_old = query_ifc_graph(
        G_min,
        "find_nodes",
        {"property_filters": {"ThermalTransmittance": 0.28}},
    )
    switched_old_matches = (switched_old.get("data") or {}).get("elements", [])
    check(
        len(switched_old_matches) == 0,
        "after DB switch to DB-B: old value 0.28 no longer matches",
    )

    switched_new = query_ifc_graph(
        G_min,
        "find_nodes",
        {"property_filters": {"ThermalTransmittance": 0.41}},
    )
    switched_new_matches = (switched_new.get("data") or {}).get("elements", [])
    check(
        len(switched_new_matches) == 1,
        "after DB switch to DB-B: new value 0.41 matches",
    )

    # -----------------------------------------------------------------------
    print("\n[Part 8] query_service context rewiring clears graph property caches")
    # -----------------------------------------------------------------------
    G_ctx = build_graph_from_jsonl([tmp_path], payload_mode="minimal")
    G_ctx.graph["_db_path"] = tmp_db
    G_ctx.graph["_property_cache"] = {("old-db", wall_nid): {"payload": {}}}
    G_ctx.graph["_property_key_cache"] = {("old-db", ""): {"k": [1]}}
    _ensure_graph_context(
        G_ctx,
        cast(Any, object()),
        False,
        db_path=tmp_db_alt,
    )
    ctx_db_path = G_ctx.graph.get("_db_path")
    check(
        ctx_db_path is not None and Path(ctx_db_path).resolve() == tmp_db_alt.resolve(),
        "_ensure_graph_context updates graph _db_path to new DB",
    )
    check(
        "_property_cache" not in G_ctx.graph,
        "_ensure_graph_context clears _property_cache on DB switch",
    )
    check(
        "_property_key_cache" not in G_ctx.graph,
        "_ensure_graph_context clears _property_key_cache on DB switch",
    )

    # -----------------------------------------------------------------------
    print("\n[Part 9] get_element_properties preserves typed/nested fidelity")
    # -----------------------------------------------------------------------
    G_full.graph.pop("_db_path", None)
    props_no_db = query_ifc_graph(
        G_full,
        "get_element_properties",
        {"element_id": wall_nid},
    )
    check(
        props_no_db.get("status") == "ok",
        "get_element_properties without DB returns ok",
    )
    payload_no_db = (props_no_db.get("data") or {}).get("payload") or {}

    G_full.graph["_db_path"] = tmp_db
    props_with_db = query_ifc_graph(
        G_full,
        "get_element_properties",
        {"element_id": wall_nid},
    )
    check(
        props_with_db.get("status") == "ok", "get_element_properties with DB returns ok"
    )
    payload_with_db = (props_with_db.get("data") or {}).get("payload") or {}

    pset_no_db = (
        payload_no_db.get("PropertySets", {})
        .get("Official", {})
        .get("Pset_WallCommon", {})
    )
    pset_with_db = (
        payload_with_db.get("PropertySets", {})
        .get("Official", {})
        .get("Pset_WallCommon", {})
    )
    check(
        pset_with_db.get("ThermalTransmittance") == 0.28
        and isinstance(pset_with_db.get("ThermalTransmittance"), float),
        "DB merge keeps ThermalTransmittance as float (not string)",
    )
    check(
        pset_with_db.get("IsExternal") is True,
        "DB merge keeps boolean pset value type",
    )
    check(
        isinstance(pset_with_db.get("NestedSpec"), dict),
        "DB merge keeps nested dict pset value (not string blob)",
    )
    check(
        pset_with_db == pset_no_db,
        "DB merge does not mutate richer in-memory pset structure",
    )

    # -----------------------------------------------------------------------
    print("\n[Part 10] Invalid payload_mode falls back to 'full'")
    # -----------------------------------------------------------------------
    G_bad = build_graph_from_jsonl([tmp_path], payload_mode="bogus")
    check(
        G_bad.graph["_payload_mode"] == "full",
        "Invalid payload_mode 'bogus' silently resolved to 'full'",
    )
    bad_wall_payload = G_bad.nodes[wall_nid].get("payload") or {}
    check(
        "PropertySets" in bad_wall_payload,
        "Fallback 'full' mode: PropertySets present in payload",
    )

    # -----------------------------------------------------------------------
    tmp_path.unlink(missing_ok=True)
    tmp_db.unlink(missing_ok=True)
    tmp_path_alt.unlink(missing_ok=True)
    tmp_db_alt.unlink(missing_ok=True)

    if _failures:
        print(f"\n\033[31m{len(_failures)} check(s) FAILED:\033[0m")
        for f in _failures:
            print(f"  - {f}")
        return 1

    print("\n\033[32mAll checks passed.\033[0m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
