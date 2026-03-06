"""Parity check: verify GRAPH_PAYLOAD_MODE=full vs minimal behaviour.

Builds synthetic graphs in both modes from the same JSONL fixture and asserts:

  1. Node IDs and flat ``properties`` parity.
  2. ``_payload_mode`` metadata reflects the requested mode.
  3. ``full`` keeps PropertySets/Quantities while ``minimal`` omits them.
  4. ``find_nodes`` flat + dotted filtering parity (with and without DB wiring).
  5. Minimal+DB supports flat pset fallback (e.g. ThermalTransmittance).
  6. Minimal+DB supports numeric dotted-key parity.
  7. DB context switching on one graph object does not leak stale cache data.
  8. ``get_element_properties`` returns typed/nested payload values and does not
     degrade richer in-memory payload content when DB rows are legacy/degraded.

Run with:
    uv run python scripts/check_payload_mode_parity.py
"""

from __future__ import annotations

import json
import shutil
import sqlite3
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

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
                "PerformanceData": {
                    "UValue": 0.28,
                    "IsExternal": True,
                },
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

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir_path = Path(tmp_dir)
        tmp_path = tmp_dir_path / "model_a.jsonl"
        tmp_db = tmp_dir_path / "model_a.db"
        _write_jsonl(tmp_path, [_STOREY_RECORD, _WALL_RECORD])
        jsonl_to_sql(tmp_path, tmp_db)

        wall_record_db2 = json.loads(json.dumps(_WALL_RECORD))
        wall_pset_db2 = wall_record_db2["PropertySets"]["Official"]["Pset_WallCommon"]
        wall_pset_db2["FireRating"] = "EI 30"
        wall_pset_db2["ThermalTransmittance"] = 0.31
        wall_pset_db2["PerformanceData"] = {"UValue": 0.31, "IsExternal": False}

        tmp_path_db2 = tmp_dir_path / "model_b.jsonl"
        tmp_db2 = tmp_dir_path / "model_b.db"
        _write_jsonl(tmp_path_db2, [_STOREY_RECORD, wall_record_db2])
        jsonl_to_sql(tmp_path_db2, tmp_db2)

        class _ToolCollector:
            def __init__(self) -> None:
                self.tools: dict[str, Any] = {}

            def tool(self, fn):
                self.tools[fn.__name__] = fn
                return fn

        def _wall_common_pset(result: dict[str, Any]) -> dict[str, Any]:
            payload = (result.get("data") or {}).get("payload") or {}
            psets = (payload.get("PropertySets") or {}).get("Official") or {}
            return psets.get("Pset_WallCommon") or {}

        # -------------------------------------------------------------------
        print("\n[Part 1] Build graphs in both modes")
        # -------------------------------------------------------------------
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

        # -------------------------------------------------------------------
        print("\n[Part 2] Node ID parity")
        # -------------------------------------------------------------------
        full_ids = set(G_full.nodes)
        min_ids = set(G_min.nodes)
        check(
            full_ids == min_ids,
            f"Same node IDs in both modes ({len(full_ids)} nodes)",
        )

        # -------------------------------------------------------------------
        print("\n[Part 3] Flat properties parity")
        # -------------------------------------------------------------------
        wall_nid = "Element::WALL001"
        check(wall_nid in G_full, f"{wall_nid} present in full graph")
        check(wall_nid in G_min, f"{wall_nid} present in minimal graph")

        full_props = G_full.nodes[wall_nid].get("properties") or {}
        min_props = G_min.nodes[wall_nid].get("properties") or {}
        check(full_props == min_props, "Flat properties are identical between modes")
        check(
            full_props.get("GlobalId") == "WALL001",
            "GlobalId present in flat properties",
        )
        check(full_props.get("Name") == "Wall 1", "Name present in flat properties")

        # -------------------------------------------------------------------
        print("\n[Part 4] Payload content diverges as expected")
        # -------------------------------------------------------------------
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

        # -------------------------------------------------------------------
        print("\n[Part 5] find_nodes flat filter works in both modes")
        # -------------------------------------------------------------------
        for mode_label, G in (("full", G_full), ("minimal", G_min)):
            res = query_ifc_graph(
                G,
                "find_nodes",
                {"property_filters": {"Name": "Wall 1"}},
            )
            elements = (res.get("data") or {}).get("elements", [])
            check(
                len(elements) == 1,
                (
                    f"{mode_label} mode: find_nodes flat filter Name='Wall 1' "
                    "returns 1 match"
                ),
            )

        # -------------------------------------------------------------------
        print("\n[Part 6] Dotted pset filter: full=match, minimal=no match (no DB)")
        # -------------------------------------------------------------------
        pset_filters = {"Pset_WallCommon.FireRating": "EI 90"}

        res_full = query_ifc_graph(
            G_full,
            "find_nodes",
            {"property_filters": pset_filters},
        )
        full_matches = (res_full.get("data") or {}).get("elements", [])
        check(len(full_matches) == 1, "full mode: dotted pset filter returns 1 match")

        res_min = query_ifc_graph(
            G_min,
            "find_nodes",
            {"property_filters": pset_filters},
        )
        min_matches = (res_min.get("data") or {}).get("elements", [])
        check(
            len(min_matches) == 0,
            "minimal mode (no DB): dotted pset filter returns 0 matches"
            " (graceful degradation)",
        )

        # Wire DB context and verify dotted filter parity is restored in minimal mode.
        G_min.graph["_db_path"] = tmp_db
        res_min_db = query_ifc_graph(
            G_min,
            "find_nodes",
            {"property_filters": pset_filters},
        )
        min_db_matches = (res_min_db.get("data") or {}).get("elements", [])
        check(
            len(min_db_matches) == 1,
            "minimal mode (with DB): dotted pset filter returns 1 match",
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

        # -------------------------------------------------------------------
        print("\n[Part 7] Minimal+DB flat fallback + numeric dotted-key parity")
        # -------------------------------------------------------------------
        flat_pset_filter = {"ThermalTransmittance": 0.28}
        full_flat = query_ifc_graph(
            G_full,
            "find_nodes",
            {"property_filters": flat_pset_filter},
        )
        min_flat = query_ifc_graph(
            G_min,
            "find_nodes",
            {"property_filters": flat_pset_filter},
        )
        full_flat_ids = {
            e["id"] for e in (full_flat.get("data") or {}).get("elements", [])
        }
        min_flat_ids = {
            e["id"] for e in (min_flat.get("data") or {}).get("elements", [])
        }
        check(
            full_flat_ids == {wall_nid},
            "full mode: flat pset fallback ThermalTransmittance=0.28 matches wall",
        )
        check(
            min_flat_ids == full_flat_ids,
            "minimal mode (with DB): flat pset fallback matches full mode",
        )

        numeric_dotted_filter = {"Pset_WallCommon.ThermalTransmittance": 0.28}
        full_num = query_ifc_graph(
            G_full,
            "find_nodes",
            {"property_filters": numeric_dotted_filter},
        )
        min_num = query_ifc_graph(
            G_min,
            "find_nodes",
            {"property_filters": numeric_dotted_filter},
        )
        full_num_ids = {
            e["id"] for e in (full_num.get("data") or {}).get("elements", [])
        }
        min_num_ids = {e["id"] for e in (min_num.get("data") or {}).get("elements", [])}
        check(
            full_num_ids == {wall_nid},
            "full mode: numeric dotted-key filter matches wall",
        )
        check(
            min_num_ids == full_num_ids,
            "minimal mode (with DB): numeric dotted-key parity matches full mode",
        )

        # -------------------------------------------------------------------
        print("\n[Part 8] DB-switch cache invalidation + property fidelity")
        # -------------------------------------------------------------------
        props_db1 = query_ifc_graph(
            G_min,
            "get_element_properties",
            {"element_id": "WALL001"},
        )
        wall_common_db1 = _wall_common_pset(props_db1)
        check(
            wall_common_db1.get("FireRating") == "EI 90",
            "minimal+DB1: get_element_properties returns FireRating='EI 90'",
        )

        G_min.graph["_db_path"] = tmp_db2
        props_db2 = query_ifc_graph(
            G_min,
            "get_element_properties",
            {"element_id": "WALL001"},
        )
        wall_common_db2 = _wall_common_pset(props_db2)
        check(
            wall_common_db2.get("FireRating") == "EI 30",
            "minimal+DB2: switching _db_path updates values on same graph object",
        )

        # Switch back to DB1 and validate typed decoding (float + nested dict).
        G_min.graph["_db_path"] = tmp_db
        props_typed = query_ifc_graph(
            G_min,
            "get_element_properties",
            {"element_id": "WALL001"},
        )
        wall_common_typed = _wall_common_pset(props_typed)
        thermal_value = wall_common_typed.get("ThermalTransmittance")
        performance_value = wall_common_typed.get("PerformanceData")
        check(
            isinstance(thermal_value, float) and thermal_value == 0.28,
            "minimal+DB: ThermalTransmittance preserved as float (not string)",
        )
        check(
            isinstance(performance_value, dict)
            and performance_value.get("UValue") == 0.28,
            "minimal+DB: nested PerformanceData payload preserved as dict",
        )

        # Legacy compatibility: unprefixed JSON/decimal strings should still
        # decode with best-effort fidelity.
        legacy_untyped_db = tmp_dir_path / "model_legacy_untyped.db"
        shutil.copyfile(tmp_db, legacy_untyped_db)
        with sqlite3.connect(str(legacy_untyped_db)) as conn:
            conn.execute(
                "UPDATE properties SET value = ? WHERE element_id = ? "
                "AND pset_name = ? AND property_name = ?",
                (
                    '"EI 90"',
                    101,
                    "Pset_WallCommon",
                    "FireRating",
                ),
            )
            conn.execute(
                "UPDATE properties SET value = ? WHERE element_id = ? "
                "AND pset_name = ? AND property_name = ?",
                (
                    "0.28",
                    101,
                    "Pset_WallCommon",
                    "ThermalTransmittance",
                ),
            )
            conn.execute(
                "UPDATE properties SET value = ? WHERE element_id = ? "
                "AND pset_name = ? AND property_name = ?",
                (
                    '{"UValue":0.28,"IsExternal":true}',
                    101,
                    "Pset_WallCommon",
                    "PerformanceData",
                ),
            )
            conn.execute(
                "INSERT INTO properties "
                "(element_id, pset_name, property_name, value, is_official) "
                "VALUES (?, ?, ?, ?, ?)",
                (101, "Pset_WallCommon", "LegacyIntTest", "10", 1),
            )
            conn.execute(
                "INSERT INTO properties "
                "(element_id, pset_name, property_name, value, is_official) "
                "VALUES (?, ?, ?, ?, ?)",
                (101, "Pset_WallCommon", "LegacyBoolTest", "True", 1),
            )
            conn.execute(
                "INSERT INTO properties "
                "(element_id, pset_name, property_name, value, is_official) "
                "VALUES (?, ?, ?, ?, ?)",
                (101, "Pset_WallCommon", "LegacyNoneTest", "None", 1),
            )
            conn.commit()

        G_min.graph["_db_path"] = legacy_untyped_db
        props_legacy_untyped = query_ifc_graph(
            G_min,
            "get_element_properties",
            {"element_id": "WALL001"},
        )
        wall_common_legacy_untyped = _wall_common_pset(props_legacy_untyped)
        check(
            isinstance(wall_common_legacy_untyped.get("ThermalTransmittance"), float)
            and wall_common_legacy_untyped.get("ThermalTransmittance") == 0.28,
            "minimal+legacy DB: decimal string ThermalTransmittance decodes to float",
        )
        check(
            isinstance(wall_common_legacy_untyped.get("PerformanceData"), dict),
            "minimal+legacy DB: unprefixed JSON object decodes to dict",
        )
        check(
            wall_common_legacy_untyped.get("FireRating") == "EI 90",
            "minimal+legacy DB: quoted JSON scalar decodes to plain string",
        )
        check(
            isinstance(wall_common_legacy_untyped.get("LegacyIntTest"), int)
            and wall_common_legacy_untyped.get("LegacyIntTest") == 10,
            "minimal+legacy DB: integer scalar decodes to int",
        )

        key_result_samples = list_property_keys(
            SimpleNamespace(deps=G_min),
            class_=None,
            sample_values=True,
        )
        key_samples = (key_result_samples.get("data") or {}).get("samples") or {}
        legacy_bool_samples = key_samples.get("Pset_WallCommon.LegacyBoolTest") or []
        legacy_none_samples = key_samples.get("Pset_WallCommon.LegacyNoneTest") or []
        check(
            bool(legacy_bool_samples) and isinstance(legacy_bool_samples[0], bool),
            "minimal+legacy DB: sample_values decodes LegacyBoolTest as bool",
        )
        check(
            bool(legacy_none_samples) and legacy_none_samples[0] is None,
            "minimal+legacy DB: sample_values decodes LegacyNoneTest as None",
        )

        # Simulate legacy/degraded DB row and ensure full-mode in-memory payload
        # is not clobbered by lower-fidelity DB data.
        legacy_db = tmp_dir_path / "model_legacy.db"
        shutil.copyfile(tmp_db, legacy_db)
        with sqlite3.connect(str(legacy_db)) as conn:
            conn.execute(
                "UPDATE properties SET value = ? WHERE element_id = ? "
                "AND pset_name = ? AND property_name = ?",
                (
                    "legacy-string",
                    101,
                    "Pset_WallCommon",
                    "PerformanceData",
                ),
            )
            conn.commit()

        G_full.graph["_db_path"] = legacy_db
        props_full_legacy = query_ifc_graph(
            G_full,
            "get_element_properties",
            {"element_id": "WALL001"},
        )
        wall_common_full_legacy = _wall_common_pset(props_full_legacy)
        check(
            isinstance(wall_common_full_legacy.get("PerformanceData"), dict),
            "full+DB: merge keeps richer nested in-memory payload over degraded DB",
        )

        # -------------------------------------------------------------------
        print("\n[Part 9] Invalid payload_mode falls back to 'full'")
        # -------------------------------------------------------------------
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

    if _failures:
        print(f"\n\033[31m{len(_failures)} check(s) FAILED:\033[0m")
        for f in _failures:
            print(f"  - {f}")
        return 1

    print("\n\033[32mAll checks passed.\033[0m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
