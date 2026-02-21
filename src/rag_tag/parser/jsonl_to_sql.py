# takes the .jsonl files from ifc_to_jsonl.py and writes them into SQLite
# properties and quantities get their own tables with an is_official flag
# so we can later filter by whether a pset is IFC-standard or custom
#
# run with: uv run rag-tag-jsonl-to-sql

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
from pathlib import Path
from typing import Any

from rag_tag.parser.sql_schema import SCHEMA_SQL
from rag_tag.paths import find_project_root

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _coalesce(value: Any) -> str | None:
    if value is None:
        return None
    s = str(value).strip()
    return s if s else None


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _insert_element(conn: sqlite3.Connection, rec: dict) -> None:
    conn.execute(
        "INSERT OR IGNORE INTO elements "
        "(express_id, global_id, ifc_class, predefined_type, name, "
        "description, object_type, tag, level, type_name) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            rec.get("ExpressId"),
            _coalesce(rec.get("GlobalId")),
            _coalesce(rec.get("IfcType")) or "IfcProduct",
            _coalesce(rec.get("PredefinedType")),
            _coalesce(rec.get("Name")),
            _coalesce(rec.get("Description")),
            _coalesce(rec.get("ObjectType")),
            _coalesce(rec.get("Tag")),
            _coalesce((rec.get("Hierarchy") or {}).get("Level")),
            _coalesce(rec.get("TypeName")),
        ),
    )


def _collect_property_rows(
    express_id: int,
    psets: dict[str, dict],
    is_official: int,
) -> list[tuple]:
    rows: list[tuple] = []
    for pset_name, props in psets.items():
        if not isinstance(props, dict):
            continue
        for prop_name, value in props.items():
            if prop_name == "id":
                continue
            rows.append(
                (express_id, pset_name, prop_name, _coalesce(value), is_official)
            )
    return rows


def _collect_quantity_rows(
    express_id: int,
    quantities: dict[str, dict],
) -> list[tuple]:
    rows: list[tuple] = []
    for qto_name, qto_props in quantities.items():
        if not isinstance(qto_props, dict):
            continue
        # anything named Qto_* follows the IFC standard naming convention
        is_official = 1 if qto_name.startswith("Qto_") else 0
        for qty_name, value in qto_props.items():
            if qty_name == "id":
                continue
            rows.append(
                (express_id, qto_name, qty_name, _to_float(value), is_official)
            )
    return rows


def jsonl_to_sql(jsonl_path: Path, db_path: Path) -> tuple[int, int, int]:
    logger.info("Reading %s", jsonl_path)

    # always rebuild from scratch so we don't get stale data
    if db_path.exists():
        db_path.unlink()
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_path))
    conn.executescript(SCHEMA_SQL)

    elem_count = 0
    all_props: list[tuple] = []
    all_qties: list[tuple] = []

    with jsonl_path.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec: dict = json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning("Line %d: JSON parse error: %s", line_no, exc)
                continue

            express_id = rec.get("ExpressId")
            if express_id is None:
                logger.debug("Line %d: missing ExpressId, skipping", line_no)
                continue

            try:
                _insert_element(conn, rec)
                elem_count += 1
            except Exception as exc:
                logger.warning("Line %d: element insert failed: %s", line_no, exc)
                continue

            pset_block = rec.get("PropertySets") or {}
            all_props.extend(
                _collect_property_rows(
                    express_id,
                    pset_block.get("Official") or {},
                    is_official=1,
                )
            )
            all_props.extend(
                _collect_property_rows(
                    express_id,
                    pset_block.get("Custom") or {},
                    is_official=0,
                )
            )
            all_qties.extend(
                _collect_quantity_rows(express_id, rec.get("Quantities") or {})
            )

    # batch insert everything at once — much faster than row by row
    with conn:
        if all_props:
            conn.executemany(
                "INSERT INTO properties "
                "(element_id, pset_name, property_name, value, is_official) "
                "VALUES (?, ?, ?, ?, ?)",
                all_props,
            )
        if all_qties:
            conn.executemany(
                "INSERT INTO quantities "
                "(element_id, qto_name, quantity_name, value, is_official) "
                "VALUES (?, ?, ?, ?, ?)",
                all_qties,
            )

    conn.close()

    logger.info(
        "DB written: %s (%d elements, %d properties, %d quantities)",
        db_path,
        elem_count,
        len(all_props),
        len(all_qties),
    )
    return elem_count, len(all_props), len(all_qties)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Convert IFC JSONL exports to normalised SQLite databases."
    )
    ap.add_argument(
        "--jsonl-dir",
        type=Path,
        default=None,
        help="Directory containing .jsonl files (default: <project-root>/output/).",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Directory for .db files (default: same as --jsonl-dir).",
    )
    ap.add_argument(
        "--jsonl-file",
        type=Path,
        default=None,
        help="Single .jsonl file to convert (overrides --jsonl-dir).",
    )
    args = ap.parse_args()

    script_dir = Path(__file__).resolve().parent
    project_root = find_project_root(script_dir) or script_dir.parent.parent.parent

    if args.out_dir is not None:
        out_dir = args.out_dir.expanduser().resolve()
    else:
        out_dir = (project_root / "output").resolve()

    if args.jsonl_file is not None:
        jsonl_file = args.jsonl_file.expanduser().resolve()
        if not jsonl_file.is_file():
            print(f"JSONL file not found: {jsonl_file}")
            return
        db_path = out_dir / (jsonl_file.stem + ".db")
        elems, props, qties = jsonl_to_sql(jsonl_file, db_path)
        print(
            f"\n{jsonl_file.name} -> {db_path.name} "
            f"({elems} elements, {props} properties, {qties} quantities)"
        )
        return

    if args.jsonl_dir is not None:
        jsonl_dir = args.jsonl_dir.expanduser().resolve()
    else:
        jsonl_dir = (project_root / "output").resolve()

    if not jsonl_dir.is_dir():
        print(f"JSONL directory not found: {jsonl_dir}")
        return

    jsonl_files = sorted(jsonl_dir.glob("*.jsonl"))
    if not jsonl_files:
        print(f"No .jsonl files found in {jsonl_dir}")
        return

    print(f"Converting {len(jsonl_files)} JSONL file(s) from {jsonl_dir}")
    print(f"Writing databases to {out_dir}\n")

    for jsonl_path in jsonl_files:
        db_path = out_dir / (jsonl_path.stem + ".db")
        try:
            elems, props, qties = jsonl_to_sql(jsonl_path, db_path)
            print(
                f"  {jsonl_path.name} -> {db_path.name} "
                f"({elems} elements, {props} props, {qties} quantities)"
            )
        except Exception as exc:
            print(f"  {jsonl_path.name} -> ERROR: {exc}")

    print("\nDone.")


if __name__ == "__main__":
    main()
