"""SQLite element lookup helper for property enrichment in graph queries.

Provides parameterized lookups by GlobalId and ExpressId against the
``elements``, ``properties``, and ``quantities`` tables written by
``jsonl_to_sql.py``.

All queries use ``?`` placeholders — no string interpolation of caller inputs.
All public functions return ``None`` gracefully on any error (missing DB,
missing table, missing row, type mismatch) so that callers can fall back to
in-memory graph data without extra error-handling boilerplate.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import Any

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def open_lookup_connection(db_path: Path) -> sqlite3.Connection | None:
    """Open a reusable SQLite connection for batched element lookups."""
    if not db_path.exists():
        _logger.debug("sql_element_lookup: DB not found: %s", db_path)
        return None
    try:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
    except Exception as exc:  # noqa: BLE001
        _logger.debug("sql_element_lookup: failed to open DB %s: %s", db_path, exc)
        return None
    return conn


def lookup_element_by_globalid(
    db_path: Path,
    global_id: str,
    conn: sqlite3.Connection | None = None,
) -> dict[str, Any] | None:
    """Look up element properties from SQLite by GlobalId.

    Args:
        db_path: Path to the SQLite database file produced by
            ``rag-tag-jsonl-to-sql``.
        global_id: The IFC GlobalId string to search for.

    Returns:
        A dict with ``properties`` and ``payload`` keys whose structure matches
        the ``get_element_properties`` contract, or ``None`` if the element is
        not found or any error occurs (missing DB, schema mismatch, etc.).
    """
    if conn is not None:
        try:
            return _fetch_element(conn, "global_id = ?", global_id)
        except Exception as exc:  # noqa: BLE001
            _logger.debug(
                "sql_element_lookup: GlobalId=%r lookup failed: %s",
                global_id,
                exc,
            )
            return None

    conn_local = open_lookup_connection(db_path)
    if conn_local is None:
        return None
    try:
        return _fetch_element(conn_local, "global_id = ?", global_id)
    except Exception as exc:  # noqa: BLE001
        _logger.debug(
            "sql_element_lookup: GlobalId=%r lookup failed: %s", global_id, exc
        )
        return None
    finally:
        conn_local.close()


def lookup_element_by_express_id(
    db_path: Path,
    express_id: int,
    conn: sqlite3.Connection | None = None,
) -> dict[str, Any] | None:
    """Look up element properties from SQLite by ExpressId.

    Args:
        db_path: Path to the SQLite database file.
        express_id: The integer IFC ExpressId to search for.

    Returns:
        A dict with ``properties`` and ``payload`` keys, or ``None`` if not
        found or on any error.
    """
    if conn is not None:
        try:
            return _fetch_element(conn, "express_id = ?", express_id)
        except Exception as exc:  # noqa: BLE001
            _logger.debug(
                "sql_element_lookup: ExpressId=%r lookup failed: %s",
                express_id,
                exc,
            )
            return None

    conn_local = open_lookup_connection(db_path)
    if conn_local is None:
        return None
    try:
        return _fetch_element(conn_local, "express_id = ?", express_id)
    except Exception as exc:  # noqa: BLE001
        _logger.debug(
            "sql_element_lookup: ExpressId=%r lookup failed: %s", express_id, exc
        )
        return None
    finally:
        conn_local.close()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _fetch_element(
    conn: sqlite3.Connection,
    where_clause: str,
    param: Any,
) -> dict[str, Any] | None:
    """Fetch one element row plus its psets/quantities; return structured result.

    ``where_clause`` must be a hardcoded clause from within this module
    (never caller-supplied), ensuring parameterized query safety.
    ``param`` is always bound via the ``?`` placeholder.
    """
    row = conn.execute(
        "SELECT express_id, global_id, ifc_class, predefined_type, name, "
        "description, object_type, tag, level, type_name "
        f"FROM elements WHERE {where_clause}",
        (param,),
    ).fetchone()

    if row is None:
        return None

    eid: int = row["express_id"]

    # Build flat properties dict mirroring the node ``properties`` contract.
    properties: dict[str, Any] = {"ExpressId": eid}
    _set_if_not_none(properties, "GlobalId", row["global_id"])
    _set_if_not_none(properties, "Class", row["ifc_class"])
    _set_if_not_none(properties, "Name", row["name"])
    _set_if_not_none(properties, "TypeName", row["type_name"])
    _set_if_not_none(properties, "Level", row["level"])
    _set_if_not_none(properties, "PredefinedType", row["predefined_type"])
    _set_if_not_none(properties, "ObjectType", row["object_type"])
    _set_if_not_none(properties, "Description", row["description"])
    _set_if_not_none(properties, "Tag", row["tag"])

    # Build payload with reconstructed PropertySets and Quantities.
    psets = _fetch_psets(conn, eid)
    quantities = _fetch_quantities(conn, eid)

    payload: dict[str, Any] = {}
    if psets:
        payload["PropertySets"] = psets
    if quantities:
        payload["Quantities"] = quantities

    return {"properties": properties, "payload": payload}


def _set_if_not_none(target: dict[str, Any], key: str, value: Any) -> None:
    """Set ``target[key] = value`` only when value is not None."""
    if value is not None:
        target[key] = value


def _decode_json_value(raw_value: Any) -> Any:
    """Decode JSON-encoded DB values while remaining backward-compatible."""
    if raw_value is None:
        return None
    if isinstance(raw_value, (bool, int, float, dict, list)):
        return raw_value

    text: str
    if isinstance(raw_value, (bytes, bytearray)):
        text = raw_value.decode("utf-8", errors="replace")
    else:
        text = str(raw_value)

    if text == "":
        return ""

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def _fetch_psets(
    conn: sqlite3.Connection,
    element_id: int,
) -> dict[str, Any]:
    """Reconstruct PropertySets for *element_id* from the ``properties`` table.

    Returns a dict of the form::

        {
            "Official": {"Pset_WallCommon": {"FireRating": "EI 90", ...}, ...},
            "Custom":   {"MyPset": {...}, ...},
        }

    Returns an empty dict when there are no rows.
    """
    rows = conn.execute(
        "SELECT pset_name, property_name, value, is_official "
        "FROM properties WHERE element_id = ?",
        (element_id,),
    ).fetchall()

    official: dict[str, dict[str, Any]] = {}
    custom: dict[str, dict[str, Any]] = {}

    for row in rows:
        pset_name: str = row["pset_name"]
        prop_name: str = row["property_name"]
        value = _decode_json_value(row["value"])
        target = official if row["is_official"] else custom
        if pset_name not in target:
            target[pset_name] = {}
        target[pset_name][prop_name] = value

    result: dict[str, Any] = {}
    if official:
        result["Official"] = official
    if custom:
        result["Custom"] = custom
    return result


def _fetch_quantities(
    conn: sqlite3.Connection,
    element_id: int,
) -> dict[str, dict[str, Any]]:
    """Reconstruct Quantities for *element_id* from the ``quantities`` table.

    Returns a dict of the form::

        {"Qto_WallBaseQuantities": {"Length": 3.5, "Height": 2.7, ...}, ...}

    Returns an empty dict when there are no rows.
    """
    rows = conn.execute(
        "SELECT qto_name, quantity_name, value FROM quantities WHERE element_id = ?",
        (element_id,),
    ).fetchall()

    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        qto_name: str = row["qto_name"]
        qty_name: str = row["quantity_name"]
        value = _decode_json_value(row["value"])
        if qto_name not in result:
            result[qto_name] = {}
        result[qto_name][qty_name] = value
    return result
