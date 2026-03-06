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
import re
import sqlite3
from ast import literal_eval
from pathlib import Path
from typing import Any

_logger = logging.getLogger(__name__)
_TYPED_JSON_PREFIX = "json:"
_LEGACY_FLOAT_RE = re.compile(r"^-?(?:\d+\.\d*|\.\d+)(?:[eE][+-]?\d+)?$")


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def open_lookup_connection(db_path: Path) -> sqlite3.Connection | None:
    """Open a SQLite connection configured for element lookup.

    Returns ``None`` on any open/configuration error so callers can
    gracefully fall back to in-memory graph payloads.
    """
    if not db_path.exists():
        _logger.debug("sql_element_lookup: DB not found: %s", db_path)
        return None

    try:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        return conn
    except Exception as exc:  # noqa: BLE001
        _logger.debug("sql_element_lookup: failed opening %s: %s", db_path, exc)
        return None


def lookup_element_by_globalid(
    db_path: Path,
    global_id: str,
    *,
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
    active_conn = conn
    owns_connection = False
    if active_conn is None:
        active_conn = open_lookup_connection(db_path)
        if active_conn is None:
            return None
        owns_connection = True

    try:
        return _fetch_element(active_conn, "global_id = ?", global_id)
    except Exception as exc:  # noqa: BLE001
        _logger.debug(
            "sql_element_lookup: GlobalId=%r lookup failed: %s", global_id, exc
        )
        return None
    finally:
        if owns_connection:
            active_conn.close()


def lookup_element_by_express_id(
    db_path: Path,
    express_id: int,
    *,
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
    active_conn = conn
    owns_connection = False
    if active_conn is None:
        active_conn = open_lookup_connection(db_path)
        if active_conn is None:
            return None
        owns_connection = True

    try:
        return _fetch_element(active_conn, "express_id = ?", express_id)
    except Exception as exc:  # noqa: BLE001
        _logger.debug(
            "sql_element_lookup: ExpressId=%r lookup failed: %s", express_id, exc
        )
        return None
    finally:
        if owns_connection:
            active_conn.close()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def decode_db_value(raw_value: Any) -> Any:
    """Decode typed/legacy DB values into best-effort Python types.

    New rows are stored with the ``json:`` prefix followed by JSON text.
    Legacy rows are decoded with conservative fallback rules.
    """
    if not isinstance(raw_value, str):
        return raw_value

    text = raw_value.strip()
    if not text:
        return raw_value

    if not text.startswith(_TYPED_JSON_PREFIX):
        # Backward-compat fallback for legacy databases written before typed
        # storage. Those rows may contain JSON-ish strings or Python bool
        # literals ("True"/"False") from ``str(value)`` coercion.

        # First, attempt strict JSON decode for any scalar/container form.
        # Handles values like:
        #   10, 0.28, true, null, "EI 90", {"A":1}, [1,2]
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # JSON booleans/null + Python-style booleans/null.
        low = text.lower()
        if low == "true":
            return True
        if low == "false":
            return False
        if low in {"none", "null"}:
            return None

        # Decode structured legacy payloads when they are valid JSON.
        if text[0] in {"{", "["}:
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                # Legacy ``str(dict)`` payloads use single quotes and are not
                # valid JSON; ``literal_eval`` safely recovers Python literals.
                try:
                    return literal_eval(text)
                except (SyntaxError, ValueError):
                    return raw_value

        # Recover numeric fidelity for common legacy decimal encodings.
        if _LEGACY_FLOAT_RE.match(text):
            try:
                return float(text)
            except ValueError:
                return raw_value

        return raw_value

    payload = text[len(_TYPED_JSON_PREFIX) :]
    try:
        return json.loads(payload)
    except json.JSONDecodeError as exc:
        _logger.debug("sql_element_lookup: invalid typed JSON payload: %s", exc)
        return raw_value


def _decode_typed_value(raw_value: Any) -> Any:
    """Backward-compatible alias for older internal call sites."""
    return decode_db_value(raw_value)


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
        value: Any = _decode_typed_value(row["value"])
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
        value: Any = _decode_typed_value(row["value"])
        if qto_name not in result:
            result[qto_name] = {}
        result[qto_name][qty_name] = value
    return result
