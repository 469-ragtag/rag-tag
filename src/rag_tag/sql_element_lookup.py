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
_MISSING = object()
_CORE_FIELD_ALIASES: dict[str, tuple[str, str]] = {
    "expressid": ("ExpressId", "express_id"),
    "express_id": ("ExpressId", "express_id"),
    "globalid": ("GlobalId", "global_id"),
    "global_id": ("GlobalId", "global_id"),
    "class": ("Class", "ifc_class"),
    "ifcclass": ("Class", "ifc_class"),
    "ifc_class": ("Class", "ifc_class"),
    "predefinedtype": ("PredefinedType", "predefined_type"),
    "predefined_type": ("PredefinedType", "predefined_type"),
    "name": ("Name", "name"),
    "description": ("Description", "description"),
    "objecttype": ("ObjectType", "object_type"),
    "object_type": ("ObjectType", "object_type"),
    "tag": ("Tag", "tag"),
    "level": ("Level", "level"),
    "levelkey": ("LevelKey", "level_key"),
    "level_key": ("LevelKey", "level_key"),
    "typename": ("TypeName", "type_name"),
    "type_name": ("TypeName", "type_name"),
}


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


def lookup_elements_by_identifiers(
    conn: sqlite3.Connection,
    *,
    global_ids: list[str] | tuple[str, ...] = (),
    express_ids: list[int] | tuple[int, ...] = (),
) -> list[sqlite3.Row]:
    """Look up element rows by GlobalId and/or ExpressId.

    Returns rows from ``elements`` only. Empty identifier inputs yield an empty
    result without querying the database.
    """
    normalized_global_ids = [str(value) for value in global_ids if str(value).strip()]
    normalized_express_ids = [int(value) for value in express_ids]
    if not normalized_global_ids and not normalized_express_ids:
        return []

    clauses: list[str] = []
    params: list[Any] = []
    if normalized_global_ids:
        placeholders = ", ".join("?" for _ in normalized_global_ids)
        clauses.append(f"global_id IN ({placeholders})")
        params.extend(normalized_global_ids)
    if normalized_express_ids:
        placeholders = ", ".join("?" for _ in normalized_express_ids)
        clauses.append(f"express_id IN ({placeholders})")
        params.extend(normalized_express_ids)

    sql = (
        "SELECT express_id, global_id, ifc_class, predefined_type, name, "
        "description, object_type, tag, level, level_key, type_name "
        f"FROM elements WHERE {' OR '.join(clauses)}"
    )
    return list(conn.execute(sql, tuple(params)).fetchall())


def fetch_element_field_values(
    conn: sqlite3.Connection,
    *,
    express_ids: list[int] | tuple[int, ...],
    field: str,
) -> dict[str, Any]:
    """Fetch one field value for each requested element row.

    Args:
        conn: Open SQLite lookup connection.
        express_ids: Exact element rows to inspect.
        field: Core column key (for example ``Name`` or ``Level``) or a dotted
            property/quantity key such as ``Pset_DoorCommon.FireRating`` or
            ``Qto_WallBaseQuantities.NetVolume``.

    Returns:
        Dict with ``field``, ``field_source``, and ``values`` keyed by
        ``express_id``.

    Raises:
        ValueError: If the field is invalid or unsupported.
    """
    normalized_ids = [int(value) for value in express_ids]
    if not normalized_ids:
        return {"field": field, "field_source": None, "values": {}}

    field_spec = _resolve_field_spec(field)
    if field_spec["kind"] == "core":
        return {
            "field": field_spec["field"],
            "field_source": "core",
            "values": _fetch_core_field_values(
                conn,
                express_ids=normalized_ids,
                column=str(field_spec["column"]),
            ),
        }

    container_name = str(field_spec["container"])
    item_name = str(field_spec["item"])
    nested_path = tuple(str(part) for part in field_spec["nested_path"])
    property_values = _fetch_dotted_property_values(
        conn,
        express_ids=normalized_ids,
        container_name=container_name,
        item_name=item_name,
        nested_path=nested_path,
    )
    quantity_values = _fetch_dotted_quantity_values(
        conn,
        express_ids=normalized_ids,
        container_name=container_name,
        item_name=item_name,
        nested_path=nested_path,
    )

    if property_values and quantity_values:
        if container_name.lower().startswith("qto_"):
            return {
                "field": field_spec["field"],
                "field_source": "quantity",
                "values": quantity_values,
            }
        raise ValueError(
            (
                f"Ambiguous dotted field '{field}': it matches both properties "
                "and quantities."
            )
        )
    if quantity_values:
        return {
            "field": field_spec["field"],
            "field_source": "quantity",
            "values": quantity_values,
        }
    return {
        "field": field_spec["field"],
        "field_source": "property",
        "values": property_values,
    }


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


def _resolve_field_spec(field: str) -> dict[str, Any]:
    raw_field = field.strip()
    if not raw_field:
        raise ValueError("Field must be a non-empty string.")

    normalized_core = _CORE_FIELD_ALIASES.get(_normalize_field_alias(raw_field))
    if normalized_core is not None:
        canonical_field, column_name = normalized_core
        return {
            "kind": "core",
            "field": canonical_field,
            "column": column_name,
        }

    container_name, separator, remainder = raw_field.partition(".")
    if not separator or not container_name.strip() or not remainder.strip():
        raise ValueError(
            "Field must be a supported core column or dotted property/quantity key."
        )

    path_parts = [part.strip() for part in remainder.split(".") if part.strip()]
    if not path_parts:
        raise ValueError(
            "Field must be a supported core column or dotted property/quantity key."
        )

    return {
        "kind": "dotted",
        "field": raw_field,
        "container": container_name.strip(),
        "item": path_parts[0],
        "nested_path": path_parts[1:],
    }


def _normalize_field_alias(value: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "", value.strip().lower())


def _in_clause(values: list[Any] | tuple[Any, ...]) -> str:
    return ", ".join("?" for _ in values)


def _fetch_core_field_values(
    conn: sqlite3.Connection,
    *,
    express_ids: list[int],
    column: str,
) -> dict[int, Any]:
    rows = conn.execute(
        "SELECT express_id, "
        f"{column} AS field_value "
        f"FROM elements WHERE express_id IN ({_in_clause(express_ids)})",
        tuple(express_ids),
    ).fetchall()
    return {int(row["express_id"]): row["field_value"] for row in rows}


def _fetch_dotted_property_values(
    conn: sqlite3.Connection,
    *,
    express_ids: list[int],
    container_name: str,
    item_name: str,
    nested_path: tuple[str, ...],
) -> dict[int, Any]:
    rows = conn.execute(
        "SELECT element_id, value "
        "FROM properties "
        f"WHERE element_id IN ({_in_clause(express_ids)}) "
        "AND pset_name = ? AND property_name = ? "
        "ORDER BY element_id, is_official DESC, id ASC",
        (*express_ids, container_name, item_name),
    ).fetchall()

    values: dict[int, Any] = {}
    for row in rows:
        express_id = int(row["element_id"])
        if express_id in values:
            continue
        decoded = decode_db_value(row["value"])
        extracted = _extract_nested_value(decoded, nested_path)
        if extracted is _MISSING:
            continue
        values[express_id] = extracted
    return values


def _fetch_dotted_quantity_values(
    conn: sqlite3.Connection,
    *,
    express_ids: list[int],
    container_name: str,
    item_name: str,
    nested_path: tuple[str, ...],
) -> dict[int, Any]:
    rows = conn.execute(
        "SELECT element_id, value "
        "FROM quantities "
        f"WHERE element_id IN ({_in_clause(express_ids)}) "
        "AND qto_name = ? AND quantity_name = ? "
        "ORDER BY element_id, is_official DESC, id ASC",
        (*express_ids, container_name, item_name),
    ).fetchall()

    values: dict[int, Any] = {}
    for row in rows:
        express_id = int(row["element_id"])
        if express_id in values:
            continue
        extracted = _extract_nested_value(row["value"], nested_path)
        if extracted is _MISSING:
            continue
        values[express_id] = extracted
    return values


def _extract_nested_value(value: Any, nested_path: tuple[str, ...]) -> Any:
    if not nested_path:
        return value

    current = value
    for part in nested_path:
        if not isinstance(current, dict) or part not in current:
            return _MISSING
        current = current[part]
    return current


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
