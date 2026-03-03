"""SQLite schema and column definitions for the IFC SQL database."""

from __future__ import annotations

REQUIRED_COLUMNS = frozenset({"ExpressId", "GlobalId", "Class"})

CORE_COLUMNS = [
    "ExpressId",
    "GlobalId",
    "Class",
    "PredefinedType",
    "Name",
    "Description",
    "ObjectType",
    "Tag",
    "Level",
    "TypeName",
]

SCHEMA_SQL = """\
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS elements (
    express_id      INTEGER PRIMARY KEY,
    global_id       TEXT    UNIQUE NOT NULL,
    ifc_class       TEXT    NOT NULL,
    predefined_type TEXT,
    name            TEXT,
    description     TEXT,
    object_type     TEXT,
    tag             TEXT,
    level           TEXT,
    type_name       TEXT
);

CREATE TABLE IF NOT EXISTS properties (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    element_id    INTEGER NOT NULL REFERENCES elements(express_id),
    pset_name     TEXT    NOT NULL,
    property_name TEXT    NOT NULL,
    value         TEXT,
    is_official   INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS quantities (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    element_id    INTEGER NOT NULL REFERENCES elements(express_id),
    qto_name      TEXT    NOT NULL,
    quantity_name TEXT    NOT NULL,
    value         REAL,
    is_official   INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_elements_class  ON elements(ifc_class);
CREATE INDEX IF NOT EXISTS idx_elements_level  ON elements(level);
CREATE INDEX IF NOT EXISTS idx_props_element   ON properties(element_id);
CREATE INDEX IF NOT EXISTS idx_props_pset      ON properties(pset_name);
CREATE INDEX IF NOT EXISTS idx_quants_element  ON quantities(element_id);
CREATE INDEX IF NOT EXISTS idx_quants_qto      ON quantities(qto_name);
"""
