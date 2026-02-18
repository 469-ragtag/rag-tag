"""Convert IFC CSV exports to a flat, query-ready SQLite database."""

from __future__ import annotations

import argparse
import logging
import sqlite3
from pathlib import Path

import pandas as pd

from rag_tag.parser.sql_schema import CORE_COLUMNS, REQUIRED_COLUMNS, SCHEMA_SQL
from rag_tag.paths import find_project_root

logger = logging.getLogger(__name__)


class CsvToSqlError(Exception):
    """Base exception for csv_to_sql pipeline errors."""


class InvalidCsvError(CsvToSqlError):
    """Raised when a CSV file is missing required columns or is empty."""


# Check the CSV has the columns we need before doing any work
def _validate_csv(df: pd.DataFrame, csv_path: Path) -> None:
    if df.empty:
        raise InvalidCsvError(f"CSV is empty: {csv_path}")
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise InvalidCsvError(
            f"CSV {csv_path.name} missing required columns: {sorted(missing)}"
        )


# Turn NaN and empty strings into None, otherwise return the cleaned string
def _coalesce(value: object) -> str | None:
    if pd.isna(value):
        return None
    s = str(value).strip()
    return s if s else None


# Safe float conversion returns None instead of crashing on bad data
def _to_float(value: object) -> float | None:
    if pd.isna(value):
        return None
    if isinstance(value, str) and value.strip() == "":
        return None
    try:
        return float(value)
    except (ValueError, TypeError) as exc:
        logger.debug("Non-numeric QTO value %r: %s", value, exc)
        return None


# Map each CSV row to a tuple and batch-insert into the elements table
def _insert_elements(conn: sqlite3.Connection, df: pd.DataFrame) -> int:
    rows = [
        (
            int(row.ExpressId),
            str(row.GlobalId),
            str(row.Class),
            _coalesce(getattr(row, "PredefinedType", "")),
            _coalesce(getattr(row, "Name", "")),
            _coalesce(getattr(row, "Description", "")),
            _coalesce(getattr(row, "ObjectType", "")),
            _coalesce(getattr(row, "Tag", "")),
            _coalesce(getattr(row, "Level", "")),
            _coalesce(getattr(row, "TypeName", "")),
        )
        for row in df.itertuples(index=False)
    ]
    # All values use ? placeholders to prevent SQL injection
    conn.executemany(
        "INSERT OR IGNORE INTO elements "
        "(express_id, global_id, ifc_class, predefined_type, name, "
        "description, object_type, tag, level, type_name) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    return len(rows)


# Take wide CSV columns like "Pset_SlabCommon.FireRating" and normalise them
# into rows in the properties and quantities tables
def _insert_properties_and_quantities(
    conn: sqlite3.Connection, df: pd.DataFrame
) -> tuple[int, int]:
    # Find columns with a dot that aren't core fields (these are pset/qto columns)
    pset_qto_cols = [c for c in df.columns if "." in c and c not in CORE_COLUMNS]

    prop_rows: list[tuple[int, str, str, str]] = []
    qty_rows: list[tuple[int, str, str, float | None]] = []

    # Process each pset/qto column and collect non-empty values
    for col in pset_qto_cols:
        set_name, prop_name = col.split(".", 1)
        mask = df[col].notna() & (df[col] != "")
        subset = df.loc[mask, ["ExpressId", col]]

        for _, row in subset.iterrows():
            eid = int(row["ExpressId"])
            value = row[col]

            # Pset_ columns are text properties (fire rating, acoustic rating)
            if set_name.startswith("Pset_"):
                prop_rows.append((eid, set_name, prop_name, str(value)))
            # Qto_ columns are numeric quantities (length, volume, area)
            elif set_name.startswith("Qto_"):
                qty_rows.append((eid, set_name, prop_name, _to_float(value)))

    # Batch insert for performance one call instead of N individual inserts
    if prop_rows:
        conn.executemany(
            "INSERT INTO properties "
            "(element_id, pset_name, property_name, value) VALUES (?, ?, ?, ?)",
            prop_rows,
        )
    if qty_rows:
        conn.executemany(
            "INSERT INTO quantities "
            "(element_id, qto_name, quantity_name, value) VALUES (?, ?, ?, ?)",
            qty_rows,
        )
    return len(prop_rows), len(qty_rows)


def csv_to_sql(csv_path: Path, db_path: Path) -> Path:
    """Convert an IFC CSV export to a normalised SQLite database.

    Raises ``InvalidCsvError`` if the CSV is empty or missing required columns.
    """
    logger.info("Reading CSV: %s", csv_path)
    df = pd.read_csv(csv_path)
    logger.info("Loaded %d rows, %d columns", len(df), len(df.columns))

    _validate_csv(df, csv_path)

    # Remove old db if it exists so we start fresh
    if db_path.exists():
        db_path.unlink()
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_path))
    conn.executescript(SCHEMA_SQL)

    # with conn auto-commits on success, auto-rolls-back on exception
    with conn:
        elem_count = _insert_elements(conn, df)
        logger.info("Inserted %d elements", elem_count)

        prop_count, qty_count = _insert_properties_and_quantities(conn, df)
        logger.info("Inserted %d properties, %d quantities", prop_count, qty_count)

    conn.close()

    logger.info("Database written: %s", db_path)
    return db_path


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    ap = argparse.ArgumentParser(
        description="Convert IFC CSV exports to normalised SQLite databases."
    )
    ap.add_argument(
        "--csv-dir",
        type=Path,
        default=None,
        help="Directory containing CSV files (default: <project-root>/output).",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Directory to write .db files (default: same as --csv-dir).",
    )
    args = ap.parse_args()

    script_dir = Path(__file__).resolve().parent
    project_root = find_project_root(script_dir) or script_dir.parent

    csv_dir = (args.csv_dir or project_root / "output").resolve()
    out_dir = (args.out_dir or csv_dir).resolve()

    csv_files = sorted(csv_dir.glob("*.csv"))
    if not csv_files:
        logger.error("No CSV files found in %s", csv_dir)
        return

    logger.info("Found %d CSV file(s) in %s", len(csv_files), csv_dir)

    # Convert each CSV into its own .db file
    for csv_path in csv_files:
        db_name = csv_path.stem + ".db"
        db_path = out_dir / db_name
        try:
            csv_to_sql(csv_path, db_path)
        except Exception:
            logger.exception("Failed to convert %s", csv_path.name)


if __name__ == "__main__":
    main()
