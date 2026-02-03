# rag-tag

Utilities for working with IFC files.

## Formatting & Linting (Ruff)

This repo uses Ruff for consistent formatting and basic linting across contributors.

### One-time setup

Install dev tools (Ruff + pre-commit):

```bash
uv sync --group dev
```

Install the git pre-commit hook:

```bash
uv run pre-commit install
```

### Run manually (recommended before pushing)

Format:

```bash
uv run ruff format .
```

Lint (auto-fix safe issues):

```bash
uv run ruff check --fix .
```

Or run exactly what pre-commit runs:

```bash
uv run pre-commit run --all-files
```

## IFC → CSV

The script in [parser/ifc-to-csv.py](parser/ifc-to-csv.py) reads `.ifc` files and exports CSV output.

### Default behavior

- Input: auto-detects an `IFC-Files/` folder by searching upward from the script location (so running from `parser/` works).
- Output: writes CSVs to `output/` at the project root.

Run:

```bash
cd parser
uv run ifc-to-csv.py
```

### Override paths

```bash
uv run ifc-to-csv.py --ifc-dir ./IFC-Files --out-dir ./output
```
