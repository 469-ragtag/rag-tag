# rag-tag

Utilities for working with IFC files.

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
