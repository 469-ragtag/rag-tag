"""Compatibility wrapper for ontology-map generation.

Use this for direct script execution:

    uv run python scripts/generate_ontology_map.py --help

The canonical command remains:

    uv run rag-tag-generate-ontology-map
"""

from __future__ import annotations

from rag_tag.cli.generate_ontology_map import main

if __name__ == "__main__":
    main()
