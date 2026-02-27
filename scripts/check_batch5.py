"""Batch 5 smoke-tests: filter correctness, dotted-key support, _node_payload.

Run with:
    uv run python scripts/check_batch5.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

# Minimal setup so we can import ifc_graph_tool without heavy extras.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import networkx as nx  # networkx is a direct dep, no stub needed

from rag_tag.agent.graph_tools import register_graph_tools  # noqa: E402
from rag_tag.ifc_graph_tool import query_ifc_graph  # noqa: E402

# Helpers
FAIL = "\033[31mFAIL\033[0m"
PASS = "\033[32mPASS\033[0m"
_failures: list[str] = []


def check(condition: bool, name: str) -> None:
    if condition:
        print(f"  {PASS}  {name}")
    else:
        print(f"  {FAIL}  {name}")
        _failures.append(name)


class _ToolCollector:
    """Minimal stand-in for a PydanticAI agent tool registry."""

    def __init__(self) -> None:
        self.tools: dict[str, Any] = {}

    def tool(self, fn):
        self.tools[fn.__name__] = fn
        return fn


# Build a minimal test graph.
G: nx.DiGraph = nx.DiGraph()

# Wall1: has Pset_WallCommon with FireRating="EI 90" and flat GlobalId.
G.add_node(
    "Element::Wall1",
    label="Wall 1",
    class_="IfcWall",
    properties={"GlobalId": "WALL001", "Name": "Wall 1"},
    payload={
        "PropertySets": {
            "Official": {
                "Pset_WallCommon": {
                    "FireRating": "EI 90",
                    "ThermalTransmittance": 0.5,
                },
            },
            "Custom": {},
        }
    },
)

# Wall2: same pset but NO FireRating key (ThermalTransmittance only).
G.add_node(
    "Element::Wall2",
    label="Wall 2",
    class_="IfcWall",
    properties={"GlobalId": "WALL002", "Name": "Wall 2"},
    payload={
        "PropertySets": {
            "Official": {
                "Pset_WallCommon": {
                    "ThermalTransmittance": 0.3,
                },
            },
            "Custom": {},
        }
    },
)

# Wall3: FireRating explicitly stored as None (should match None filter).
G.add_node(
    "Element::Wall3",
    label="Wall 3",
    class_="IfcWall",
    properties={"GlobalId": "WALL003", "Name": "Wall 3"},
    payload={
        "PropertySets": {
            "Official": {
                "Pset_WallCommon": {
                    "FireRating": None,
                    "ThermalTransmittance": 0.4,
                },
            },
            "Custom": {},
        }
    },
)

# Door1: has a Custom pset with a unique key, and a flat prop NominalWidth.
G.add_node(
    "Element::Door1",
    label="Door 1",
    class_="IfcDoor",
    properties={"GlobalId": "DOOR001", "Name": "Door 1", "NominalWidth": 0.9},
    payload={
        "PropertySets": {
            "Official": {},
            "Custom": {
                "Pset_DoorCustom": {
                    "SecurityClass": "A",
                },
            },
        }
    },
)

# Node with no payload (legacy/stripped node).
G.add_node(
    "Element::LegacyWall",
    label="Legacy Wall",
    class_="IfcWall",
    properties={"GlobalId": "LEGWALL", "Name": "Legacy Wall"},
)

# WallNested: nested property path + malformed Custom block to test robustness.
G.add_node(
    "Element::WallNested",
    label="Wall Nested",
    class_="IfcWall",
    properties={"GlobalId": "WALLNEST", "Name": "Wall Nested"},
    payload={
        "PropertySets": {
            "Official": {
                "Pset_WallNested": {
                    "Group": {
                        "Code": "N-1",
                    }
                },
            },
            "Custom": ["not-a-dict"],
        }
    },
)


print("\n[Test 1] None false-positive: missing key must not match expected=None")

result = query_ifc_graph(
    G,
    "find_nodes",
    {"class": "IfcWall", "property_filters": {"FireRating": None}},
)
assert result["status"] == "ok"
matched_ids = {e["id"] for e in result["data"]["elements"]}

# Wall3 has FireRating=None explicitly → SHOULD match.
check("Element::Wall3" in matched_ids, "Wall3 (FireRating=None explicit) matches")
# Wall2 has no FireRating key at all → must NOT match.
check(
    "Element::Wall2" not in matched_ids,
    "Wall2 (FireRating key absent) does not match None filter",
)
# Wall1 has FireRating="EI 90" → must NOT match None.
check(
    "Element::Wall1" not in matched_ids,
    "Wall1 (FireRating='EI 90') does not match None filter",
)
# LegacyWall has no payload at all → must NOT match.
check(
    "Element::LegacyWall" not in matched_ids,
    "LegacyWall (no payload) does not match None filter",
)


print('\n[Test 2] Dotted key filter: "Pset_WallCommon.FireRating" = "EI 90"')

result2 = query_ifc_graph(
    G,
    "find_nodes",
    {"class": "IfcWall", "property_filters": {"Pset_WallCommon.FireRating": "EI 90"}},
)
assert result2["status"] == "ok"
matched2 = {e["id"] for e in result2["data"]["elements"]}

check("Element::Wall1" in matched2, "Wall1 matches dotted FireRating=EI 90")
check(
    "Element::Wall2" not in matched2, "Wall2 (no FireRating) excluded by dotted filter"
)
check(
    "Element::Wall3" not in matched2,
    "Wall3 (FireRating=None) excluded by dotted filter",
)


print('\n[Test 3] Dotted key in Custom pset: "Pset_DoorCustom.SecurityClass" = "A"')

result3 = query_ifc_graph(
    G,
    "find_nodes",
    {"class": "IfcDoor", "property_filters": {"Pset_DoorCustom.SecurityClass": "A"}},
)
assert result3["status"] == "ok"
matched3 = {e["id"] for e in result3["data"]["elements"]}

check("Element::Door1" in matched3, "Door1 matches Custom pset dotted key")
check(len(matched3) == 1, f"Exactly 1 match (got {len(matched3)})")


print('\n[Test 3b] Nested dotted key: "Pset_WallNested.Group.Code" = "N-1"')

result3b = query_ifc_graph(
    G,
    "find_nodes",
    {"class": "IfcWall", "property_filters": {"Pset_WallNested.Group.Code": "N-1"}},
)
assert result3b["status"] == "ok"
matched3b = {e["id"] for e in result3b["data"]["elements"]}

check("Element::WallNested" in matched3b, "WallNested matches nested dotted key")
check(len(matched3b) == 1, f"Exactly 1 nested dotted match (got {len(matched3b)})")


print("\n[Test 4] Flat key backward compat: GlobalId in direct properties")

result4 = query_ifc_graph(
    G,
    "find_nodes",
    {"property_filters": {"GlobalId": "WALL001"}},
)
assert result4["status"] == "ok"
matched4 = {e["id"] for e in result4["data"]["elements"]}

check("Element::Wall1" in matched4, "Wall1 matched by flat GlobalId filter")
check(len(matched4) == 1, f"Exactly 1 match (got {len(matched4)})")


print("\n[Test 5] Flat key fallback: ThermalTransmittance in nested pset")

result5 = query_ifc_graph(
    G,
    "find_nodes",
    {"class": "IfcWall", "property_filters": {"ThermalTransmittance": 0.5}},
)
assert result5["status"] == "ok"
matched5 = {e["id"] for e in result5["data"]["elements"]}

check(
    "Element::Wall1" in matched5,
    "Wall1 matched by flat ThermalTransmittance=0.5 fallback",
)
check(
    "Element::Wall2" not in matched5,
    "Wall2 (0.3) excluded by ThermalTransmittance filter",
)
check("Element::LegacyWall" not in matched5, "LegacyWall (no pset) excluded")


print("\n[Test 6] _node_payload: payload field present in find_nodes results")

result6 = query_ifc_graph(G, "find_nodes", {"class": "IfcWall"})
assert result6["status"] == "ok"
elements6 = result6["data"]["elements"]

check(len(elements6) >= 1, f"find_nodes returned {len(elements6)} IfcWall elements")

for elem in elements6:
    check("payload" in elem, f"Node {elem['id']} result contains 'payload' field")
    check("properties" in elem, f"Node {elem['id']} result contains 'properties' field")
    check("id" in elem, f"Node {elem['id']} result contains 'id' field")
    check("class_" in elem, f"Node {elem['id']} result contains 'class_' field")

# LegacyWall (no payload in graph) should still have the key but with None value.
legacy = next((e for e in elements6 if e["id"] == "Element::LegacyWall"), None)
if legacy is not None:
    check("payload" in legacy, "LegacyWall result has 'payload' key (may be None)")


print("\n[Test 7] list_property_keys tool: robust nested/dotted enumeration")

collector = _ToolCollector()
register_graph_tools(collector)
list_property_keys_tool = collector.tools["list_property_keys"]
ctx = SimpleNamespace(deps=G)
result7 = list_property_keys_tool(ctx, class_=None, sample_values=True)

check(result7.get("status") == "ok", "list_property_keys returns status=ok")
all_keys = set((result7.get("data") or {}).get("keys") or [])

check(
    "Pset_WallCommon.FireRating" in all_keys,
    "'Pset_WallCommon.FireRating' in list_property_keys output",
)
check(
    "Pset_WallCommon.ThermalTransmittance" in all_keys,
    "'Pset_WallCommon.ThermalTransmittance' in list_property_keys output",
)
check(
    "Pset_DoorCustom.SecurityClass" in all_keys,
    "'Pset_DoorCustom.SecurityClass' in list_property_keys output",
)
check(
    "Pset_WallNested.Group.Code" in all_keys,
    "Nested dotted key 'Pset_WallNested.Group.Code' is discoverable",
)
check(
    "GlobalId" in all_keys,
    "'GlobalId' flat key in list_property_keys output",
)
check(
    "Name" in all_keys,
    "'Name' flat key in list_property_keys output",
)


print("\n[Test 8] Dotted key for non-existent pset returns zero matches")

result8 = query_ifc_graph(
    G,
    "find_nodes",
    {"property_filters": {"Pset_NonExistent.SomeKey": "value"}},
)
assert result8["status"] == "ok"
matched8 = result8["data"]["elements"]
check(len(matched8) == 0, f"No nodes match non-existent pset (got {len(matched8)})")


print()
if _failures:
    print(f"\033[31m{len(_failures)} check(s) FAILED:\033[0m")
    for f in _failures:
        print(f"  - {f}")
    sys.exit(1)
else:
    print("\033[32mAll checks passed.\033[0m")
