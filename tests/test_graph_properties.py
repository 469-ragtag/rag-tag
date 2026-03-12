from __future__ import annotations

import networkx as nx

from rag_tag.graph.properties import apply_property_filters


def test_property_filters_flat_and_dotted() -> None:
    G = nx.DiGraph()
    G.add_node(
        "Element::W1",
        properties={"Name": "Wall 1", "GlobalId": "W1"},
        payload={
            "PropertySets": {
                "Official": {"Pset_WallCommon": {"FireRating": "EI 90"}},
                "Custom": {},
            },
            "Quantities": {"Qto_WallBaseQuantities": {"Length": 5.0}},
        },
    )

    assert apply_property_filters(G, "Element::W1", {"Name": "Wall 1"})
    assert apply_property_filters(
        G, "Element::W1", {"Pset_WallCommon.FireRating": "EI 90"}
    )
    assert apply_property_filters(
        G, "Element::W1", {"Qto_WallBaseQuantities.Length": 5.0}
    )
    assert not apply_property_filters(
        G, "Element::W1", {"Pset_WallCommon.FireRating": "EI 60"}
    )
