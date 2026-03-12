from __future__ import annotations

import os

import networkx as nx
import pytest

from rag_tag.parser.jsonl_to_neo4j import import_networkx_graph

try:
    from neo4j import GraphDatabase  # noqa: F401
except Exception:
    GraphDatabase = None


NEO4J_ENV = all(
    os.environ.get(k) for k in ("NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD")
)


@pytest.mark.skipif(GraphDatabase is None, reason="neo4j package not installed")
@pytest.mark.skipif(not NEO4J_ENV, reason="Neo4j env not configured")
@pytest.mark.skipif(
    not os.environ.get("NEO4J_SMOKE"),
    reason="NEO4J_SMOKE not enabled",
)
def test_neo4j_import_smoke() -> None:
    G = nx.MultiDiGraph()
    G.add_node("Element::A", label="A", class_="IfcWall", properties={}, payload={})
    G.add_node("Element::B", label="B", class_="IfcDoor", properties={}, payload={})
    G.add_edge("Element::A", "Element::B", relation="adjacent_to", source="heuristic")

    import_networkx_graph(G, batch_size=100)

    uri = os.environ.get("NEO4J_URI")
    user = os.environ.get("NEO4J_USERNAME")
    password = os.environ.get("NEO4J_PASSWORD")
    database = (os.environ.get("NEO4J_DATABASE") or "").strip() or None

    driver = GraphDatabase.driver(uri, auth=(user, password))
    try:
        with driver.session(database=database) as session:
            count_nodes = session.run("MATCH (n:Node) RETURN count(n) AS c").single()[
                "c"
            ]
            count_rels = session.run(
                "MATCH ()-[r:REL]->() RETURN count(r) AS c"
            ).single()["c"]
            assert count_nodes >= 2
            assert count_rels >= 1
    finally:
        driver.close()
