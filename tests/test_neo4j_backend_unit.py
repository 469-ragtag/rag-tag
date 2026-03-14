from __future__ import annotations

import networkx as nx

from rag_tag.graph.backends.neo4j_backend import Neo4jBackend


class _FakeResult:
    def __init__(self, *, single_record=None, rows=None) -> None:
        self._single_record = single_record
        self._rows = list(rows or [])

    def single(self):
        return self._single_record

    def __iter__(self):
        return iter(self._rows)


class _FakeSession:
    def run(self, query: str, **params):
        if "MATCH (n:Node {node_id: $node_id})" in query:
            node_id = params["node_id"]
            if node_id in {"Element::W1", "Storey::S1"}:
                return _FakeResult(single_record={"n": {"node_id": node_id}})
            return _FakeResult(single_record=None)
        if "AND n.global_id = $gid RETURN n.node_id AS id" in query:
            return _FakeResult(rows=[])
        if "RETURN n" in query and "ifcbuildingstorey" in query:
            return _FakeResult(rows=[])
        return _FakeResult(rows=[])


class _FakeSessionContext:
    def __init__(self) -> None:
        self.session = _FakeSession()
        self.enter_count = 0
        self.exit_count = 0

    def __enter__(self):
        self.enter_count += 1
        return self.session

    def __exit__(self, exc_type, exc, tb) -> None:
        self.exit_count += 1


def test_resolve_element_id_uses_context_managed_session(
    monkeypatch,
) -> None:
    backend = Neo4jBackend(graph=nx.MultiDiGraph())
    fake_context = _FakeSessionContext()
    monkeypatch.setattr(Neo4jBackend, "_session", lambda self: fake_context)

    resolved, err = backend._resolve_element_id("W1")

    assert resolved == "Element::W1"
    assert err is None
    assert fake_context.enter_count == 1
    assert fake_context.exit_count == 1


def test_resolve_storey_node_uses_context_managed_session(
    monkeypatch,
) -> None:
    backend = Neo4jBackend(graph=nx.MultiDiGraph())
    fake_context = _FakeSessionContext()
    monkeypatch.setattr(Neo4jBackend, "_session", lambda self: fake_context)

    resolved, err = backend._resolve_storey_node("S1")

    assert resolved == "Storey::S1"
    assert err is None
    assert fake_context.enter_count == 1
    assert fake_context.exit_count == 1


def test_catalog_backed_geometry_actions_work_without_neo4j_env(monkeypatch) -> None:
    graph = nx.MultiDiGraph()
    graph.add_node(
        "Element::A",
        label="A",
        class_="IfcWall",
        properties={"GlobalId": "A"},
        payload={},
        bbox={"min": [0.0, 0.0, 0.0], "max": [1.0, 0.2, 2.0]},
        spatial_descriptor={
            "principal_axis": [1.0, 0.0, 0.0],
            "dominant_horizontal_direction": [1.0, 0.0, 0.0],
        },
    )
    graph.add_node(
        "Element::B",
        label="B",
        class_="IfcWall",
        properties={"GlobalId": "B"},
        payload={},
        bbox={"min": [0.0, 1.0, 0.0], "max": [1.0, 1.2, 2.0]},
        spatial_descriptor={
            "principal_axis": [1.0, 0.0, 0.0],
            "dominant_horizontal_direction": [1.0, 0.0, 0.0],
        },
    )

    monkeypatch.delenv("NEO4J_URI", raising=False)
    monkeypatch.delenv("NEO4J_USERNAME", raising=False)
    monkeypatch.delenv("NEO4J_PASSWORD", raising=False)

    backend = Neo4jBackend(graph=graph)

    result = backend.query(
        "spatial_compare",
        {"element_a": "Element::A", "element_b": "Element::B"},
    )

    assert result["status"] == "ok"
    assert result["data"]["element_a"] == "Element::A"
    assert result["data"]["element_b"] == "Element::B"
