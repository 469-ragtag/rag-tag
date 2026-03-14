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
