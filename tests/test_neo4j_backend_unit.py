from __future__ import annotations

import json
from typing import Any, cast

import networkx as nx
import pytest

from rag_tag.graph.backends.neo4j_backend import Neo4jBackend
from rag_tag.graph.backends.neo4j_cypher import INSERT_NODES, INSERT_RELS
from rag_tag.parser import jsonl_to_neo4j as jsonl_to_neo4j_module


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


class _RowsSession:
    def __init__(self, rows) -> None:
        self._rows = rows

    def run(self, query: str, **params):
        return _FakeResult(rows=self._rows)


class _RowsSessionContext:
    def __init__(self, rows) -> None:
        self.session = _RowsSession(rows)

    def __enter__(self):
        return self.session

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


class _CapturingSession:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []

    def run(self, query: str, **params):
        self.calls.append((query, params))
        if "MATCH (n:Node {node_id: $node_id})" in query:
            return _FakeResult(single_record=None)
        if "AND n.global_id = $gid" in query:
            datasets = params.get("datasets")
            if datasets == ["model-a"]:
                return _FakeResult(rows=[{"id": "Element::model-a::W1"}])
            return _FakeResult(
                rows=[
                    {"id": "Element::model-a::W1"},
                    {"id": "Element::model-b::W1"},
                ]
            )
        if "MATCH (n:Node)" in query and "toLower(n.class_)" in query:
            return _FakeResult(
                rows=[
                    {
                        "n": {
                            "node_id": "Element::model-a::W1",
                            "label": "Wall A",
                            "class_": "IfcWall",
                            "dataset": "model-a",
                            "properties_json": "{}",
                            "payload_json": "{}",
                            "geometry_json": "{}",
                        }
                    }
                ]
            )
        if "MATCH (n:Node)" in query and "RETURN n" in query:
            return _FakeResult(
                rows=[
                    {
                        "n": {
                            "node_id": "Element::model-a::W1",
                            "label": "Wall A",
                            "class_": "IfcWall",
                            "dataset": "model-a",
                            "properties_json": "{}",
                            "payload_json": "{}",
                            "geometry_json": "{}",
                        }
                    }
                ]
            )
        return _FakeResult(rows=[])


class _CapturingSessionContext:
    def __init__(self) -> None:
        self.session = _CapturingSession()

    def __enter__(self):
        return self.session

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


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


@pytest.mark.parametrize(
    ("action", "params", "data_key", "rows"),
    [
        (
            "get_adjacent_elements",
            {"element_id": "Element::W1"},
            "adjacent",
            [
                {
                    "m": {
                        "node_id": "Element::D1",
                        "label": "Door 1",
                        "class_": "IfcDoor",
                    },
                    "r": {
                        "relation": "adjacent_to",
                        "distance": 0.5,
                        "source": "heuristic",
                    },
                },
                {
                    "m": {
                        "node_id": "Element::D1",
                        "label": "Door 1",
                        "class_": "IfcDoor",
                    },
                    "r": {
                        "relation": "adjacent_to",
                        "distance": 0.5,
                        "source": "heuristic",
                    },
                },
            ],
        ),
        (
            "get_topology_neighbors",
            {"element_id": "Element::W1", "relation": "intersects_3d"},
            "neighbors",
            [
                {
                    "m": {
                        "node_id": "Element::S1",
                        "label": "Slab 1",
                        "class_": "IfcSlab",
                    },
                    "r": {
                        "relation": "intersects_3d",
                        "intersection_volume": 1.0,
                        "contact_area": 0.5,
                        "source": "topology",
                    },
                },
                {
                    "m": {
                        "node_id": "Element::S1",
                        "label": "Slab 1",
                        "class_": "IfcSlab",
                    },
                    "r": {
                        "relation": "intersects_3d",
                        "intersection_volume": 1.0,
                        "contact_area": 0.5,
                        "source": "topology",
                    },
                },
            ],
        ),
    ],
)
def test_neighbor_queries_dedupe_duplicate_symmetric_hits(
    monkeypatch,
    action: str,
    params: dict[str, object],
    data_key: str,
    rows: list[dict[str, object]],
) -> None:
    backend = Neo4jBackend(graph=nx.MultiDiGraph())
    backend._conn_error = None
    monkeypatch.setattr(
        Neo4jBackend,
        "_resolve_element_id",
        lambda self, element_id: ("Element::W1", None),
    )
    monkeypatch.setattr(
        Neo4jBackend,
        "_session",
        lambda self: _RowsSessionContext(rows),
    )

    result = backend.query(action, params)

    assert result["status"] == "ok"
    assert len(result["data"][data_key]) == 1


def test_neo4j_backend_scopes_queries_to_selected_datasets(monkeypatch) -> None:
    backend = Neo4jBackend(graph=None, selected_datasets=("model-a",))
    backend._conn_error = None
    fake_context = _CapturingSessionContext()
    monkeypatch.setattr(Neo4jBackend, "_session", lambda self: fake_context)

    resolved, err = backend._resolve_element_id("W1")
    result = backend.query("find_elements_by_class", {"class": "IfcWall"})
    catalog = backend.get_networkx_graph()

    assert resolved == "Element::model-a::W1"
    assert err is None
    assert result["status"] == "ok"
    assert result["data"]["elements"][0]["id"] == "Element::model-a::W1"
    assert list(catalog.nodes) == ["Element::model-a::W1"]
    assert catalog.graph["datasets"] == ["model-a"]
    assert all(
        params.get("datasets") == ["model-a"]
        for _, params in fake_context.session.calls
    )


def test_import_networkx_graph_projects_dataset_metadata(monkeypatch) -> None:
    graph = nx.MultiDiGraph()
    graph.add_node(
        "Element::model-a::W1",
        label="Wall A",
        class_="IfcWall",
        dataset="model-a",
        properties={"GlobalId": "W1"},
        payload={},
    )
    graph.add_node(
        "Storey::model-a::S1",
        label="Level A",
        class_="IfcBuildingStorey",
        dataset="model-a",
        properties={"GlobalId": "S1"},
        payload={},
    )
    graph.add_edge(
        "Storey::model-a::S1",
        "Element::model-a::W1",
        relation="contains",
    )

    calls: list[tuple[str, dict[str, object]]] = []

    class _FakeDriverSession:
        def run(self, query: str, **params):
            calls.append((query, params))
            return _FakeResult(rows=[])

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

    class _FakeDriver:
        def session(self, database=None):
            return _FakeDriverSession()

        def close(self) -> None:
            return None

    class _FakeGraphDatabase:
        @staticmethod
        def driver(uri: str, auth: tuple[str, str]):
            return _FakeDriver()

    monkeypatch.setattr(jsonl_to_neo4j_module, "GraphDatabase", _FakeGraphDatabase)

    jsonl_to_neo4j_module.import_networkx_graph(
        graph,
        uri="bolt://test",
        username="neo4j",
        password="password",
    )

    node_rows = next(params["rows"] for query, params in calls if query == INSERT_NODES)
    rel_rows = next(params["rows"] for query, params in calls if query == INSERT_RELS)

    assert node_rows[0]["dataset"] == "model-a"
    assert rel_rows[0]["dataset"] == "model-a"


def test_import_jsonl_files_streams_base_graph_into_neo4j(
    monkeypatch,
    tmp_path,
) -> None:
    jsonl_path = tmp_path / "model-a.jsonl"
    jsonl_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "GlobalId": "S1",
                        "IfcType": "IfcBuildingStorey",
                        "Name": "Level 0",
                        "Hierarchy": {"Level": "Level 0"},
                    }
                ),
                json.dumps(
                    {
                        "GlobalId": "WT1",
                        "IfcType": "IfcWallType",
                        "Name": "Wall Type 1",
                        "Hierarchy": {},
                    }
                ),
                json.dumps(
                    {
                        "GlobalId": "W1",
                        "IfcType": "IfcWall",
                        "Name": "Wall 1",
                        "Hierarchy": {"ParentId": "S1", "Level": "Level 0"},
                        "Geometry": {
                            "Centroid": [1.0, 2.0, 3.0],
                            "BoundingBox": {
                                "min": [0.0, 0.0, 0.0],
                                "max": [2.0, 2.0, 3.0],
                            },
                        },
                        "Relationships": {
                            "typed_by": ["WT1"],
                            "ifc_connected_to": ["D1"],
                            "belongs_to_system": [" Supply   Air "],
                            "classified_as": ["UniFormat B2010"],
                        },
                    }
                ),
                json.dumps(
                    {
                        "GlobalId": "D1",
                        "IfcType": "IfcDoor",
                        "Name": "Door 1",
                        "Hierarchy": {"ParentId": "S1", "Level": "Level 0"},
                        "Relationships": {"ifc_connected_to": ["W1"]},
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )

    calls: list[tuple[str, dict[str, object]]] = []

    class _FakeDriverSession:
        def run(self, query: str, **params):
            calls.append((query, params))
            return _FakeResult(rows=[])

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

    class _FakeDriver:
        def session(self, database=None):
            return _FakeDriverSession()

        def close(self) -> None:
            return None

    class _FakeGraphDatabase:
        @staticmethod
        def driver(uri: str, auth: tuple[str, str]):
            return _FakeDriver()

    monkeypatch.setattr(jsonl_to_neo4j_module, "GraphDatabase", _FakeGraphDatabase)

    jsonl_to_neo4j_module.import_jsonl_files(
        [jsonl_path],
        uri="bolt://test",
        username="neo4j",
        password="password",
        batch_size=2,
    )

    node_rows = [
        row
        for query, params in calls
        if query == INSERT_NODES
        for row in cast(list[dict[str, object]], params["rows"])
    ]
    rel_rows = [
        row
        for query, params in calls
        if query == INSERT_RELS
        for row in cast(list[dict[str, object]], params["rows"])
    ]

    node_ids = {cast(str, row["node_id"]) for row in node_rows}
    rel_triplets = {
        (
            cast(str, row["from"]),
            cast(str, row["to"]),
            cast(str, row["relation"]),
        )
        for row in rel_rows
    }

    assert "IfcProject" in node_ids
    assert "IfcBuilding" in node_ids
    assert "Storey::S1" in node_ids
    assert "Element::WT1" in node_ids
    assert "Element::W1" in node_ids
    assert "Element::D1" in node_ids
    assert "System::Supply Air" in node_ids
    assert "Classification::UniFormat B2010" in node_ids

    assert ("IfcProject", "IfcBuilding", "aggregates") in rel_triplets
    assert ("IfcBuilding", "Storey::S1", "aggregates") in rel_triplets
    assert ("Storey::S1", "Element::W1", "contains") in rel_triplets
    assert ("Element::W1", "Storey::S1", "contained_in") in rel_triplets
    assert ("Element::W1", "Element::WT1", "typed_by") in rel_triplets
    assert ("Element::W1", "Element::D1", "ifc_connected_to") in rel_triplets
    assert ("Element::D1", "Element::W1", "ifc_connected_to") in rel_triplets
    assert ("Element::W1", "System::Supply Air", "belongs_to_system") in rel_triplets
    assert (
        "Element::W1",
        "Classification::UniFormat B2010",
        "classified_as",
    ) in rel_triplets
    assert all(
        cast(str, row["relation"])
        not in {"adjacent_to", "connected_to", "above", "below", "intersects_bbox"}
        for row in rel_rows
    )


def test_import_jsonl_files_derives_spatial_and_topology_edges(
    monkeypatch,
    tmp_path,
) -> None:
    jsonl_path = tmp_path / "model-a.jsonl"
    jsonl_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "GlobalId": "S1",
                        "IfcType": "IfcBuildingStorey",
                        "Name": "Level 0",
                        "Hierarchy": {"Level": "Level 0"},
                    }
                ),
                json.dumps(
                    {
                        "GlobalId": "W1",
                        "IfcType": "IfcWall",
                        "Name": "Wall 1",
                        "Hierarchy": {"ParentId": "S1", "Level": "Level 0"},
                        "Geometry": {
                            "Centroid": [1.0, 1.0, 1.0],
                            "BoundingBox": {
                                "min": [0.0, 0.0, 0.0],
                                "max": [2.0, 2.0, 2.0],
                            },
                        },
                    }
                ),
                json.dumps(
                    {
                        "GlobalId": "D1",
                        "IfcType": "IfcDoor",
                        "Name": "Door 1",
                        "Hierarchy": {"ParentId": "S1", "Level": "Level 0"},
                        "Geometry": {
                            "Centroid": [2.0, 2.0, 1.0],
                            "BoundingBox": {
                                "min": [1.0, 1.0, 0.0],
                                "max": [3.0, 3.0, 2.0],
                            },
                        },
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )

    calls: list[tuple[str, dict[str, object]]] = []

    class _FakeDriverSession:
        def run(self, query: str, **params):
            calls.append((query, params))
            return _FakeResult(rows=[])

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

    class _FakeDriver:
        def session(self, database=None):
            return _FakeDriverSession()

        def close(self) -> None:
            return None

    class _FakeGraphDatabase:
        @staticmethod
        def driver(uri: str, auth: tuple[str, str]):
            return _FakeDriver()

    monkeypatch.setattr(jsonl_to_neo4j_module, "GraphDatabase", _FakeGraphDatabase)

    jsonl_to_neo4j_module.import_jsonl_files(
        [jsonl_path],
        uri="bolt://test",
        username="neo4j",
        password="password",
        batch_size=10,
    )

    rel_rows = [
        row
        for query, params in calls
        if query == INSERT_RELS
        for row in cast(list[dict[str, object]], params["rows"])
    ]

    def _has_rel(source: str, target: str, relation: str, rel_source: str) -> bool:
        return any(
            cast(str, row["from"]) == source
            and cast(str, row["to"]) == target
            and cast(str, row["relation"]) == relation
            and cast(str, row["source"]) == rel_source
            for row in rel_rows
        )

    assert _has_rel("Element::W1", "Element::D1", "connected_to", "heuristic")
    assert _has_rel("Element::D1", "Element::W1", "connected_to", "heuristic")
    assert _has_rel("Element::W1", "Element::D1", "intersects_bbox", "topology")
    assert _has_rel("Element::D1", "Element::W1", "intersects_bbox", "topology")
    assert _has_rel("Element::W1", "Element::D1", "overlaps_xy", "topology")
    assert _has_rel("Element::D1", "Element::W1", "overlaps_xy", "topology")


def test_import_jsonl_files_partitioned_spatial_pass_keeps_neighbor_level_edges(
    monkeypatch,
    tmp_path,
) -> None:
    jsonl_path = tmp_path / "model-a.jsonl"
    jsonl_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "GlobalId": "W1",
                        "IfcType": "IfcWall",
                        "Name": "Wall 1",
                        "Hierarchy": {"Level": "Level 0"},
                        "Geometry": {
                            "Centroid": [1.0, 1.0, 1.0],
                            "BoundingBox": {
                                "min": [0.0, 0.0, 0.0],
                                "max": [2.0, 2.0, 2.0],
                            },
                        },
                    }
                ),
                json.dumps(
                    {
                        "GlobalId": "D1",
                        "IfcType": "IfcDoor",
                        "Name": "Door 1",
                        "Hierarchy": {"Level": "Level 1"},
                        "Geometry": {
                            "Centroid": [1.5, 1.5, 1.2],
                            "BoundingBox": {
                                "min": [0.5, 0.5, 0.2],
                                "max": [2.5, 2.5, 2.2],
                            },
                        },
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )

    calls: list[tuple[str, dict[str, object]]] = []

    class _FakeDriverSession:
        def run(self, query: str, **params):
            calls.append((query, params))
            return _FakeResult(rows=[])

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

    class _FakeDriver:
        def session(self, database=None):
            return _FakeDriverSession()

        def close(self) -> None:
            return None

    class _FakeGraphDatabase:
        @staticmethod
        def driver(uri: str, auth: tuple[str, str]):
            return _FakeDriver()

    monkeypatch.setattr(jsonl_to_neo4j_module, "GraphDatabase", _FakeGraphDatabase)
    monkeypatch.setattr(jsonl_to_neo4j_module, "FULL_DATASET_SPATIAL_LIMIT", 1)

    jsonl_to_neo4j_module.import_jsonl_files(
        [jsonl_path],
        uri="bolt://test",
        username="neo4j",
        password="password",
        batch_size=10,
    )

    rel_rows = [
        row
        for query, params in calls
        if query == INSERT_RELS
        for row in cast(list[dict[str, object]], params["rows"])
    ]

    assert any(
        cast(str, row["from"]) == "Element::W1"
        and cast(str, row["to"]) == "Element::D1"
        and cast(str, row["relation"]) == "connected_to"
        and cast(str, row["source"]) == "heuristic"
        for row in rel_rows
    )
    assert any(
        cast(str, row["from"]) == "Element::D1"
        and cast(str, row["to"]) == "Element::W1"
        and cast(str, row["relation"]) == "connected_to"
        and cast(str, row["source"]) == "heuristic"
        for row in rel_rows
    )


def test_set_context_db_path_updates_neo4j_backend_and_catalog_graph(
    tmp_path,
) -> None:
    graph = nx.MultiDiGraph()
    backend = Neo4jBackend(graph=graph)
    updated_db = tmp_path / "updated.db"

    backend.set_context_db_path(updated_db)

    assert backend.db_path == updated_db.resolve()
    assert backend.get_networkx_graph().graph["_db_path"] == updated_db.resolve()


def test_catalog_backed_find_nodes_respects_selected_datasets() -> None:
    graph = nx.MultiDiGraph()
    graph.graph["datasets"] = ["model-a", "model-b"]
    graph.add_node(
        "Element::model-a::W1",
        label="Wall A",
        class_="IfcWall",
        dataset="model-a",
        properties={"GlobalId": "W1"},
        payload={},
    )
    graph.add_node(
        "Element::model-b::W1",
        label="Wall B",
        class_="IfcWall",
        dataset="model-b",
        properties={"GlobalId": "W1"},
        payload={},
    )

    backend = Neo4jBackend(graph=graph, selected_datasets=("model-a",))

    result = backend.query("find_nodes", {"class": "IfcWall"})
    data = cast(dict[str, Any], result["data"])
    elements = cast(list[dict[str, Any]], data["elements"])

    assert result["status"] == "ok"
    assert [element["id"] for element in elements] == ["Element::model-a::W1"]
    assert list(backend.get_networkx_graph().nodes) == ["Element::model-a::W1"]
    assert backend.get_networkx_graph().graph["datasets"] == ["model-a"]
