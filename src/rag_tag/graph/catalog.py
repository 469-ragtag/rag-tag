from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator


@dataclass(slots=True)
class _NodeView:
    catalog: "GraphCatalog"

    def __call__(self, data: bool = False) -> Iterable[Any]:
        if data:
            return list(self.catalog._nodes.items())
        return list(self.catalog._nodes.keys())

    def __iter__(self) -> Iterator[str]:
        return iter(self.catalog._nodes.keys())

    def __contains__(self, node_id: object) -> bool:
        return node_id in self.catalog._nodes

    def __getitem__(self, node_id: str) -> dict[str, Any]:
        return self.catalog._nodes[node_id]

    def get(self, node_id: str, default: Any = None) -> Any:
        return self.catalog._nodes.get(node_id, default)


@dataclass(slots=True)
class GraphCatalog:
    """Small in-memory multigraph used by the Neo4j-backed runtime."""

    graph: dict[str, Any] = field(default_factory=dict)
    _nodes: dict[str, dict[str, Any]] = field(default_factory=dict)
    _out_edges: dict[str, dict[str, list[dict[str, Any]]]] = field(
        default_factory=dict
    )
    _in_edges: dict[str, dict[str, list[dict[str, Any]]]] = field(
        default_factory=dict
    )
    nodes: _NodeView = field(init=False)

    def __post_init__(self) -> None:
        self.nodes = _NodeView(self)

    def __contains__(self, node_id: object) -> bool:
        return node_id in self._nodes

    def add_node(self, node_id: str, **attrs: Any) -> None:
        node_data = self._nodes.setdefault(node_id, {})
        node_data.update(attrs)
        self._out_edges.setdefault(node_id, {})
        self._in_edges.setdefault(node_id, {})

    def add_edge(self, source_id: str, target_id: str, **attrs: Any) -> None:
        self._out_edges.setdefault(source_id, {}).setdefault(target_id, []).append(
            dict(attrs)
        )
        self._in_edges.setdefault(target_id, {}).setdefault(source_id, []).append(
            dict(attrs)
        )
        self._nodes.setdefault(source_id, {})
        self._nodes.setdefault(target_id, {})

    def successors(self, node_id: str) -> Iterator[str]:
        return iter(self._out_edges.get(node_id, {}).keys())

    def predecessors(self, node_id: str) -> Iterator[str]:
        return iter(self._in_edges.get(node_id, {}).keys())

    def get_edge_data(
        self,
        source_id: str,
        target_id: str,
    ) -> dict[int, dict[str, Any]]:
        edge_list = self._out_edges.get(source_id, {}).get(target_id, [])
        return {index: dict(edge) for index, edge in enumerate(edge_list)}

    def edges(
        self,
        keys: bool = False,
        data: bool = False,
    ) -> Iterator[Any]:
        for source_id, targets in self._out_edges.items():
            for target_id, edge_list in targets.items():
                for index, edge_data in enumerate(edge_list):
                    if keys and data:
                        yield source_id, target_id, index, dict(edge_data)
                    elif data:
                        yield source_id, target_id, dict(edge_data)
                    else:
                        yield source_id, target_id

    def out_edges(self, node_id: str, data: bool = False) -> Iterator[Any]:
        for target_id, edge_list in self._out_edges.get(node_id, {}).items():
            for edge_data in edge_list:
                if data:
                    yield node_id, target_id, dict(edge_data)
                else:
                    yield node_id, target_id

    def in_edges(self, node_id: str, data: bool = False) -> Iterator[Any]:
        for source_id, edge_list in self._in_edges.get(node_id, {}).items():
            for edge_data in edge_list:
                if data:
                    yield source_id, node_id, dict(edge_data)
                else:
                    yield source_id, node_id

    def number_of_nodes(self) -> int:
        return len(self._nodes)

    def number_of_edges(self) -> int:
        return sum(
            len(edge_list)
            for targets in self._out_edges.values()
            for edge_list in targets.values()
        )

    def is_multigraph(self) -> bool:
        return True

    def subgraph(self, node_ids: Iterable[str]) -> "GraphCatalog":
        selected = set(node_ids)
        subgraph = GraphCatalog(graph=dict(self.graph))
        for node_id in selected:
            if node_id in self._nodes:
                subgraph.add_node(node_id, **dict(self._nodes[node_id]))
        for source_id, target_id, edge_data in self.edges(data=True):
            if source_id in selected and target_id in selected:
                subgraph.add_edge(source_id, target_id, **edge_data)
        return subgraph

    def copy(self) -> "GraphCatalog":
        copied = GraphCatalog(graph=dict(self.graph))
        for node_id, node_data in self._nodes.items():
            copied.add_node(node_id, **dict(node_data))
        for source_id, target_id, edge_data in self.edges(data=True):
            copied.add_edge(source_id, target_id, **edge_data)
        return copied
