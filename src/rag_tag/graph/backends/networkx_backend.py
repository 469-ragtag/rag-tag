from __future__ import annotations

import logging
import re
from collections import deque
from typing import Any, Iterable

import networkx as nx

from rag_tag.graph.payloads import (
    INTERNAL_PAYLOAD_MODE,
    build_node_payload,
    resolve_payload_mode,
)
from rag_tag.graph.properties import (
    cached_db_lookup,
    clear_runtime_db_caches,
    collect_dotted_keys_from_sqlite,
    get_property_key_cache,
    merge_db_element_data,
)
from rag_tag.graph.types import GraphRuntime
from rag_tag.graph_contract import (
    CANONICAL_ACTIONS,
    CANONICAL_RELATION_SET,
    KNOWN_RELATION_SOURCE_SET,
    SPATIAL_RELATIONS,
    TOPOLOGY_RELATIONS,
    is_allowed_action,
    make_error_envelope,
    make_ok_envelope,
    normalize_action_name,
    normalize_relation_name,
    normalize_relation_source,
    relation_bucket,
)

_logger = logging.getLogger(__name__)


def _ok_action(action: str, data: dict[str, Any] | None) -> dict[str, Any]:
    return make_ok_envelope(action, data)


def _err(message: str, code: str, details: dict | None = None) -> dict[str, Any]:
    return make_error_envelope(message, code, details)


class NetworkXGraphBackend:
    """Concrete graph backend backed by a NetworkX graph."""

    name = "networkx"

    def load(
        self,
        *,
        dataset: str | None = None,
        payload_mode: str | None = None,
    ) -> GraphRuntime:
        from rag_tag.parser.jsonl_to_graph import build_graph  # noqa: PLC0415

        graph = build_graph(dataset=dataset, payload_mode=payload_mode)
        raw_datasets = graph.graph.get("datasets")
        datasets = (
            sorted(raw_datasets)
            if isinstance(raw_datasets, list)
            and all(isinstance(item, str) for item in raw_datasets)
            else ([dataset] if dataset else [])
        )
        runtime_payload_mode = str(graph.graph.get("_payload_mode", "full")).lower()
        return GraphRuntime(
            backend_name=self.name,
            backend=self,
            selected_datasets=datasets,
            payload_mode=runtime_payload_mode,
            context_db_path=None,
            backend_handle=graph,
        )

    def close(self, runtime: GraphRuntime) -> None:
        clear_runtime_db_caches(runtime)

    def query(
        self,
        runtime: GraphRuntime,
        action: str,
        params: dict[str, Any],
        payload_mode: str,
    ) -> dict[str, Any]:
        G = runtime.backend_handle
        if not isinstance(G, (nx.DiGraph, nx.MultiDiGraph)):
            return _err("NetworkX backend handle is invalid", "invalid")

        if not isinstance(action, str):
            return _err("Invalid action: action must be a string", "invalid")

        action = normalize_action_name(action)
        if not is_allowed_action(action):
            return _err(
                f"Unknown action: {action}",
                "unknown_action",
                {"allowed_actions": sorted(CANONICAL_ACTIONS)},
            )

        if not isinstance(params, dict):
            return _err("Invalid params: params must be an object", "invalid")

        resolved_payload_mode = resolve_payload_mode(payload_mode)

        def normalize_class(value: str) -> str:
            v = value.strip()
            if not v:
                return v
            if not v.lower().startswith("ifc"):
                v = f"Ifc{v}"
            return v

        def find_nodes_by_label(
            label: str, class_filter: str | None = None
        ) -> list[str]:
            target = label.strip().lower()
            if not target:
                return []
            matches = []
            for n, d in G.nodes(data=True):
                lbl = str(d.get("label", "")).strip().lower()
                if lbl != target:
                    continue
                if class_filter is not None:
                    if str(d.get("class_", "")).lower() != class_filter.lower():
                        continue
                matches.append(n)
            return matches

        def normalize_text(value: str) -> str:
            return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()

        def edge_relation(edge: dict[str, Any]) -> str | None:
            relation = normalize_relation_name(edge.get("relation"))
            if relation in CANONICAL_RELATION_SET:
                return relation
            return None

        def iter_edge_dicts(u: str, v: str) -> Iterable[dict[str, Any]]:
            edge_data = G.get_edge_data(u, v)
            if edge_data is None:
                return ()
            if G.is_multigraph():
                if isinstance(edge_data, dict):
                    return tuple(
                        attrs for attrs in edge_data.values() if isinstance(attrs, dict)
                    )
                return ()
            if isinstance(edge_data, dict):
                return (edge_data,)
            return ()

        def expected_source_for_relation(relation: str | None) -> str | None:
            bucket = relation_bucket(relation)
            if bucket == "explicit_ifc":
                return "ifc"
            if bucket == "topology":
                if relation in {
                    "space_bounded_by",
                    "bounds_space",
                    "path_connected_to",
                }:
                    return "ifc"
                return "topology"
            if bucket == "spatial":
                return "heuristic"
            return None

        def edge_source(
            edge: dict[str, Any], relation: str | None = None
        ) -> str | None:
            canonical_relation = (
                relation if relation in CANONICAL_RELATION_SET else None
            )
            if canonical_relation is None:
                canonical_relation = edge_relation(edge)

            bucket = relation_bucket(canonical_relation)
            if bucket is None or bucket == "hierarchy":
                return None

            expected_source = expected_source_for_relation(canonical_relation)
            if expected_source is not None:
                return expected_source

            source = normalize_relation_source(edge.get("source"))
            if source in KNOWN_RELATION_SOURCE_SET:
                return source
            return None

        def resolve_element_id(
            element_id: str,
        ) -> tuple[str | None, dict[str, Any] | None]:
            def element_candidates_by_global_id(global_id: str) -> list[str]:
                matches: list[str] = []
                for n, d in G.nodes(data=True):
                    if not str(n).startswith("Element::"):
                        continue
                    gid = d.get("properties", {}).get("GlobalId")
                    if gid == global_id:
                        matches.append(n)
                return matches

            def element_candidates_by_legacy_id(raw_id: str) -> list[str]:
                suffix = raw_id.strip()
                if not suffix:
                    return []
                if suffix.startswith("Element::"):
                    suffix = suffix.split("::", 1)[1]
                matches: list[str] = []
                for n in G.nodes:
                    node_id = str(n)
                    if not node_id.startswith("Element::"):
                        continue
                    if node_id.endswith(f"::{suffix}"):
                        matches.append(node_id)
                return matches

            if not isinstance(element_id, str):
                return None, {
                    "error": "Invalid element_id: element_id must be a string"
                }
            if element_id in G:
                if str(element_id).startswith("Element::"):
                    return element_id, None
                return None, {
                    "error": f"Invalid element_id (not an element): {element_id}"
                }
            matches = element_candidates_by_legacy_id(element_id)
            if not matches:
                matches = element_candidates_by_global_id(element_id)
            if len(matches) == 1:
                return matches[0], None
            if len(matches) > 1:
                return None, {"error": "Ambiguous element_id", "candidates": matches}
            return None, {"error": f"Element not found: {element_id}"}

        def resolve_storey_node(
            storey_query: str,
        ) -> tuple[str | None, dict[str, Any] | None]:
            query = storey_query.strip()
            if not query:
                return None, {"error": "Missing param: storey"}

            direct = query if query.startswith("Storey::") else f"Storey::{query}"
            if direct in G and (
                str(G.nodes[direct].get("class_", "")).lower() == "ifcbuildingstorey"
            ):
                return direct, None

            legacy_suffix = (
                query.split("::", 1)[1] if query.startswith("Storey::") else query
            )
            legacy_matches: list[str] = []
            for n, d in G.nodes(data=True):
                node_id = str(n)
                if not node_id.startswith("Storey::"):
                    continue
                if str(d.get("class_", "")).lower() != "ifcbuildingstorey":
                    continue
                if node_id.endswith(f"::{legacy_suffix}"):
                    legacy_matches.append(node_id)
            if len(legacy_matches) == 1:
                return legacy_matches[0], None
            if len(legacy_matches) > 1:
                return None, {"error": "Ambiguous storey", "candidates": legacy_matches}

            exact = find_nodes_by_label(query, class_filter="IfcBuildingStorey")
            if len(exact) == 1:
                return exact[0], None
            if len(exact) > 1:
                return None, {"error": "Ambiguous storey", "candidates": exact}

            qn = normalize_text(query)
            norm_matches: list[str] = []
            for n, d in G.nodes(data=True):
                if str(d.get("class_", "")).lower() != "ifcbuildingstorey":
                    continue
                if normalize_text(str(d.get("label", ""))) == qn:
                    norm_matches.append(n)
            if len(norm_matches) == 1:
                return norm_matches[0], None
            if len(norm_matches) > 1:
                return None, {"error": "Ambiguous storey", "candidates": norm_matches}

            return None, {"error": f"Storey not found: {storey_query}"}

        def storey_elements(start: str) -> Iterable[str]:
            visited = {start}
            q = deque([start])
            while q:
                node = q.popleft()
                for nbr in G.successors(node):
                    has_contains = any(
                        normalize_relation_name(edge.get("relation")) == "contains"
                        for edge in iter_edge_dicts(node, nbr)
                    )
                    if not has_contains:
                        continue
                    if nbr in visited:
                        continue
                    visited.add(nbr)
                    q.append(nbr)
                    yield nbr

        def spatial_neighbors(node_id: str) -> Iterable[tuple[str, dict[str, Any]]]:
            seen: set[tuple[str, str, str]] = set()
            spatial_relations = set(SPATIAL_RELATIONS)

            for nbr in G.successors(node_id):
                for edge in iter_edge_dicts(node_id, nbr):
                    relation = normalize_relation_name(edge.get("relation"))
                    if relation not in spatial_relations:
                        continue
                    dedupe_key = (nbr, relation or "", str(edge.get("distance")))
                    if dedupe_key in seen:
                        continue
                    seen.add(dedupe_key)
                    yield nbr, edge

            for nbr in G.predecessors(node_id):
                for edge in iter_edge_dicts(nbr, node_id):
                    relation = normalize_relation_name(edge.get("relation"))
                    if relation not in spatial_relations:
                        continue
                    dedupe_key = (nbr, relation or "", str(edge.get("distance")))
                    if dedupe_key in seen:
                        continue
                    seen.add(dedupe_key)
                    yield nbr, edge

        def topology_neighbors(
            node_id: str,
            allowed_relations: set[str] | None = None,
        ) -> Iterable[tuple[str, dict[str, Any]]]:
            seen: set[tuple[str, str, str]] = set()
            topology_relations = set(TOPOLOGY_RELATIONS)
            if allowed_relations is None:
                allowed_relations = topology_relations

            for nbr in G.successors(node_id):
                for edge in iter_edge_dicts(node_id, nbr):
                    relation = normalize_relation_name(edge.get("relation"))
                    if relation not in allowed_relations or relation is None:
                        continue
                    key = (
                        nbr,
                        relation,
                        str(
                            edge.get("vertical_gap")
                            or edge.get("intersection_volume")
                            or ""
                        ),
                    )
                    if key in seen:
                        continue
                    seen.add(key)
                    yield nbr, edge

            for nbr in G.predecessors(node_id):
                for edge in iter_edge_dicts(nbr, node_id):
                    relation = normalize_relation_name(edge.get("relation"))
                    if relation not in allowed_relations or relation is None:
                        continue
                    key = (
                        nbr,
                        relation,
                        str(
                            edge.get("vertical_gap")
                            or edge.get("intersection_volume")
                            or ""
                        ),
                    )
                    if key in seen:
                        continue
                    seen.add(key)
                    yield nbr, edge

        def apply_property_filters(
            node_id: str,
            filters: dict[str, Any],
            *,
            db_conn: Any | None = None,
        ) -> bool:
            if not filters:
                return True
            data = G.nodes[node_id]
            props = data.get("properties") or {}
            if not isinstance(props, dict):
                props = {}

            payload = data.get("payload") or {}
            if not isinstance(payload, dict):
                payload = {}

            pset_block = payload.get("PropertySets") or {}
            if not isinstance(pset_block, dict):
                pset_block = {}

            effective_quantities: dict[str, Any] = payload.get("Quantities") or {}
            if not isinstance(effective_quantities, dict):
                effective_quantities = {}

            def match_flat_property_value(prop_val: Any, expected: Any) -> bool:
                if isinstance(prop_val, list):
                    if isinstance(expected, str):
                        return expected in prop_val
                    if isinstance(expected, list):
                        return all(v in prop_val for v in expected)
                    return False
                return prop_val == expected

            has_dotted_filter = any("." in k for k in filters)
            needs_flat_pset_fallback = any(
                key not in props or not match_flat_property_value(props[key], expected)
                for key, expected in filters.items()
                if "." not in key
            )

            if (
                (has_dotted_filter or needs_flat_pset_fallback)
                and not pset_block
                and not effective_quantities
            ):
                db_path = runtime.context_db_path
                if db_path is not None:
                    db_data = cached_db_lookup(
                        runtime,
                        node_id,
                        db_path,
                        db_conn=db_conn,
                    )
                    if db_data is not None:
                        db_payload: dict[str, Any] = db_data.get("payload") or {}
                        if isinstance(db_payload, dict):
                            enriched_psets = db_payload.get("PropertySets") or {}
                            if isinstance(enriched_psets, dict):
                                pset_block = enriched_psets
                            enriched_qty = db_payload.get("Quantities") or {}
                            if isinstance(enriched_qty, dict):
                                effective_quantities = enriched_qty

            def iter_psets() -> Iterable[tuple[str, dict[str, Any]]]:
                for section in ("Official", "Custom"):
                    section_block = pset_block.get(section) or {}
                    if not isinstance(section_block, dict):
                        continue
                    for raw_name, pset_props in section_block.items():
                        if not isinstance(pset_props, dict):
                            continue
                        yield str(raw_name), pset_props

                for qto_name, qto_data in effective_quantities.items():
                    if isinstance(qto_data, dict):
                        yield str(qto_name), qto_data

            def nested_lookup(
                mapping: dict[str, Any],
                dotted_path: str,
            ) -> tuple[bool, Any]:
                current: Any = mapping
                for part in dotted_path.split("."):
                    if not isinstance(current, dict) or part not in current:
                        return False, None
                    current = current[part]
                return True, current

            def match_dotted(pset_name: str, prop_name: str, expected: Any) -> bool:
                for current_pset, pset_props in iter_psets():
                    if current_pset != pset_name:
                        continue
                    exists, value = nested_lookup(pset_props, prop_name)
                    if exists and value == expected:
                        return True
                return False

            def match_flat_in_psets(key: str, expected: Any) -> bool:
                for _pset_name, pset_props in iter_psets():
                    if key in pset_props and pset_props[key] == expected:
                        return True
                return False

            for key, expected in filters.items():
                if "." in key:
                    pset_name, _, prop_name = key.partition(".")
                    if not match_dotted(pset_name, prop_name, expected):
                        return False
                    continue

                if key in props:
                    prop_val = props[key]
                    if match_flat_property_value(prop_val, expected):
                        continue

                if not match_flat_in_psets(key, expected):
                    return False

            return True

        if action == "get_elements_in_storey":
            storey = params.get("storey")
            if not storey:
                return _err("Missing param: storey", "missing_param")
            if not isinstance(storey, str):
                return _err("Invalid param: storey must be a string", "invalid")
            node, err = resolve_storey_node(storey)
            if err:
                error_msg = err.get("error", "Unknown error")
                if "Ambiguous" in str(error_msg):
                    return _err(
                        str(error_msg),
                        "ambiguous",
                        {"candidates": err.get("candidates", [])},
                    )
                return _err(str(error_msg), "not_found")
            if node is None:
                return _err(f"Storey not found: {storey}", "not_found")

            container_classes = {
                "IfcProject",
                "IfcSite",
                "IfcBuilding",
                "IfcBuildingStorey",
                "IfcSpace",
                "IfcZone",
                "IfcSpatialZone",
                "IfcTypeObject",
            }

            elements = []
            for e in storey_elements(node):
                cls = G.nodes[e].get("class_")
                if cls in container_classes:
                    continue
                elements.append(
                    {
                        "id": e,
                        "label": G.nodes[e].get("label"),
                        "class_": cls,
                    }
                )
            return _ok_action(action, {"storey": storey, "elements": elements})

        if action == "find_elements_by_class":
            cls = params.get("class")
            if not cls:
                return _err("Missing param: class", "missing_param")
            if not isinstance(cls, str):
                return _err("Invalid param: class must be a string", "invalid")
            target = normalize_class(cls)

            matches = []
            for n, d in G.nodes(data=True):
                if str(d.get("class_", "")).lower() == target.lower():
                    matches.append(
                        build_node_payload(n, d, payload_mode=resolved_payload_mode)
                    )
            return _ok_action(action, {"class": target, "elements": matches})

        if action == "get_adjacent_elements":
            element_id = params.get("element_id")
            if not element_id:
                return _err("Missing param: element_id", "missing_param")
            if not isinstance(element_id, str):
                return _err("Invalid param: element_id must be a string", "invalid")
            resolved, err = resolve_element_id(element_id)
            if err:
                error_msg = err.get("error", "Unknown error")
                if "Ambiguous" in str(error_msg):
                    return _err(
                        str(error_msg),
                        "ambiguous",
                        {"candidates": err.get("candidates", [])},
                    )
                if "Invalid element_id" in str(error_msg):
                    return _err(str(error_msg), "invalid")
                return _err(str(error_msg), "not_found")
            if resolved is None:
                return _err(f"Element not found: {element_id}", "not_found")

            neighbors = []
            for nbr, edge in spatial_neighbors(resolved):
                current_relation = edge_relation(edge)
                neighbors.append(
                    {
                        "id": nbr,
                        "label": G.nodes[nbr].get("label"),
                        "class_": G.nodes[nbr].get("class_"),
                        "relation": current_relation,
                        "distance": edge.get("distance"),
                        "source": edge_source(edge, current_relation),
                    }
                )
            return _ok_action(action, {"element_id": resolved, "adjacent": neighbors})

        if action == "get_topology_neighbors":
            element_id = params.get("element_id")
            relation = params.get("relation")
            if not element_id:
                return _err("Missing param: element_id", "missing_param")
            if not isinstance(element_id, str):
                return _err("Invalid param: element_id must be a string", "invalid")
            if not relation:
                return _err("Missing param: relation", "missing_param")
            if not isinstance(relation, str):
                return _err("Invalid param: relation must be a string", "invalid")

            relation_value = normalize_relation_name(relation)
            allowed = set(TOPOLOGY_RELATIONS)
            if relation_value not in allowed:
                return _err(
                    f"Unsupported topology relation: {relation}",
                    "invalid",
                    {"allowed_relations": sorted(allowed)},
                )

            resolved, err = resolve_element_id(element_id)
            if err:
                error_msg = err.get("error", "Unknown error")
                if "Ambiguous" in str(error_msg):
                    return _err(
                        str(error_msg),
                        "ambiguous",
                        {"candidates": err.get("candidates", [])},
                    )
                if "Invalid element_id" in str(error_msg):
                    return _err(str(error_msg), "invalid")
                return _err(str(error_msg), "not_found")
            if resolved is None:
                return _err(f"Element not found: {element_id}", "not_found")

            neighbors = []
            for nbr, edge in topology_neighbors(resolved, {relation_value}):
                current_relation = edge_relation(edge)
                neighbors.append(
                    {
                        "id": nbr,
                        "label": G.nodes[nbr].get("label"),
                        "class_": G.nodes[nbr].get("class_"),
                        "relation": current_relation,
                        "vertical_gap": edge.get("vertical_gap"),
                        "overlap_area_xy": edge.get("overlap_area_xy"),
                        "intersection_volume": edge.get("intersection_volume"),
                        "contact_area": edge.get("contact_area"),
                        "source": edge_source(edge, current_relation),
                    }
                )

            return _ok_action(
                action,
                {
                    "element_id": resolved,
                    "relation": relation_value,
                    "neighbors": neighbors,
                },
            )

        if action == "get_intersections_3d":
            element_id = params.get("element_id")
            if not element_id:
                return _err("Missing param: element_id", "missing_param")
            if not isinstance(element_id, str):
                return _err("Invalid param: element_id must be a string", "invalid")

            resolved, err = resolve_element_id(element_id)
            if err:
                error_msg = err.get("error", "Unknown error")
                if "Ambiguous" in str(error_msg):
                    return _err(
                        str(error_msg),
                        "ambiguous",
                        {"candidates": err.get("candidates", [])},
                    )
                if "Invalid element_id" in str(error_msg):
                    return _err(str(error_msg), "invalid")
                return _err(str(error_msg), "not_found")
            if resolved is None:
                return _err(f"Element not found: {element_id}", "not_found")

            neighbors = []
            for nbr, edge in topology_neighbors(resolved, {"intersects_3d"}):
                current_relation = edge_relation(edge)
                neighbors.append(
                    {
                        "id": nbr,
                        "label": G.nodes[nbr].get("label"),
                        "class_": G.nodes[nbr].get("class_"),
                        "relation": current_relation,
                        "intersection_volume": edge.get("intersection_volume"),
                        "contact_area": edge.get("contact_area"),
                        "source": edge_source(edge, current_relation),
                    }
                )
            return _ok_action(
                action, {"element_id": resolved, "intersections_3d": neighbors}
            )

        if action == "find_nodes":
            cls = params.get("class")
            if cls is not None and not isinstance(cls, str):
                return _err("Invalid param: class must be a string", "invalid")
            class_filter = normalize_class(cls) if cls else None
            property_filters = params.get("property_filters", {})
            if property_filters and not isinstance(property_filters, dict):
                return _err(
                    "Invalid param: property_filters must be an object", "invalid"
                )

            db_lookup_conn: Any | None = None
            if property_filters and runtime.context_db_path is not None:
                from rag_tag.sql_element_lookup import (
                    open_lookup_connection,  # noqa: PLC0415
                )

                db_lookup_conn = open_lookup_connection(runtime.context_db_path)
                runtime.caches["db_lookup_conn"] = db_lookup_conn

            matches = []
            try:
                for n, d in G.nodes(data=True):
                    if class_filter is not None:
                        if str(d.get("class_", "")).lower() != class_filter.lower():
                            continue
                    if not apply_property_filters(
                        n,
                        property_filters,
                        db_conn=db_lookup_conn,
                    ):
                        continue
                    matches.append(
                        build_node_payload(n, d, payload_mode=resolved_payload_mode)
                    )
            finally:
                if db_lookup_conn is not None:
                    db_lookup_conn.close()
                    if runtime.caches.get("db_lookup_conn") is db_lookup_conn:
                        runtime.caches.pop("db_lookup_conn", None)
            return _ok_action(action, {"class": class_filter, "elements": matches})

        if action == "traverse":
            start = params.get("start")
            relation_param = params.get("relation")
            if not start:
                return _err("Missing param: start", "missing_param")
            if not isinstance(start, str):
                return _err("Invalid param: start must be a string", "invalid")
            if relation_param is not None and not isinstance(relation_param, str):
                return _err("Invalid param: relation must be a string", "invalid")
            try:
                depth = int(params.get("depth", 1))
            except (TypeError, ValueError):
                return _err("Invalid param: depth must be an integer", "invalid")
            if start not in G:
                return _err(f"Start node not found: {start}", "not_found")
            if depth < 1:
                return _err("Depth must be >= 1", "invalid")

            visited = {start}
            frontier = {start}
            results = []
            relation_filter: set[str] | None = None
            relation_value: str | None = None
            if relation_param is not None:
                relation_value = normalize_relation_name(relation_param)
                if relation_value is None:
                    return _err("Invalid param: relation must be non-empty", "invalid")
                if relation_value not in CANONICAL_RELATION_SET:
                    return _err(
                        f"Unsupported traverse relation: {relation_param}",
                        "invalid",
                        {"allowed_relations": sorted(CANONICAL_RELATION_SET)},
                    )
                relation_filter = {relation_value}

            for _ in range(depth):
                next_frontier = set()
                for node in frontier:
                    for nbr in G.successors(node):
                        matched_edges = []
                        for edge in iter_edge_dicts(node, nbr):
                            current_relation = edge_relation(edge)
                            if current_relation is None:
                                continue
                            if (
                                relation_filter
                                and current_relation not in relation_filter
                            ):
                                continue
                            matched_edges.append((edge, current_relation))
                        if not matched_edges:
                            continue
                        if nbr in visited:
                            continue
                        visited.add(nbr)
                        next_frontier.add(nbr)
                        for edge, current_relation in matched_edges:
                            results.append(
                                {
                                    "from": node,
                                    "to": nbr,
                                    "relation": current_relation,
                                    "source": edge_source(edge, current_relation),
                                    "node": build_node_payload(
                                        nbr,
                                        G.nodes[nbr],
                                        payload_mode=resolved_payload_mode,
                                    ),
                                }
                            )
                frontier = next_frontier

            return _ok_action(
                action,
                {
                    "start": start,
                    "relation": relation_value,
                    "depth": depth,
                    "results": results,
                },
            )

        if action == "spatial_query":
            cls = params.get("class")
            if cls is not None and not isinstance(cls, str):
                return _err("Invalid param: class must be a string", "invalid")
            class_filter = normalize_class(cls) if cls else None
            near = params.get("near")
            max_distance = params.get("max_distance")
            if near is None:
                return _err("Missing param: near", "missing_param")
            if not isinstance(near, (str, int, float)):
                return _err("Invalid param: near must be a string or number", "invalid")
            resolved, err = resolve_element_id(str(near))
            if err:
                error_msg = err.get("error", "Unknown error")
                if "Ambiguous" in str(error_msg):
                    return _err(
                        str(error_msg),
                        "ambiguous",
                        {"candidates": err.get("candidates", [])},
                    )
                if "Invalid element_id" in str(error_msg):
                    return _err(str(error_msg), "invalid")
                return _err(str(error_msg), "not_found")
            if resolved is None:
                return _err(f"Element not found: {near}", "not_found")
            if max_distance is None:
                return _err("Missing param: max_distance", "missing_param")
            try:
                max_distance_value = float(max_distance)
            except (TypeError, ValueError):
                return _err("Invalid param: max_distance must be a number", "invalid")

            results = []
            for nbr, edge in spatial_neighbors(resolved):
                dist = edge.get("distance")
                if dist is None or float(dist) > max_distance_value:
                    continue
                if class_filter is not None:
                    if (
                        str(G.nodes[nbr].get("class_", "")).lower()
                        != class_filter.lower()
                    ):
                        continue
                current_relation = edge_relation(edge)
                results.append(
                    {
                        "id": nbr,
                        "label": G.nodes[nbr].get("label"),
                        "class_": G.nodes[nbr].get("class_"),
                        "relation": current_relation,
                        "distance": dist,
                        "source": edge_source(edge, current_relation),
                    }
                )
            return _ok_action(
                action,
                {
                    "near": resolved,
                    "max_distance": max_distance_value,
                    "results": results,
                },
            )

        if action == "find_elements_above":
            element_id = params.get("element_id")
            max_gap = params.get("max_gap")
            if not element_id:
                return _err("Missing param: element_id", "missing_param")
            if not isinstance(element_id, str):
                return _err("Invalid param: element_id must be a string", "invalid")
            max_gap_value: float | None = None
            if max_gap is not None:
                try:
                    max_gap_value = float(max_gap)
                except (TypeError, ValueError):
                    return _err("Invalid param: max_gap must be a number", "invalid")

            resolved, err = resolve_element_id(element_id)
            if err:
                error_msg = err.get("error", "Unknown error")
                if "Ambiguous" in str(error_msg):
                    return _err(
                        str(error_msg),
                        "ambiguous",
                        {"candidates": err.get("candidates", [])},
                    )
                if "Invalid element_id" in str(error_msg):
                    return _err(str(error_msg), "invalid")
                return _err(str(error_msg), "not_found")
            if resolved is None:
                return _err(f"Element not found: {element_id}", "not_found")

            results = []
            for nbr, edge in topology_neighbors(resolved, {"above"}):
                gap = edge.get("vertical_gap")
                if (
                    max_gap_value is not None
                    and gap is not None
                    and float(gap) > max_gap_value
                ):
                    continue
                current_relation = edge_relation(edge)
                results.append(
                    {
                        "id": nbr,
                        "label": G.nodes[nbr].get("label"),
                        "class_": G.nodes[nbr].get("class_"),
                        "relation": current_relation,
                        "vertical_gap": gap,
                        "source": edge_source(edge, current_relation),
                    }
                )
            return _ok_action(
                action,
                {
                    "element_id": resolved,
                    "max_gap": max_gap_value,
                    "results": results,
                },
            )

        if action == "find_elements_below":
            element_id = params.get("element_id")
            max_gap = params.get("max_gap")
            if not element_id:
                return _err("Missing param: element_id", "missing_param")
            if not isinstance(element_id, str):
                return _err("Invalid param: element_id must be a string", "invalid")
            max_gap_value: float | None = None
            if max_gap is not None:
                try:
                    max_gap_value = float(max_gap)
                except (TypeError, ValueError):
                    return _err("Invalid param: max_gap must be a number", "invalid")

            resolved, err = resolve_element_id(element_id)
            if err:
                error_msg = err.get("error", "Unknown error")
                if "Ambiguous" in str(error_msg):
                    return _err(
                        str(error_msg),
                        "ambiguous",
                        {"candidates": err.get("candidates", [])},
                    )
                if "Invalid element_id" in str(error_msg):
                    return _err(str(error_msg), "invalid")
                return _err(str(error_msg), "not_found")
            if resolved is None:
                return _err(f"Element not found: {element_id}", "not_found")

            results = []
            for nbr, edge in topology_neighbors(resolved, {"below"}):
                gap = edge.get("vertical_gap")
                if (
                    max_gap_value is not None
                    and gap is not None
                    and float(gap) > max_gap_value
                ):
                    continue
                current_relation = edge_relation(edge)
                results.append(
                    {
                        "id": nbr,
                        "label": G.nodes[nbr].get("label"),
                        "class_": G.nodes[nbr].get("class_"),
                        "relation": current_relation,
                        "vertical_gap": gap,
                        "source": edge_source(edge, current_relation),
                    }
                )
            return _ok_action(
                action,
                {
                    "element_id": resolved,
                    "max_gap": max_gap_value,
                    "results": results,
                },
            )

        if action == "list_property_keys":
            cls = params.get("class")
            sample_values = params.get("sample_values", False)

            if cls is not None and not isinstance(cls, str):
                return _err("Invalid param: class must be a string", "invalid")
            if not isinstance(sample_values, bool):
                return _err("Invalid param: sample_values must be a boolean", "invalid")

            class_filter = normalize_class(cls) if cls else None
            key_samples: dict[str, list[Any]] = {}

            def record_key(key: str, value: Any) -> None:
                if key not in key_samples:
                    key_samples[key] = []
                if sample_values and len(key_samples[key]) < 3:
                    key_samples[key].append(value)

            def collect_pset_leaf_keys(
                pset_name: str,
                node: dict[str, Any],
                path_prefix: str = "",
            ) -> None:
                for raw_key, raw_value in node.items():
                    key_part = str(raw_key)
                    path = f"{path_prefix}.{key_part}" if path_prefix else key_part
                    if isinstance(raw_value, dict):
                        collect_pset_leaf_keys(pset_name, raw_value, path)
                    else:
                        record_key(f"{pset_name}.{path}", raw_value)

            for _, data in G.nodes(data=True):
                if class_filter is not None:
                    if str(data.get("class_", "")).lower() != class_filter.lower():
                        continue

                props = data.get("properties") or {}
                if not isinstance(props, dict):
                    props = {}
                for key, value in props.items():
                    record_key(str(key), value)

                payload = data.get("payload") or {}
                if not isinstance(payload, dict):
                    continue

                pset_block = payload.get("PropertySets") or {}
                if isinstance(pset_block, dict):
                    for section in ("Official", "Custom"):
                        section_block = pset_block.get(section) or {}
                        if not isinstance(section_block, dict):
                            continue
                        for pset_name, pset_props in section_block.items():
                            if not isinstance(pset_props, dict):
                                continue
                            collect_pset_leaf_keys(str(pset_name), pset_props)

                quantities_block = payload.get("Quantities") or {}
                if not isinstance(quantities_block, dict):
                    continue
                for qto_name, qto_data in quantities_block.items():
                    if not isinstance(qto_data, dict):
                        continue
                    collect_pset_leaf_keys(str(qto_name), qto_data)

            if (
                runtime.payload_mode == "minimal"
                and runtime.context_db_path is not None
            ):
                db_path = runtime.context_db_path
                cache = get_property_key_cache(runtime)
                cache_key = (str(db_path.resolve()), class_filter or "")
                db_key_samples = cache.get(cache_key)
                if db_key_samples is None:
                    db_key_samples = collect_dotted_keys_from_sqlite(
                        db_path, class_filter
                    )
                    cache[cache_key] = db_key_samples

                for key, samples in db_key_samples.items():
                    if sample_values:
                        existing = key_samples.setdefault(key, [])
                        for sample in samples:
                            if len(existing) >= 3:
                                break
                            existing.append(sample)
                    else:
                        record_key(key, None)

            data: dict[str, Any] = {
                "keys": sorted(key_samples.keys()),
                "class_filter": class_filter,
                "class_filter_raw": cls,
            }
            if sample_values:
                data["samples"] = key_samples
            return _ok_action(action, data)

        if action == "get_element_properties":
            element_id = params.get("element_id")
            if element_id is None or element_id == "":
                return _err("Missing param: element_id", "missing_param")
            if not isinstance(element_id, str):
                return _err("Invalid param: element_id must be a string", "invalid")

            resolved, err = resolve_element_id(element_id)
            if err:
                error_msg = err.get("error", "Unknown error")
                if "Ambiguous" in str(error_msg):
                    return _err(
                        str(error_msg),
                        "ambiguous",
                        {"candidates": err.get("candidates", [])},
                    )
                if "Invalid element_id" in str(error_msg):
                    return _err(str(error_msg), "invalid")
                return _err(str(error_msg), "not_found")
            if not resolved:
                return _err(f"Element not found: {element_id}", "not_found")

            base_node_data: dict[str, Any] = dict(G.nodes[resolved])

            db_path = runtime.context_db_path
            if db_path is not None:
                db_data = cached_db_lookup(runtime, resolved, db_path)
                if db_data is not None:
                    base_node_data = merge_db_element_data(base_node_data, db_data)
                else:
                    _logger.debug(
                        "get_element_properties: DB lookup found no row for %s"
                        " (db=%s) — using in-memory payload",
                        resolved,
                        db_path,
                    )

            return _ok_action(
                action,
                build_node_payload(
                    resolved,
                    base_node_data,
                    payload_mode=INTERNAL_PAYLOAD_MODE,
                ),
            )

        return _err(
            f"Unknown action: {action}",
            "unknown_action",
            {"allowed_actions": sorted(CANONICAL_ACTIONS)},
        )
