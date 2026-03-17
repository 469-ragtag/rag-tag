from __future__ import annotations

import json
import logging
import re
import sqlite3
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
    ROADMAP_ACTION_SET,
    SPATIAL_RELATIONS,
    TOPOLOGY_RELATIONS,
    build_evidence_item,
    collect_evidence,
    is_allowed_action,
    make_error_envelope,
    make_ok_envelope,
    merge_evidence_items,
    normalize_action_name,
    normalize_relation_name,
    normalize_relation_source,
    relation_bucket,
)

_logger = logging.getLogger(__name__)

_NETWORK_TRACE_DEFAULT_RELATIONS = {
    "ifc_connected_to",
    "path_connected_to",
    "connected_to",
}
_NETWORK_TRACE_ALLOWED_RELATIONS = {
    *_NETWORK_TRACE_DEFAULT_RELATIONS,
    "belongs_to_system",
    "hosts",
    "hosted_by",
}
_SERVING_SPACE_NETWORK_RELATIONS = {
    "ifc_connected_to",
    "path_connected_to",
    "belongs_to_system",
    "connected_to",
    "hosts",
    "hosted_by",
}
_TERMINAL_CLASS_NAMES = {
    "IfcFlowTerminal",
    "IfcDistributionPort",
    "IfcOutlet",
    "IfcSanitaryTerminal",
    "IfcWasteTerminal",
    "IfcStackTerminal",
    "IfcAirTerminalBox",
}
_EQUIPMENT_CLASS_NAMES = {
    "IfcFlowController",
    "IfcFlowMovingDevice",
    "IfcEnergyConversionDevice",
    "IfcUnitaryEquipment",
    "IfcElectricAppliance",
    "IfcBoiler",
    "IfcBurner",
    "IfcChiller",
    "IfcCoil",
    "IfcCondenser",
    "IfcCooledBeam",
    "IfcCoolingTower",
    "IfcDamper",
    "IfcDuctFitting",
    "IfcDuctSilencer",
    "IfcElectricDistributionBoard",
    "IfcEvaporativeCooler",
    "IfcFan",
    "IfcFilter",
    "IfcFireSuppressionTerminal",
    "IfcHeatExchanger",
    "IfcHumidifier",
    "IfcInterceptor",
    "IfcLamp",
    "IfcMedicalDevice",
    "IfcMotorConnection",
    "IfcPump",
    "IfcProtectiveDevice",
    "IfcSensor",
    "IfcSolarDevice",
    "IfcSpaceHeater",
    "IfcSwitchingDevice",
    "IfcTransformer",
    "IfcTubeBundle",
    "IfcValve",
}


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

        def node_global_id(node_id: str) -> Any | None:
            node_data = G.nodes[node_id]
            properties = node_data.get("properties") or {}
            if isinstance(properties, dict) and properties.get("GlobalId") not in {
                None,
                "",
            }:
                return properties.get("GlobalId")

            payload = node_data.get("payload") or {}
            if isinstance(payload, dict):
                return payload.get("GlobalId")
            return None

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
                    results: list[dict[str, Any]] = []
                    for attrs in edge_data.values():
                        if not isinstance(attrs, dict):
                            continue
                        normalized_attrs = {
                            str(key): value for key, value in attrs.items()
                        }
                        results.append(normalized_attrs)
                    return tuple(results)
                return ()
            if isinstance(edge_data, dict):
                return ({str(key): value for key, value in edge_data.items()},)
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

        def compact_node(node_id: str) -> dict[str, Any]:
            return {
                "id": node_id,
                "global_id": node_global_id(node_id),
                "label": G.nodes[node_id].get("label"),
                "class_": G.nodes[node_id].get("class_"),
            }

        def resolve_graph_node(
            node_ref: str,
        ) -> tuple[str | None, dict[str, Any] | None]:
            if not isinstance(node_ref, str):
                return None, {"error": "Invalid node_ref: node_ref must be a string"}

            query = node_ref.strip()
            if not query:
                return None, {"error": "Invalid node_ref: node_ref must be non-empty"}

            if query in G:
                return query, None

            resolved_element, element_err = resolve_element_id(query)
            if resolved_element is not None:
                return resolved_element, None
            if element_err and "Ambiguous" in str(element_err.get("error", "")):
                return None, {
                    "error": "Ambiguous node_id",
                    "candidates": element_err.get("candidates", []),
                }

            global_id_matches = [
                node_id for node_id in G.nodes if node_global_id(node_id) == query
            ]
            if len(global_id_matches) == 1:
                return global_id_matches[0], None
            if len(global_id_matches) > 1:
                return None, {
                    "error": "Ambiguous node_id",
                    "candidates": sorted(global_id_matches),
                }

            return None, {"error": f"Node not found: {node_ref}"}

        def resolve_space_node(
            space_query: str,
        ) -> tuple[str | None, dict[str, Any] | None]:
            resolved, err = resolve_graph_node(space_query)
            if resolved is not None:
                if str(G.nodes[resolved].get("class_", "")).lower() == "ifcspace":
                    return resolved, None
                return None, {"error": f"Node is not an IfcSpace: {space_query}"}
            if err and "Ambiguous" in str(err.get("error", "")):
                return None, err

            exact = find_nodes_by_label(space_query, class_filter="IfcSpace")
            if len(exact) == 1:
                return exact[0], None
            if len(exact) > 1:
                return None, {"error": "Ambiguous space", "candidates": exact}

            query_norm = normalize_text(space_query)
            norm_matches: list[str] = []
            for node_id, node_data in G.nodes(data=True):
                if str(node_data.get("class_", "")).lower() != "ifcspace":
                    continue
                if normalize_text(str(node_data.get("label", ""))) == query_norm:
                    norm_matches.append(node_id)

            if len(norm_matches) == 1:
                return norm_matches[0], None
            if len(norm_matches) > 1:
                return None, {"error": "Ambiguous space", "candidates": norm_matches}

            return None, {"error": f"Space not found: {space_query}"}

        def open_context_db_connection(
            *,
            action_name: str,
        ) -> tuple[sqlite3.Connection | None, dict[str, Any] | None]:
            db_path = runtime.context_db_path
            if db_path is None:
                return None, {
                    "error": (
                        "No SQLite context DB is wired into the graph runtime. "
                        "Provide or build the SQLite DB context before calling "
                        f"{action_name}."
                    ),
                    "code": "missing_context_db",
                }

            from rag_tag.sql_element_lookup import (  # noqa: PLC0415
                open_lookup_connection,
            )

            conn = open_lookup_connection(db_path)
            if conn is None:
                return None, {
                    "error": f"SQLite context DB is unavailable: {db_path}",
                    "code": "db_unavailable",
                }
            return conn, None

        def resolve_requested_element_refs(
            raw_element_ids: Any,
        ) -> tuple[list[dict[str, Any]], list[str], dict[str, Any] | None]:
            if not isinstance(raw_element_ids, list) or not all(
                isinstance(item, str) for item in raw_element_ids
            ):
                return (
                    [],
                    [],
                    {
                        "error": (
                            "Invalid param: element_ids must be an array of strings"
                        ),
                        "code": "invalid",
                    },
                )
            if not raw_element_ids:
                return (
                    [],
                    [],
                    {
                        "error": "element_ids must contain at least one element ID",
                        "code": "invalid",
                    },
                )

            seen_node_ids: set[str] = set()
            resolved_refs: list[dict[str, Any]] = []
            unresolved_ids: list[str] = []

            for requested_id in raw_element_ids:
                resolved_id, err = resolve_element_id(requested_id)
                if resolved_id is None or err is not None:
                    unresolved_ids.append(requested_id)
                    continue
                if resolved_id in seen_node_ids:
                    continue

                seen_node_ids.add(resolved_id)
                node_data = G.nodes[resolved_id]
                node_props = node_data.get("properties") or {}
                express_id_raw = node_props.get("ExpressId")
                express_id: int | None = None
                if express_id_raw not in {None, ""}:
                    try:
                        express_id_text = str(express_id_raw)
                        express_id = int(express_id_text)
                    except (TypeError, ValueError):
                        express_id = None

                resolved_refs.append(
                    {
                        "requested_id": requested_id,
                        "node_id": resolved_id,
                        "global_id": node_global_id(resolved_id),
                        "express_id": express_id,
                        "label": node_data.get("label"),
                        "class_": node_data.get("class_"),
                    }
                )

            return resolved_refs, unresolved_ids, None

        def resolve_db_element_refs(
            db_conn: sqlite3.Connection,
            element_refs: list[dict[str, Any]],
        ) -> tuple[list[dict[str, Any]], list[str]]:
            if not element_refs:
                return [], []

            from rag_tag.sql_element_lookup import (  # noqa: PLC0415
                lookup_elements_by_identifiers,
            )

            global_ids = [
                str(ref["global_id"])
                for ref in element_refs
                if ref.get("global_id") not in {None, ""}
            ]
            express_ids = [
                int(ref["express_id"])
                for ref in element_refs
                if ref.get("express_id") is not None
            ]
            rows = lookup_elements_by_identifiers(
                db_conn,
                global_ids=global_ids,
                express_ids=express_ids,
            )

            rows_by_global_id = {
                str(row["global_id"]): row
                for row in rows
                if row["global_id"] not in {None, ""}
            }
            rows_by_express_id = {int(row["express_id"]): row for row in rows}

            matched_refs: list[dict[str, Any]] = []
            unmatched_ids: list[str] = []
            for ref in element_refs:
                row = None
                global_id = ref.get("global_id")
                if global_id not in {None, ""}:
                    row = rows_by_global_id.get(str(global_id))
                if row is None and ref.get("express_id") is not None:
                    row = rows_by_express_id.get(int(ref["express_id"]))
                if row is None:
                    unmatched_ids.append(str(ref["requested_id"]))
                    continue

                matched_ref = dict(ref)
                matched_ref["db_express_id"] = int(row["express_id"])
                matched_ref["db_global_id"] = row["global_id"]
                matched_ref["db_label"] = row["name"]
                matched_ref["db_class_"] = row["ifc_class"]
                matched_refs.append(matched_ref)

            return matched_refs, unmatched_ids

        def make_bridge_sample_item(
            ref: dict[str, Any],
            *,
            field_value: Any | None = None,
            include_field_value: bool = False,
        ) -> dict[str, Any]:
            sample = compact_node(str(ref["node_id"]))
            if include_field_value:
                sample["field_value"] = field_value
            return sample

        def merge_requested_ids(*groups: list[str]) -> list[str]:
            merged: list[str] = []
            seen: set[str] = set()
            for group in groups:
                for value in group:
                    if value in seen:
                        continue
                    seen.add(value)
                    merged.append(value)
            return merged

        def is_numeric_scalar(value: Any) -> bool:
            return isinstance(value, (int, float)) and not isinstance(value, bool)

        def normalize_group_value(value: Any) -> Any:
            if isinstance(value, (dict, list)):
                return json.dumps(value, sort_keys=True, separators=(",", ":"))
            return value

        def group_sort_key(group: dict[str, Any]) -> tuple[int, str]:
            return (-int(group["count"]), str(group.get("value")))

        def parse_relation_filters(
            *,
            relation_value: Any,
            relations_value: Any,
            default_relations: set[str],
            allowed_relations: set[str] | None = None,
        ) -> tuple[set[str], str | None, list[str], dict[str, Any] | None]:
            normalized_relations: set[str] = set()

            if relation_value is not None:
                if not isinstance(relation_value, str):
                    return (
                        set(),
                        None,
                        [],
                        {"error": "Invalid param: relation must be a string"},
                    )
                normalized_relation = normalize_relation_name(relation_value)
                if normalized_relation is None:
                    return (
                        set(),
                        None,
                        [],
                        {"error": "Invalid param: relation must be non-empty"},
                    )
                normalized_relations.add(normalized_relation)

            if relations_value is not None:
                if not isinstance(relations_value, list) or not all(
                    isinstance(item, str) for item in relations_value
                ):
                    return (
                        set(),
                        None,
                        [],
                        {
                            "error": (
                                "Invalid param: relations must be an array of strings"
                            )
                        },
                    )
                for raw_relation in relations_value:
                    normalized_relation = normalize_relation_name(raw_relation)
                    if normalized_relation is None:
                        return (
                            set(),
                            None,
                            [],
                            {
                                "error": (
                                    "Invalid param: relations must contain "
                                    "non-empty strings"
                                )
                            },
                        )
                    normalized_relations.add(normalized_relation)

            if not normalized_relations:
                normalized_relations = set(default_relations)

            allowed = allowed_relations or set(CANONICAL_RELATION_SET)
            unsupported = sorted(normalized_relations - allowed)
            if unsupported:
                return (
                    set(),
                    None,
                    [],
                    {
                        "error": "Unsupported relation filter",
                        "unsupported": unsupported,
                        "allowed_relations": sorted(allowed),
                    },
                )

            ordered_relations = sorted(normalized_relations)
            single_relation = (
                ordered_relations[0] if len(ordered_relations) == 1 else None
            )
            return normalized_relations, single_relation, ordered_relations, None

        def iter_relation_neighbors(
            node_id: str,
            *,
            allowed_relations: set[str] | None = None,
        ) -> Iterable[dict[str, Any]]:
            seen: set[tuple[str, str, str, str, str]] = set()

            for nbr in G.successors(node_id):
                for edge in iter_edge_dicts(node_id, nbr):
                    relation = edge_relation(edge)
                    if relation is None:
                        continue
                    if (
                        allowed_relations is not None
                        and relation not in allowed_relations
                    ):
                        continue
                    source = edge_source(edge, relation)
                    key = (
                        nbr,
                        relation,
                        "out",
                        source or "",
                        str(
                            edge.get("distance")
                            or edge.get("vertical_gap")
                            or edge.get("intersection_volume")
                            or ""
                        ),
                    )
                    if key in seen:
                        continue
                    seen.add(key)
                    yield {
                        "neighbor": nbr,
                        "relation": relation,
                        "source": source,
                        "from": node_id,
                        "to": nbr,
                    }

            for nbr in G.predecessors(node_id):
                for edge in iter_edge_dicts(nbr, node_id):
                    relation = edge_relation(edge)
                    if relation is None:
                        continue
                    if (
                        allowed_relations is not None
                        and relation not in allowed_relations
                    ):
                        continue
                    source = edge_source(edge, relation)
                    key = (
                        nbr,
                        relation,
                        "in",
                        source or "",
                        str(
                            edge.get("distance")
                            or edge.get("vertical_gap")
                            or edge.get("intersection_volume")
                            or ""
                        ),
                    )
                    if key in seen:
                        continue
                    seen.add(key)
                    yield {
                        "neighbor": nbr,
                        "relation": relation,
                        "source": source,
                        "from": nbr,
                        "to": node_id,
                    }

        def build_relation_graph(
            allowed_relations: set[str],
        ) -> nx.Graph:
            relation_graph = nx.Graph()
            relation_graph.add_nodes_from(G.nodes)
            for source_id, target_id, edge in G.edges(data=True):
                relation = edge_relation(edge)
                if relation is None or relation not in allowed_relations:
                    continue
                relation_graph.add_edge(source_id, target_id)
            return relation_graph

        def relation_priority(relation: str | None) -> tuple[int, str]:
            bucket = relation_bucket(relation)
            bucket_ranks = {
                "explicit_ifc": 0,
                "topology": 1,
                "spatial": 2,
                "hierarchy": 3,
            }
            rank = bucket_ranks[bucket] if bucket in bucket_ranks else 4
            return rank, relation or ""

        def preferred_step_metadata(
            source_id: str,
            target_id: str,
            *,
            allowed_relations: set[str] | None = None,
        ) -> dict[str, Any] | None:
            candidates: list[dict[str, Any]] = []

            for edge in iter_edge_dicts(source_id, target_id):
                relation = edge_relation(edge)
                if relation is None:
                    continue
                if allowed_relations is not None and relation not in allowed_relations:
                    continue
                candidates.append(
                    {
                        "relation": relation,
                        "source": edge_source(edge, relation),
                        "direction": "forward",
                    }
                )

            for edge in iter_edge_dicts(target_id, source_id):
                relation = edge_relation(edge)
                if relation is None:
                    continue
                if allowed_relations is not None and relation not in allowed_relations:
                    continue
                candidates.append(
                    {
                        "relation": relation,
                        "source": edge_source(edge, relation),
                        "direction": "reverse",
                    }
                )

            if not candidates:
                return None

            candidates.sort(
                key=lambda item: (
                    0 if item["direction"] == "forward" else 1,
                    *relation_priority(item.get("relation")),
                )
            )
            return candidates[0]

        def build_grounded_path(
            node_ids: list[str],
            *,
            allowed_relations: set[str] | None = None,
        ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
            grounded_path: list[dict[str, Any]] = []
            steps: list[dict[str, Any]] = []

            for index, node_id in enumerate(node_ids):
                node_record = compact_node(node_id)
                node_record["step"] = index
                if index > 0:
                    step_meta = preferred_step_metadata(
                        node_ids[index - 1],
                        node_id,
                        allowed_relations=allowed_relations,
                    )
                    if step_meta is not None:
                        node_record["incoming_relation"] = step_meta.get("relation")
                        node_record["incoming_source"] = step_meta.get("source")
                        steps.append(
                            {
                                "from": node_ids[index - 1],
                                "to": node_id,
                                "relation": step_meta.get("relation"),
                                "source": step_meta.get("source"),
                                "direction": step_meta.get("direction"),
                            }
                        )
                grounded_path.append(node_record)

            return grounded_path, steps

        def is_context_node(node_id: str) -> bool:
            node_data = G.nodes[node_id]
            return bool(node_data.get("node_kind") == "context") or str(
                node_id
            ).startswith(("System::", "Zone::", "Classification::"))

        def is_terminalish_class(class_name: str | None) -> bool:
            if not class_name:
                return False
            if class_name in _TERMINAL_CLASS_NAMES:
                return True
            return class_name.endswith("Terminal") or class_name.endswith("Port")

        def is_equipment_class(class_name: str | None) -> bool:
            if not class_name:
                return False
            if class_name in _EQUIPMENT_CLASS_NAMES:
                return True
            markers = (
                "Equipment",
                "Controller",
                "MovingDevice",
                "ConversionDevice",
                "Appliance",
                "Boiler",
                "Chiller",
                "Condenser",
                "CoolingTower",
                "Fan",
                "HeatExchanger",
                "Pump",
                "Sensor",
                "SpaceHeater",
                "Transformer",
                "Valve",
            )
            return any(marker in class_name for marker in markers)

        def classification_match_kind(
            label: str,
            query: str,
        ) -> tuple[int, str] | None:
            label_raw = label.strip().lower()
            query_raw = query.strip().lower()
            if not label_raw or not query_raw:
                return None
            if label_raw == query_raw:
                return 3, "exact_label"

            label_norm = normalize_text(label)
            query_norm = normalize_text(query)
            if not label_norm or not query_norm:
                return None
            if label_norm == query_norm:
                return 2, "normalized_label"
            if query_norm in label_norm or label_norm in query_norm:
                return 1, "substring_label"
            return None

        def serving_space_strength(score: int) -> str:
            if score >= 6:
                return "strong"
            if score >= 4:
                return "moderate"
            return "weak"

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
                        "global_id": node_global_id(e),
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
                        "global_id": node_global_id(nbr),
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
                        "global_id": node_global_id(nbr),
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
                        "global_id": node_global_id(nbr),
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
                        "global_id": node_global_id(nbr),
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
                        "global_id": node_global_id(nbr),
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
                        "global_id": node_global_id(nbr),
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

        if action == "aggregate_elements":
            metric = params.get("metric")
            field = params.get("field")
            if not isinstance(metric, str):
                return _err("Invalid param: metric must be a string", "invalid")
            metric_value = metric.strip().lower()
            if metric_value not in {"count", "sum", "avg", "min", "max"}:
                return _err(
                    "Unsupported metric for aggregate_elements",
                    "invalid",
                    {"allowed_metrics": ["count", "sum", "avg", "min", "max"]},
                )
            if field is not None and not isinstance(field, str):
                return _err("Invalid param: field must be a string", "invalid")
            if metric_value != "count" and not field:
                return _err(
                    f"Field is required for metric '{metric_value}'",
                    "missing_param",
                )

            element_refs, unresolved_ids, ref_err = resolve_requested_element_refs(
                params.get("element_ids")
            )
            if ref_err is not None:
                return _err(str(ref_err["error"]), str(ref_err["code"]))

            db_conn, db_err = open_context_db_connection(action_name=action)
            if db_err is not None:
                return _err(str(db_err["error"]), str(db_err["code"]))
            assert db_conn is not None

            try:
                matched_refs, db_unmatched_ids = resolve_db_element_refs(
                    db_conn,
                    element_refs,
                )
                unmatched_ids = merge_requested_ids(unresolved_ids, db_unmatched_ids)
                if not matched_refs:
                    return _err(
                        (
                            "None of the provided element IDs could be resolved in "
                            "the SQLite context DB."
                        ),
                        "not_found",
                        {"unmatched_element_ids": unmatched_ids[:10]},
                    )

                sample: list[dict[str, Any]] = []
                evidence: list[dict[str, Any]] = []
                warnings: list[str] = []
                field_source: str | None = None
                missing_value_count = 0
                aggregate_value: Any = None
                canonical_field = field if isinstance(field, str) else None

                if field:
                    from rag_tag.sql_element_lookup import (  # noqa: PLC0415
                        fetch_element_field_values,
                    )

                    try:
                        field_result = fetch_element_field_values(
                            db_conn,
                            express_ids=[
                                int(ref["db_express_id"]) for ref in matched_refs
                            ],
                            field=field,
                        )
                    except ValueError as exc:
                        return _err(str(exc), "invalid")

                    field_source = str(field_result.get("field_source") or "") or None
                    canonical_field = str(field_result.get("field") or field)
                    values_by_express_id = field_result.get("values") or {}

                    valued_refs: list[tuple[dict[str, Any], Any]] = []
                    missing_refs: list[dict[str, Any]] = []
                    for ref in matched_refs:
                        value = values_by_express_id.get(int(ref["db_express_id"]))
                        if value is None:
                            missing_refs.append(ref)
                            continue
                        valued_refs.append((ref, value))

                    missing_value_count = len(missing_refs)
                    sample = [
                        make_bridge_sample_item(
                            ref,
                            field_value=value,
                            include_field_value=True,
                        )
                        for ref, value in valued_refs[:3]
                    ]
                    evidence = collect_evidence(sample, source_tool=action)

                    if unmatched_ids:
                        warnings.append(
                            (
                                f"{len(unmatched_ids)} requested element ID(s) were "
                                "not matched in the graph or SQLite context."
                            )
                        )
                    if missing_value_count:
                        warnings.append(
                            (
                                f"{missing_value_count} matched element(s) had no "
                                f"value for {canonical_field}."
                            )
                        )

                    if metric_value == "count":
                        aggregate_value = len(valued_refs)
                    elif not valued_refs:
                        warnings.append(
                            f"No non-null values were found for {canonical_field}."
                        )
                        aggregate_value = None
                    else:
                        non_numeric_records = [
                            make_bridge_sample_item(
                                ref,
                                field_value=value,
                                include_field_value=True,
                            )
                            for ref, value in valued_refs
                            if not is_numeric_scalar(value)
                        ]
                        if non_numeric_records:
                            return _err(
                                (
                                    f"Metric '{metric_value}' requires numeric "
                                    f"values, but {canonical_field} is not numeric "
                                    "for all matched elements."
                                ),
                                "invalid",
                                {"sample": non_numeric_records[:3]},
                            )

                        numeric_values = [float(value) for _ref, value in valued_refs]
                        if metric_value == "sum":
                            aggregate_value = sum(numeric_values)
                        elif metric_value == "avg":
                            aggregate_value = sum(numeric_values) / len(numeric_values)
                        elif metric_value == "min":
                            aggregate_value = min(numeric_values)
                        elif metric_value == "max":
                            aggregate_value = max(numeric_values)
                else:
                    aggregate_value = len(matched_refs)
                    sample = [make_bridge_sample_item(ref) for ref in matched_refs[:3]]
                    evidence = collect_evidence(sample, source_tool=action)
                    if unmatched_ids:
                        warnings.append(
                            (
                                f"{len(unmatched_ids)} requested element ID(s) were "
                                "not matched in the graph or SQLite context."
                            )
                        )

                data = {
                    "metric": metric_value,
                    "field": canonical_field,
                    "field_source": field_source,
                    "aggregate_value": aggregate_value,
                    "matched_element_count": len(matched_refs),
                    "unmatched_element_count": len(unmatched_ids),
                    "missing_value_count": missing_value_count,
                    "sample": sample,
                    "evidence": evidence,
                }
                if unmatched_ids:
                    data["unmatched_element_ids"] = unmatched_ids[:10]
                if warnings:
                    data["warnings"] = warnings
                return _ok_action(action, data)
            finally:
                db_conn.close()

        if action == "group_elements_by_property":
            property_key = params.get("property_key")
            if not isinstance(property_key, str):
                return _err("Invalid param: property_key must be a string", "invalid")
            if not property_key.strip():
                return _err("property_key must be a non-empty string", "invalid")

            try:
                max_groups = int(params.get("max_groups", 20))
            except (TypeError, ValueError):
                return _err("Invalid param: max_groups must be an integer", "invalid")
            if max_groups < 1:
                return _err("max_groups must be >= 1", "invalid")
            max_groups = min(max_groups, 50)

            element_refs, unresolved_ids, ref_err = resolve_requested_element_refs(
                params.get("element_ids")
            )
            if ref_err is not None:
                return _err(str(ref_err["error"]), str(ref_err["code"]))

            db_conn, db_err = open_context_db_connection(action_name=action)
            if db_err is not None:
                return _err(str(db_err["error"]), str(db_err["code"]))
            assert db_conn is not None

            try:
                matched_refs, db_unmatched_ids = resolve_db_element_refs(
                    db_conn,
                    element_refs,
                )
                unmatched_ids = merge_requested_ids(unresolved_ids, db_unmatched_ids)
                if not matched_refs:
                    return _err(
                        (
                            "None of the provided element IDs could be resolved in "
                            "the SQLite context DB."
                        ),
                        "not_found",
                        {"unmatched_element_ids": unmatched_ids[:10]},
                    )

                from rag_tag.sql_element_lookup import (  # noqa: PLC0415
                    fetch_element_field_values,
                )

                try:
                    field_result = fetch_element_field_values(
                        db_conn,
                        express_ids=[int(ref["db_express_id"]) for ref in matched_refs],
                        field=property_key,
                    )
                except ValueError as exc:
                    return _err(str(exc), "invalid")

                canonical_field = str(field_result.get("field") or property_key)
                field_source = str(field_result.get("field_source") or "") or None
                values_by_express_id = field_result.get("values") or {}

                groups_by_value: dict[str, dict[str, Any]] = {}
                missing_value_count = 0
                for ref in matched_refs:
                    raw_value = values_by_express_id.get(int(ref["db_express_id"]))
                    if raw_value is None:
                        missing_value_count += 1
                        continue

                    value = normalize_group_value(raw_value)
                    bucket_key = json.dumps(
                        value,
                        sort_keys=True,
                        ensure_ascii=True,
                        default=str,
                    )
                    group = groups_by_value.setdefault(
                        bucket_key,
                        {
                            "value": value,
                            "members": [],
                        },
                    )
                    group["members"].append(
                        make_bridge_sample_item(
                            ref,
                            field_value=value,
                            include_field_value=True,
                        )
                    )

                groups: list[dict[str, Any]] = []
                for group_data in groups_by_value.values():
                    members = group_data["members"]
                    match_reason = f"value={group_data['value']}"
                    group_evidence = collect_evidence(
                        members[:2],
                        source_tool=action,
                        match_reason_builder=lambda _item, mr=match_reason: mr,
                    )
                    groups.append(
                        {
                            "value": group_data["value"],
                            "count": len(members),
                            "sample": members[:2],
                            "evidence": group_evidence,
                        }
                    )

                groups.sort(key=group_sort_key)
                selected_groups = groups[:max_groups]
                warnings: list[str] = []
                if unmatched_ids:
                    warnings.append(
                        (
                            f"{len(unmatched_ids)} requested element ID(s) were not "
                            "matched in the graph or SQLite context."
                        )
                    )
                if missing_value_count:
                    warnings.append(
                        (
                            f"{missing_value_count} matched element(s) had no value "
                            f"for {canonical_field}."
                        )
                    )
                if len(groups) > max_groups:
                    warnings.append(
                        f"Grouping truncated to {max_groups} group(s) to stay compact."
                    )

                evidence = merge_evidence_items(
                    *[
                        group.get("evidence")
                        for group in selected_groups
                        if isinstance(group.get("evidence"), list)
                    ]
                )
                data = {
                    "property_key": canonical_field,
                    "field_source": field_source,
                    "groups": selected_groups,
                    "matched_element_count": len(matched_refs),
                    "unmatched_element_count": len(unmatched_ids),
                    "missing_value_count": missing_value_count,
                    "total_groups": len(groups),
                    "evidence": evidence,
                }
                if unmatched_ids:
                    data["unmatched_element_ids"] = unmatched_ids[:10]
                if warnings:
                    data["warnings"] = warnings
                return _ok_action(action, data)
            finally:
                db_conn.close()

        if action == "trace_distribution_network":
            start = params.get("start") or params.get("element_id")
            if not start:
                return _err("Missing param: start", "missing_param")
            if not isinstance(start, str):
                return _err("Invalid param: start must be a string", "invalid")

            try:
                max_depth = int(params.get("max_depth", params.get("depth", 3)))
            except (TypeError, ValueError):
                return _err("Invalid param: max_depth must be an integer", "invalid")
            if max_depth < 1:
                return _err("max_depth must be >= 1", "invalid")
            max_depth = min(max_depth, 6)

            try:
                max_results = int(params.get("max_results", 25))
            except (TypeError, ValueError):
                return _err("Invalid param: max_results must be an integer", "invalid")
            if max_results < 1:
                return _err("max_results must be >= 1", "invalid")
            max_results = min(max_results, 50)

            resolved_start, err = resolve_graph_node(start)
            if err:
                error_msg = err.get("error", "Unknown error")
                if "Ambiguous" in str(error_msg):
                    return _err(
                        str(error_msg),
                        "ambiguous",
                        {"candidates": err.get("candidates", [])},
                    )
                if "Invalid" in str(error_msg):
                    return _err(str(error_msg), "invalid")
                return _err(str(error_msg), "not_found")
            if resolved_start is None:
                return _err(f"Start node not found: {start}", "not_found")

            relation_filter, single_relation, ordered_relations, relation_err = (
                parse_relation_filters(
                    relation_value=params.get("relation"),
                    relations_value=params.get("relations"),
                    default_relations=_NETWORK_TRACE_DEFAULT_RELATIONS,
                    allowed_relations=_NETWORK_TRACE_ALLOWED_RELATIONS,
                )
            )
            if relation_err is not None:
                return _err(
                    str(relation_err.get("error", "Invalid relation filter")),
                    "invalid",
                    {
                        key: value
                        for key, value in relation_err.items()
                        if key != "error"
                    }
                    or None,
                )

            relation_graph = build_relation_graph(relation_filter)
            try:
                paths = nx.single_source_shortest_path(
                    relation_graph, resolved_start, cutoff=max_depth
                )
            except nx.NodeNotFound:
                return _err(f"Start node not found: {start}", "not_found")

            ordered_paths = sorted(
                (
                    path
                    for target, path in paths.items()
                    if target != resolved_start and path
                ),
                key=lambda path: (
                    len(path) - 1,
                    str(G.nodes[path[-1]].get("label") or ""),
                    path[-1],
                ),
            )

            warnings: list[str] = []
            if len(ordered_paths) > max_results:
                warnings.append(
                    f"Trace truncated to {max_results} result(s) to stay bounded."
                )

            results: list[dict[str, Any]] = []
            for path in ordered_paths[:max_results]:
                grounded_path, steps = build_grounded_path(
                    path,
                    allowed_relations=relation_filter,
                )
                step_meta = steps[-1] if steps else None
                result_item = compact_node(path[-1])
                result_item.update(
                    {
                        "depth": len(path) - 1,
                        "via_relation": (
                            step_meta.get("relation") if step_meta is not None else None
                        ),
                        "via_source": (
                            step_meta.get("source") if step_meta is not None else None
                        ),
                        "path": grounded_path,
                        "steps": steps,
                    }
                )
                results.append(result_item)

            start_evidence = build_evidence_item(
                compact_node(resolved_start),
                source_tool=action,
                match_reason="start_node",
            )
            evidence = merge_evidence_items(
                [start_evidence] if start_evidence is not None else None,
                collect_evidence(
                    results,
                    source_tool=action,
                    match_reason_builder=lambda item: f"depth={item['depth']}",
                ),
            )

            data: dict[str, Any] = {
                "start": resolved_start,
                "relation": single_relation,
                "max_depth": max_depth,
                "results": results,
                "visited_count": max(len(paths) - 1, 0),
                "evidence": evidence,
            }
            if len(ordered_relations) > 1:
                data["relations"] = ordered_relations
            if warnings:
                data["warnings"] = warnings
            return _ok_action(action, data)

        if action == "find_shortest_path":
            start = params.get("start")
            end = params.get("end")
            if not start:
                return _err("Missing param: start", "missing_param")
            if not end:
                return _err("Missing param: end", "missing_param")
            if not isinstance(start, str):
                return _err("Invalid param: start must be a string", "invalid")
            if not isinstance(end, str):
                return _err("Invalid param: end must be a string", "invalid")

            try:
                max_path_length = int(params.get("max_path_length", 8))
            except (TypeError, ValueError):
                return _err(
                    "Invalid param: max_path_length must be an integer", "invalid"
                )
            if max_path_length < 1:
                return _err("max_path_length must be >= 1", "invalid")
            max_path_length = min(max_path_length, 20)

            resolved_start, start_err = resolve_graph_node(start)
            if start_err:
                error_msg = start_err.get("error", "Unknown error")
                if "Ambiguous" in str(error_msg):
                    return _err(
                        str(error_msg),
                        "ambiguous",
                        {"candidates": start_err.get("candidates", [])},
                    )
                if "Invalid" in str(error_msg):
                    return _err(str(error_msg), "invalid")
                return _err(str(error_msg), "not_found")

            resolved_end, end_err = resolve_graph_node(end)
            if end_err:
                error_msg = end_err.get("error", "Unknown error")
                if "Ambiguous" in str(error_msg):
                    return _err(
                        str(error_msg),
                        "ambiguous",
                        {"candidates": end_err.get("candidates", [])},
                    )
                if "Invalid" in str(error_msg):
                    return _err(str(error_msg), "invalid")
                return _err(str(error_msg), "not_found")

            if resolved_start is None or resolved_end is None:
                return _err("Start or end node not found", "not_found")

            relation_filter, single_relation, ordered_relations, relation_err = (
                parse_relation_filters(
                    relation_value=params.get("relation"),
                    relations_value=params.get("relations"),
                    default_relations=set(CANONICAL_RELATION_SET),
                )
            )
            if relation_err is not None:
                return _err(
                    str(relation_err.get("error", "Invalid relation filter")),
                    "invalid",
                    {
                        key: value
                        for key, value in relation_err.items()
                        if key != "error"
                    }
                    or None,
                )

            relation_graph = build_relation_graph(relation_filter)
            try:
                path = nx.shortest_path(relation_graph, resolved_start, resolved_end)
            except nx.NodeNotFound:
                return _err("Start or end node not found", "not_found")
            except nx.NetworkXNoPath:
                return _err(
                    f"No path found between {resolved_start} and {resolved_end}",
                    "no_path",
                )

            if len(path) - 1 > max_path_length:
                return _err(
                    (
                        f"Shortest path length {len(path) - 1} exceeds the configured "
                        f"limit of {max_path_length}"
                    ),
                    "path_too_long",
                    {"max_path_length": max_path_length},
                )

            grounded_path, steps = build_grounded_path(
                path,
                allowed_relations=relation_filter,
            )
            start_evidence = build_evidence_item(
                compact_node(resolved_start),
                source_tool=action,
                match_reason="start_node",
            )
            end_evidence = build_evidence_item(
                compact_node(resolved_end),
                source_tool=action,
                match_reason="end_node",
            )
            evidence = merge_evidence_items(
                [start_evidence] if start_evidence is not None else None,
                [end_evidence] if end_evidence is not None else None,
                collect_evidence(
                    grounded_path,
                    source_tool=action,
                    match_reason_builder=lambda item: f"step={item['step']}",
                ),
            )

            data = {
                "start": resolved_start,
                "end": resolved_end,
                "relation": single_relation,
                "path": grounded_path,
                "steps": steps,
                "path_length": len(path) - 1,
                "evidence": evidence,
            }
            if len(ordered_relations) > 1:
                data["relations"] = ordered_relations
            return _ok_action(action, data)

        if action == "find_by_classification":
            classification = params.get("classification")
            if not classification:
                return _err("Missing param: classification", "missing_param")
            if not isinstance(classification, str):
                return _err("Invalid param: classification must be a string", "invalid")

            try:
                max_results = int(params.get("max_results", 25))
            except (TypeError, ValueError):
                return _err("Invalid param: max_results must be an integer", "invalid")
            if max_results < 1:
                return _err("max_results must be >= 1", "invalid")
            max_results = min(max_results, 50)

            matched_classifications: list[dict[str, Any]] = []
            for node_id, node_data in G.nodes(data=True):
                if is_context_node(node_id) is False and not str(node_id).startswith(
                    "Classification::"
                ):
                    continue
                if str(node_data.get("class_", "")).lower() not in {
                    "ifcclassificationreference",
                    "ifcclassification",
                } and not str(node_id).startswith("Classification::"):
                    continue

                match = classification_match_kind(
                    str(node_data.get("label", "")),
                    classification,
                )
                if match is None:
                    continue
                score, match_reason = match
                matched = compact_node(node_id)
                matched["match_reason"] = match_reason
                matched["match_score"] = score
                matched_classifications.append(matched)

            matched_classifications.sort(
                key=lambda item: (
                    -int(item.get("match_score", 0)),
                    str(item.get("label") or ""),
                    str(item.get("id") or ""),
                )
            )

            element_index: dict[str, dict[str, Any]] = {}
            for classification_item in matched_classifications:
                class_id = str(classification_item["id"])

                for source_id in G.predecessors(class_id):
                    for edge in iter_edge_dicts(source_id, class_id):
                        if edge_relation(edge) != "classified_as":
                            continue
                        if is_context_node(source_id):
                            continue
                        candidate = element_index.setdefault(
                            source_id,
                            {
                                **compact_node(source_id),
                                "matched_classifications": [],
                                "match_reason": classification_item["match_reason"],
                                "match_score": classification_item["match_score"],
                            },
                        )
                        candidate["match_score"] = max(
                            int(candidate.get("match_score", 0)),
                            int(classification_item["match_score"]),
                        )
                        candidate["matched_classifications"].append(
                            {
                                "id": classification_item["id"],
                                "label": classification_item["label"],
                                "class_": classification_item["class_"],
                                "match_reason": classification_item["match_reason"],
                            }
                        )

            ordered_elements = sorted(
                element_index.values(),
                key=lambda item: (
                    -int(item.get("match_score", 0)),
                    str(item.get("label") or ""),
                    str(item.get("id") or ""),
                ),
            )

            warnings: list[str] = []
            if len(ordered_elements) > max_results:
                warnings.append(
                    f"Classification results truncated to {max_results} element(s)."
                )

            selected_elements = ordered_elements[:max_results]
            classification_evidence: list[dict[str, Any]] = []
            for item in matched_classifications[:2]:
                evidence_item = build_evidence_item(
                    item,
                    source_tool=action,
                    match_reason=str(
                        item.get("match_reason") or "classification_match"
                    ),
                )
                if evidence_item is not None:
                    classification_evidence.append(evidence_item)

            evidence = merge_evidence_items(
                classification_evidence,
                collect_evidence(
                    selected_elements,
                    source_tool=action,
                    match_reason_builder=lambda item: str(
                        item.get("match_reason") or "classification_match"
                    ),
                ),
            )

            data = {
                "classification": classification,
                "elements": selected_elements,
                "matched_classifications": matched_classifications,
                "total": len(ordered_elements),
                "evidence": evidence,
            }
            if warnings:
                data["warnings"] = warnings
            return _ok_action(action, data)

        if action == "find_equipment_serving_space":
            space = params.get("space")
            if not space:
                return _err("Missing param: space", "missing_param")
            if not isinstance(space, str):
                return _err("Invalid param: space must be a string", "invalid")

            try:
                max_depth = int(params.get("max_depth", 4))
            except (TypeError, ValueError):
                return _err("Invalid param: max_depth must be an integer", "invalid")
            if max_depth < 1:
                return _err("max_depth must be >= 1", "invalid")
            max_depth = min(max_depth, 6)

            try:
                max_results = int(params.get("max_results", 10))
            except (TypeError, ValueError):
                return _err("Invalid param: max_results must be an integer", "invalid")
            if max_results < 1:
                return _err("max_results must be >= 1", "invalid")
            max_results = min(max_results, 20)

            resolved_space, err = resolve_space_node(space)
            if err:
                error_msg = err.get("error", "Unknown error")
                if "Ambiguous" in str(error_msg):
                    return _err(
                        str(error_msg),
                        "ambiguous",
                        {"candidates": err.get("candidates", [])},
                    )
                if "Invalid" in str(error_msg):
                    return _err(str(error_msg), "invalid")
                return _err(str(error_msg), "not_found")
            if resolved_space is None:
                return _err(f"Space not found: {space}", "not_found")

            seed_map: dict[str, dict[str, Any]] = {}

            def add_seed(
                node_id: str,
                *,
                base_score: int,
                reason: str,
            ) -> None:
                if node_id == resolved_space or is_context_node(node_id):
                    return
                node_class = str(G.nodes[node_id].get("class_", ""))
                if not (
                    is_equipment_class(node_class)
                    or is_terminalish_class(node_class)
                    or node_class.startswith("IfcFlow")
                ):
                    return
                existing = seed_map.get(node_id)
                seed_path, _ = build_grounded_path([resolved_space, node_id])
                seed_payload = {
                    "node_id": node_id,
                    "base_score": base_score,
                    "reason": reason,
                    "space_path_ids": [resolved_space, node_id],
                    "space_path": seed_path,
                }
                if existing is None or base_score > int(existing["base_score"]):
                    seed_map[node_id] = seed_payload

            for nbr in G.successors(resolved_space):
                has_contains = any(
                    edge_relation(edge) == "contains"
                    for edge in iter_edge_dicts(resolved_space, nbr)
                )
                if not has_contains:
                    continue
                node_class = str(G.nodes[nbr].get("class_", ""))
                if is_equipment_class(node_class):
                    add_seed(nbr, base_score=6, reason="directly contained in space")
                elif is_terminalish_class(node_class):
                    add_seed(nbr, base_score=5, reason="terminal contained in space")

            for neighbor in iter_relation_neighbors(
                resolved_space,
                allowed_relations={"space_bounded_by", "bounds_space"},
            ):
                nbr = str(neighbor["neighbor"])
                node_class = str(G.nodes[nbr].get("class_", ""))
                if is_equipment_class(node_class):
                    add_seed(nbr, base_score=5, reason="space boundary equipment")
                elif is_terminalish_class(node_class):
                    add_seed(nbr, base_score=4, reason="terminal on space boundary")

            for nbr, _edge in spatial_neighbors(resolved_space):
                node_class = str(G.nodes[nbr].get("class_", ""))
                if is_equipment_class(node_class):
                    add_seed(nbr, base_score=4, reason="equipment adjacent to space")
                elif is_terminalish_class(node_class):
                    add_seed(nbr, base_score=3, reason="terminal adjacent to space")

            candidate_index: dict[str, dict[str, Any]] = {}

            def register_candidate(
                node_id: str,
                *,
                score: int,
                reason: str,
                seed_id: str,
                full_path_ids: list[str],
                candidate_type: str,
            ) -> None:
                if node_id == resolved_space or is_context_node(node_id):
                    return

                grounded_path, steps = build_grounded_path(full_path_ids)
                existing = candidate_index.get(node_id)
                if existing is None:
                    existing = {
                        **compact_node(node_id),
                        "candidate_type": candidate_type,
                        "score": score,
                        "support_reasons": [reason],
                        "seed_nodes": [compact_node(seed_id)],
                        "path": grounded_path,
                        "steps": steps,
                        "path_length": len(full_path_ids) - 1,
                    }
                    candidate_index[node_id] = existing
                    return

                existing["score"] = max(int(existing["score"]), score)
                existing["candidate_type"] = (
                    "equipment"
                    if existing["candidate_type"] == "equipment"
                    or candidate_type == "equipment"
                    else "terminal"
                )
                if reason not in existing["support_reasons"]:
                    existing["support_reasons"].append(reason)
                seed_record = compact_node(seed_id)
                if all(
                    str(current.get("id")) != seed_id
                    for current in existing["seed_nodes"]
                ):
                    existing["seed_nodes"].append(seed_record)
                current_path_length = int(existing.get("path_length", 999))
                if (
                    score > int(existing["score"])
                    or len(full_path_ids) - 1 < current_path_length
                ):
                    existing["path"] = grounded_path
                    existing["steps"] = steps
                    existing["path_length"] = len(full_path_ids) - 1

            relation_graph = build_relation_graph(_SERVING_SPACE_NETWORK_RELATIONS)
            for seed_id, seed in seed_map.items():
                seed_class = str(G.nodes[seed_id].get("class_", ""))
                if is_equipment_class(seed_class):
                    register_candidate(
                        seed_id,
                        score=int(seed["base_score"]),
                        reason=str(seed["reason"]),
                        seed_id=seed_id,
                        full_path_ids=list(seed["space_path_ids"]),
                        candidate_type="equipment",
                    )
                elif is_terminalish_class(seed_class):
                    register_candidate(
                        seed_id,
                        score=max(int(seed["base_score"]) - 2, 1),
                        reason=f"fallback terminal: {seed['reason']}",
                        seed_id=seed_id,
                        full_path_ids=list(seed["space_path_ids"]),
                        candidate_type="terminal",
                    )

                try:
                    seed_paths = nx.single_source_shortest_path(
                        relation_graph,
                        seed_id,
                        cutoff=max_depth,
                    )
                except nx.NodeNotFound:
                    continue

                for candidate_id, network_path in seed_paths.items():
                    if candidate_id in {seed_id, resolved_space}:
                        continue
                    if is_context_node(candidate_id):
                        continue
                    candidate_class = str(G.nodes[candidate_id].get("class_", ""))
                    if not (
                        is_equipment_class(candidate_class)
                        or is_terminalish_class(candidate_class)
                    ):
                        continue

                    full_path_ids = list(seed["space_path_ids"]) + network_path[1:]
                    network_hops = max(len(network_path) - 1, 0)
                    score = int(seed["base_score"]) + max(0, 2 - network_hops)
                    candidate_type = (
                        "equipment"
                        if is_equipment_class(candidate_class)
                        else "terminal"
                    )
                    if candidate_type == "equipment":
                        score += 1
                    register_candidate(
                        candidate_id,
                        score=score,
                        reason=(f"{seed['reason']} with {network_hops} network hop(s)"),
                        seed_id=seed_id,
                        full_path_ids=full_path_ids,
                        candidate_type=candidate_type,
                    )

            ordered_candidates = sorted(
                candidate_index.values(),
                key=lambda item: (
                    0 if item.get("candidate_type") == "equipment" else 1,
                    -int(item.get("score", 0)),
                    int(item.get("path_length", 999)),
                    str(item.get("label") or ""),
                    str(item.get("id") or ""),
                ),
            )

            if any(
                item.get("candidate_type") == "equipment" for item in ordered_candidates
            ):
                ordered_candidates = [
                    item
                    for item in ordered_candidates
                    if item.get("candidate_type") == "equipment"
                ]

            warnings: list[str] = []
            if not seed_map:
                warnings.append(
                    "No direct terminals or equipment were linked to the "
                    "space; results may be incomplete."
                )
            if (
                ordered_candidates
                and ordered_candidates[0].get("candidate_type") != "equipment"
            ):
                warnings.append(
                    "No upstream equipment was found; returning terminal-level "
                    "serving candidates instead."
                )
            if len(ordered_candidates) > max_results:
                warnings.append(
                    "Serving-equipment results truncated to "
                    f"{max_results} candidate(s)."
                )

            equipment = []
            for item in ordered_candidates[:max_results]:
                candidate = dict(item)
                candidate["support_strength"] = serving_space_strength(
                    int(candidate.get("score", 0))
                )
                equipment.append(candidate)

            if equipment and all(
                str(item.get("support_strength")) == "weak" for item in equipment
            ):
                warnings.append(
                    "Serving-equipment evidence is weak and based on indirect "
                    "graph links."
                )

            space_evidence = build_evidence_item(
                compact_node(resolved_space),
                source_tool=action,
                match_reason="space_anchor",
            )
            evidence = merge_evidence_items(
                [space_evidence] if space_evidence is not None else None,
                collect_evidence(
                    equipment,
                    source_tool=action,
                    match_reason_builder=lambda item: str(
                        item.get("support_strength") or "candidate"
                    ),
                ),
            )

            data = {
                "space": resolved_space,
                "equipment": equipment,
                "seed_count": len(seed_map),
                "evidence": evidence,
            }
            if warnings:
                data["warnings"] = warnings
            return _ok_action(action, data)

        if action in ROADMAP_ACTION_SET:
            return _err(
                f"Action not implemented yet: {action}",
                "not_implemented",
            )

        return _err(
            f"Unknown action: {action}",
            "unknown_action",
            {"allowed_actions": sorted(CANONICAL_ACTIONS)},
        )
