"""Import IFC graph data into Neo4j."""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Iterator

from rag_tag.graph.backends.neo4j_cypher import (
    CLEAR_DB,
    CREATE_CONSTRAINT,
    INSERT_NODES,
    INSERT_RELS,
)
from rag_tag.graph.catalog import GraphCatalog
from rag_tag.level_normalization import canonicalize_level
from rag_tag.observability import _load_dotenv
from rag_tag.parser.jsonl_to_graph import (
    _bbox_xy_overlap_area,
    _bboxes_intersect,
    _building_node_id,
    _compute_common_metrics,
    _context_node_id,
    _convex_polygon_intersection_area,
    _dataset_key_from_path,
    _element_node_id,
    _flat_properties,
    _geom_from_record,
    _get_occ_shape_index,
    _make_payload,
    _mesh_from_record,
    _normalize_context_label,
    _oriented_geom_from_record,
    _project_node_id,
    _resolve_graph_payload_mode,
    _storey_node_id,
    add_spatial_adjacency,
)

LOG = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

try:
    from neo4j import GraphDatabase
except Exception:  # pragma: no cover - optional dependency
    GraphDatabase = None


DERIVED_RELATIONS = {
    "adjacent_to",
    "connected_to",
    "intersects_bbox",
    "overlaps_xy",
    "above",
    "below",
    "intersects_3d",
    "touches_surface",
}

FULL_DATASET_SPATIAL_LIMIT = 2000

_BASEMENT_LEVEL_RE = re.compile(r"^basement(?:\s+(\d+))?$")
_STANDARD_LEVEL_RE = re.compile(r"^level\s+(\d+)$")


def _encode_json(value: Any) -> str | None:
    if value is None:
        return None
    try:
        return json.dumps(value, default=str)
    except Exception:
        return None


def _geometry_payload(node_data: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "geometry",
        "bbox",
        "mesh",
        "footprint_polygon",
        "obb",
        "local_placement_matrix",
        "z_min",
        "z_max",
        "height",
        "footprint_bbox_2d",
    )
    payload: dict[str, Any] = {}
    for key in keys:
        if key in node_data:
            payload[key] = node_data.get(key)
    return payload


def _scalar(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    return None


def _extract_property_scalars(properties: dict[str, Any]) -> dict[str, Any]:
    return {
        "global_id": _scalar(properties.get("GlobalId")),
        "express_id": _scalar(properties.get("ExpressId")),
        "name": _scalar(properties.get("Name")),
        "class_raw": _scalar(properties.get("ClassRaw")),
        "ifc_type": _scalar(properties.get("IfcType")),
        "level": _scalar(properties.get("Level")),
        "type_name": _scalar(properties.get("TypeName")),
        "predefined_type": _scalar(properties.get("PredefinedType")),
        "object_type": _scalar(properties.get("ObjectType")),
        "tag": _scalar(properties.get("Tag")),
        "description": _scalar(properties.get("Description")),
    }


def _node_row(
    node_id: str,
    *,
    label: Any,
    class_: Any,
    node_kind: Any = None,
    dataset: Any = None,
    properties: dict[str, Any] | None = None,
    payload: dict[str, Any] | None = None,
    geometry_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    props = properties or {}
    return {
        "node_id": node_id,
        "label": _scalar(label),
        "class_": _scalar(class_),
        "node_kind": _scalar(node_kind),
        "dataset": _scalar(dataset),
        "properties_json": _encode_json(props),
        "payload_json": _encode_json(payload or {}),
        "geometry_json": _encode_json(geometry_payload or {}),
        **_extract_property_scalars(props),
    }


def _context_node_row(
    node_id: str,
    *,
    label: str,
    class_: str,
    dataset: str,
) -> dict[str, Any]:
    return _node_row(
        node_id,
        label=label,
        class_=class_,
        node_kind="context",
        dataset=dataset,
        properties={},
        payload={},
        geometry_payload={},
    )


def _node_row_from_record(
    rec: dict[str, Any],
    *,
    dataset_key: str,
    node_id: str,
    payload_mode: str,
) -> dict[str, Any]:
    props = _flat_properties(rec)
    centroid, bbox = _geom_from_record(rec)
    mesh = _mesh_from_record(rec)
    footprint_poly, obb, placement = _oriented_geom_from_record(rec)
    geometry_payload = {
        "geometry": centroid,
        "bbox": bbox,
        "mesh": mesh,
        "footprint_polygon": footprint_poly,
        "obb": obb,
        "local_placement_matrix": placement,
    }
    if bbox is not None:
        geometry_payload["z_min"] = bbox[0][2]
        geometry_payload["z_max"] = bbox[1][2]
        geometry_payload["height"] = bbox[1][2] - bbox[0][2]
        if str(node_id).startswith("Element::"):
            geometry_payload["footprint_bbox_2d"] = (
                bbox[0][0],
                bbox[0][1],
                bbox[1][0],
                bbox[1][1],
            )

    ifc_type = rec.get("IfcType") or rec.get("ClassRaw") or "IfcProduct"
    name = rec.get("Name") or rec.get("GlobalId")

    return _node_row(
        node_id,
        label=name,
        class_=ifc_type,
        dataset=dataset_key,
        properties=props,
        payload=_make_payload(rec, payload_mode),
        geometry_payload=geometry_payload,
    )


def _append_rel_row(
    rel_rows: list[dict[str, Any]],
    *,
    edge_index: int,
    from_id: str,
    to_id: str,
    relation: str,
    dataset: str,
    source: str | None = None,
    distance: Any = None,
    vertical_gap: Any = None,
    overlap_area_xy: Any = None,
    intersection_volume: Any = None,
    contact_area: Any = None,
) -> int:
    rel_rows.append(
        {
            "from": from_id,
            "to": to_id,
            "relation": relation,
            "source": source,
            "dataset": dataset,
            "distance": distance,
            "vertical_gap": vertical_gap,
            "overlap_area_xy": overlap_area_xy,
            "intersection_volume": intersection_volume,
            "contact_area": contact_area,
            "edge_index": edge_index,
        }
    )
    return edge_index + 1


def _flush_node_rows(
    session: Any,
    node_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not node_rows:
        return node_rows
    session.run(INSERT_NODES, rows=node_rows)
    return []


def _flush_rel_rows(
    session: Any,
    rel_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not rel_rows:
        return rel_rows
    session.run(INSERT_RELS, rows=rel_rows)
    return []


def _emit_ifc_edge(
    rel_rows: list[dict[str, Any]],
    emitted_edges: set[tuple[str, str, str]],
    *,
    edge_index: int,
    from_id: str,
    to_id: str,
    relation: str,
    dataset: str,
) -> int:
    dedupe_key = (from_id, to_id, relation)
    if dedupe_key in emitted_edges:
        return edge_index
    emitted_edges.add(dedupe_key)
    return _append_rel_row(
        rel_rows,
        edge_index=edge_index,
        from_id=from_id,
        to_id=to_id,
        relation=relation,
        dataset=dataset,
        source="ifc",
    )


def _yield_jsonl_records(jsonl_path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    with jsonl_path.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                LOG.warning("%s line %d: JSON error: %s", jsonl_path.name, line_no, exc)
                continue
            if not isinstance(rec, dict):
                continue
            yield line_no, rec


def _build_element_geometry_catalog(
    jsonl_path: Path,
    *,
    namespaced_ids: bool,
) -> GraphCatalog:
    dataset_key = _dataset_key_from_path(jsonl_path)
    graph = GraphCatalog(graph={"datasets": [dataset_key]})

    for _, rec in _yield_jsonl_records(jsonl_path):
        gid = rec.get("GlobalId")
        if not isinstance(gid, str) or not gid:
            continue

        ifc_type = rec.get("IfcType") or rec.get("ClassRaw") or "IfcProduct"
        if ifc_type in {"IfcProject", "IfcBuilding", "IfcBuildingStorey"}:
            continue

        centroid, bbox = _geom_from_record(rec)
        if centroid is None and bbox is None:
            continue

        mesh = _mesh_from_record(rec)
        footprint_poly, _obb, _placement = _oriented_geom_from_record(rec)
        node_id = _element_node_id(
            dataset_key,
            gid,
            namespaced=namespaced_ids,
        )
        graph.add_node(
            node_id,
            label=rec.get("Name") or gid,
            class_=ifc_type,
            properties=_flat_properties(rec),
            geometry=centroid,
            bbox=bbox,
            mesh=mesh,
            footprint_polygon=footprint_poly,
            dataset=dataset_key,
        )

    return graph


def _level_sort_key(level: str | None) -> tuple[int, int] | None:
    if level is None:
        return None
    if level == "ground floor":
        return (0, 0)
    basement_match = _BASEMENT_LEVEL_RE.fullmatch(level)
    if basement_match is not None:
        number = basement_match.group(1)
        return (-1, -(int(number) if number is not None else 1))
    standard_match = _STANDARD_LEVEL_RE.fullmatch(level)
    if standard_match is not None:
        return (1, int(standard_match.group(1)))
    return None


def _spatial_partition_node_sets(graph: GraphCatalog) -> list[set[str]]:
    node_ids = list(graph.nodes)
    if len(node_ids) <= FULL_DATASET_SPATIAL_LIMIT:
        return [set(node_ids)]

    groups: dict[str, list[str]] = {}
    for node_id, node_data in graph.nodes(data=True):
        properties = node_data.get("properties")
        level_value = None
        if isinstance(properties, dict):
            level_value = canonicalize_level(str(properties.get("Level") or ""))
        group_key = level_value or "__unknown__"
        groups.setdefault(group_key, []).append(node_id)

    partitions: list[set[str]] = []
    for members in groups.values():
        if len(members) > 1:
            partitions.append(set(members))

    ranked_groups = sorted(
        (
            (sort_key, members)
            for level, members in groups.items()
            if level != "__unknown__"
            for sort_key in [_level_sort_key(level)]
            if sort_key is not None and len(members) > 0
        ),
        key=lambda item: item[0],
    )
    for (_, left_members), (_, right_members) in zip(ranked_groups, ranked_groups[1:]):
        partitions.append(set(left_members) | set(right_members))

    return partitions or [set(node_ids)]


def _iter_xy_bbox_candidate_pairs(
    element_nodes: list[str],
    element_bboxes: list[tuple],
) -> Iterator[tuple[int, int]]:
    sorted_indices = sorted(
        range(len(element_nodes)),
        key=lambda idx: element_bboxes[idx][0][0],
    )
    active: list[int] = []

    for idx in sorted_indices:
        bbox = element_bboxes[idx]
        min_x = float(bbox[0][0])
        min_y = float(bbox[0][1])
        max_y = float(bbox[1][1])
        active = [
            active_idx
            for active_idx in active
            if float(element_bboxes[active_idx][1][0]) >= min_x
        ]
        for other_idx in active:
            other_bbox = element_bboxes[other_idx]
            other_min_y = float(other_bbox[0][1])
            other_max_y = float(other_bbox[1][1])
            if other_max_y < min_y or max_y < other_min_y:
                continue
            yield other_idx, idx
        active.append(idx)


def _add_topology_facts_pruned(graph: GraphCatalog) -> None:
    def _add_topology_edge(u: str, v: str, **attrs: object) -> None:
        graph.add_edge(u, v, source="topology", **attrs)

    element_nodes: list[str] = []
    element_bboxes: list[tuple] = []
    element_refs: list[tuple[str, str, str]] = []
    element_footprints: list[list[tuple[float, float]] | None] = []

    for node_id, node_data in graph.nodes(data=True):
        if not str(node_id).startswith("Element::"):
            continue
        bbox = node_data.get("bbox")
        if bbox is None:
            continue
        properties = node_data.get("properties")
        gid = properties.get("GlobalId") if isinstance(properties, dict) else None
        dataset_key = node_data.get("dataset")
        if not isinstance(gid, str) or not gid or not isinstance(dataset_key, str):
            continue
        element_nodes.append(node_id)
        element_bboxes.append(bbox)
        element_refs.append((node_id, dataset_key, gid))
        footprint = node_data.get("footprint_polygon")
        if isinstance(footprint, list) and len(footprint) >= 3:
            parsed = []
            for point in footprint:
                if isinstance(point, tuple) and len(point) == 2:
                    parsed.append((float(point[0]), float(point[1])))
                elif isinstance(point, list) and len(point) == 2:
                    parsed.append((float(point[0]), float(point[1])))
            element_footprints.append(parsed if len(parsed) >= 3 else None)
        else:
            element_footprints.append(None)

    occ_shape_by_node_id = _get_occ_shape_index(graph, element_refs)

    for i, j in _iter_xy_bbox_candidate_pairs(element_nodes, element_bboxes):
        a = element_nodes[i]
        b = element_nodes[j]
        bbox_a = element_bboxes[i]
        bbox_b = element_bboxes[j]
        node_ref_a = element_refs[i]
        node_ref_b = element_refs[j]
        footprint_a = element_footprints[i]
        footprint_b = element_footprints[j]

        intersects_bbox = _bboxes_intersect(bbox_a, bbox_b)
        if intersects_bbox:
            _add_topology_edge(a, b, relation="intersects_bbox")
            _add_topology_edge(b, a, relation="intersects_bbox")

        if footprint_a is not None and footprint_b is not None:
            overlap_area = _convex_polygon_intersection_area(
                footprint_a,
                footprint_b,
            )
        else:
            overlap_area = _bbox_xy_overlap_area(bbox_a, bbox_b)

        if overlap_area > 0.0:
            _add_topology_edge(
                a,
                b,
                relation="overlaps_xy",
                overlap_area_xy=overlap_area,
            )
            _add_topology_edge(
                b,
                a,
                relation="overlaps_xy",
                overlap_area_xy=overlap_area,
            )

        shape_a = occ_shape_by_node_id.get(node_ref_a[0])
        shape_b = occ_shape_by_node_id.get(node_ref_b[0])
        if shape_a is not None and shape_b is not None:
            metrics = _compute_common_metrics(shape_a, shape_b)
            if metrics is not None:
                intersection_volume, contact_area = metrics
                if intersection_volume > 1e-9:
                    _add_topology_edge(
                        a,
                        b,
                        relation="intersects_3d",
                        intersection_volume=intersection_volume,
                        contact_area=contact_area,
                    )
                    _add_topology_edge(
                        b,
                        a,
                        relation="intersects_3d",
                        intersection_volume=intersection_volume,
                        contact_area=contact_area,
                    )
                elif contact_area > 1e-6:
                    _add_topology_edge(
                        a,
                        b,
                        relation="touches_surface",
                        intersection_volume=intersection_volume,
                        contact_area=contact_area,
                    )
                    _add_topology_edge(
                        b,
                        a,
                        relation="touches_surface",
                        intersection_volume=intersection_volume,
                        contact_area=contact_area,
                    )

        if overlap_area <= 0.0:
            continue
        a_min_z, a_max_z = float(bbox_a[0][2]), float(bbox_a[1][2])
        b_min_z, b_max_z = float(bbox_b[0][2]), float(bbox_b[1][2])
        if a_min_z > b_max_z:
            gap = a_min_z - b_max_z
            _add_topology_edge(a, b, relation="above", vertical_gap=gap)
            _add_topology_edge(b, a, relation="below", vertical_gap=gap)
        elif b_min_z > a_max_z:
            gap = b_min_z - a_max_z
            _add_topology_edge(b, a, relation="above", vertical_gap=gap)
            _add_topology_edge(a, b, relation="below", vertical_gap=gap)


def _append_derived_rows_from_graph(
    rel_rows: list[dict[str, Any]],
    graph: GraphCatalog,
    *,
    edge_index: int,
    emitted_edges: set[tuple[str, str, str]] | None = None,
) -> int:
    for from_id, to_id, edge_data in graph.edges(data=True):
        relation = edge_data.get("relation")
        if relation not in DERIVED_RELATIONS:
            continue
        dedupe_key = (from_id, to_id, str(relation))
        if emitted_edges is not None and dedupe_key in emitted_edges:
            continue
        dataset = graph.nodes.get(from_id, {}).get("dataset") or graph.nodes.get(
            to_id,
            {},
        ).get("dataset")
        if not isinstance(dataset, str) or not dataset:
            continue
        if emitted_edges is not None:
            emitted_edges.add(dedupe_key)
        edge_index = _append_rel_row(
            rel_rows,
            edge_index=edge_index,
            from_id=from_id,
            to_id=to_id,
            relation=relation,
            dataset=dataset,
            source=edge_data.get("source"),
            distance=edge_data.get("distance"),
            vertical_gap=edge_data.get("vertical_gap"),
            overlap_area_xy=edge_data.get("overlap_area_xy"),
            intersection_volume=edge_data.get("intersection_volume"),
            contact_area=edge_data.get("contact_area"),
        )
    return edge_index


def import_jsonl_files(
    jsonl_paths: list[Path],
    *,
    batch_size: int = 2000,
    payload_mode: str | None = None,
    database: str | None = None,
    uri: str | None = None,
    username: str | None = None,
    password: str | None = None,
) -> None:
    _load_dotenv()
    if GraphDatabase is None:
        raise RuntimeError("neo4j package not installed")

    resolved_payload_mode = _resolve_graph_payload_mode(
        payload_mode
        if payload_mode is not None
        else os.environ.get("GRAPH_PAYLOAD_MODE", "full")
    )
    uri = uri or (os.environ.get("NEO4J_URI") or "").strip()
    username = username or (os.environ.get("NEO4J_USERNAME") or "").strip()
    password = password or (os.environ.get("NEO4J_PASSWORD") or "").strip()
    database = database or (os.environ.get("NEO4J_DATABASE") or "").strip() or None
    if not uri or not username or not password:
        raise RuntimeError("NEO4J_URI/NEO4J_USERNAME/NEO4J_PASSWORD not set")

    resolved_paths = [path.expanduser().resolve() for path in jsonl_paths]
    namespaced_ids = len({_dataset_key_from_path(path) for path in resolved_paths}) > 1
    node_id_by_gid_by_dataset: dict[str, dict[str, str]] = {}

    driver = GraphDatabase.driver(uri, auth=(username, password))
    try:
        with driver.session(database=database) as session:
            session.run(CLEAR_DB)
            session.run(CREATE_CONSTRAINT)

            node_rows: list[dict[str, Any]] = []
            rel_rows: list[dict[str, Any]] = []
            edge_index = 0

            for jsonl_path in resolved_paths:
                dataset_key = _dataset_key_from_path(jsonl_path)
                node_id_by_gid = node_id_by_gid_by_dataset.setdefault(dataset_key, {})
                project_node_id = _project_node_id(
                    dataset_key,
                    namespaced=namespaced_ids,
                )
                building_node_id = _building_node_id(
                    dataset_key,
                    namespaced=namespaced_ids,
                )

                node_rows.append(
                    _node_row(
                        project_node_id,
                        label="Project",
                        class_="IfcProject",
                        dataset=dataset_key,
                        properties={},
                        payload={},
                        geometry_payload={},
                    )
                )
                node_rows.append(
                    _node_row(
                        building_node_id,
                        label="Building",
                        class_="IfcBuilding",
                        dataset=dataset_key,
                        properties={},
                        payload={},
                        geometry_payload={},
                    )
                )
                edge_index = _append_rel_row(
                    rel_rows,
                    edge_index=edge_index,
                    from_id=project_node_id,
                    to_id=building_node_id,
                    relation="aggregates",
                    dataset=dataset_key,
                )

                LOG.info("Reading %s", jsonl_path)
                for _, rec in _yield_jsonl_records(jsonl_path):
                    gid = rec.get("GlobalId")
                    if not gid:
                        continue

                    ifc_type = rec.get("IfcType") or rec.get("ClassRaw") or "IfcProduct"
                    if ifc_type == "IfcProject":
                        node_id = project_node_id
                    elif ifc_type == "IfcBuilding":
                        node_id = building_node_id
                    elif ifc_type == "IfcBuildingStorey":
                        node_id = _storey_node_id(
                            dataset_key,
                            gid,
                            namespaced=namespaced_ids,
                        )
                        edge_index = _append_rel_row(
                            rel_rows,
                            edge_index=edge_index,
                            from_id=building_node_id,
                            to_id=node_id,
                            relation="aggregates",
                            dataset=dataset_key,
                        )
                    else:
                        node_id = _element_node_id(
                            dataset_key,
                            gid,
                            namespaced=namespaced_ids,
                        )

                    node_rows.append(
                        _node_row_from_record(
                            rec,
                            dataset_key=dataset_key,
                            node_id=node_id,
                            payload_mode=resolved_payload_mode,
                        )
                    )
                    node_id_by_gid[gid] = node_id

                    if len(node_rows) >= batch_size:
                        node_rows = _flush_node_rows(session, node_rows)
                    if len(rel_rows) >= batch_size:
                        rel_rows = _flush_rel_rows(session, rel_rows)

            node_rows = _flush_node_rows(session, node_rows)
            rel_rows = _flush_rel_rows(session, rel_rows)

            emitted_ifc_edges: set[tuple[str, str, str]] = set()
            emitted_context_nodes: set[str] = set()
            emitted_derived_edges: set[tuple[str, str, str]] = set()
            node_rows = []
            rel_rows = []

            for jsonl_path in resolved_paths:
                dataset_key = _dataset_key_from_path(jsonl_path)
                node_id_by_gid = node_id_by_gid_by_dataset.get(dataset_key, {})
                for _, rec in _yield_jsonl_records(jsonl_path):
                    gid = rec.get("GlobalId")
                    if not gid:
                        continue
                    node_id = node_id_by_gid.get(gid)
                    if node_id is None:
                        continue

                    parent_gid = (rec.get("Hierarchy") or {}).get("ParentId")
                    if isinstance(parent_gid, str) and parent_gid:
                        parent_node_id = node_id_by_gid.get(parent_gid)
                        if parent_node_id is not None and parent_node_id != node_id:
                            edge_index = _append_rel_row(
                                rel_rows,
                                edge_index=edge_index,
                                from_id=parent_node_id,
                                to_id=node_id,
                                relation="contains",
                                dataset=dataset_key,
                            )
                            edge_index = _append_rel_row(
                                rel_rows,
                                edge_index=edge_index,
                                from_id=node_id,
                                to_id=parent_node_id,
                                relation="contained_in",
                                dataset=dataset_key,
                            )

                    rels = rec.get("Relationships")
                    if not isinstance(rels, dict):
                        continue

                    for target_gid in rels.get("hosts") or []:
                        target = node_id_by_gid.get(target_gid)
                        if target and target != node_id:
                            edge_index = _emit_ifc_edge(
                                rel_rows,
                                emitted_ifc_edges,
                                edge_index=edge_index,
                                from_id=node_id,
                                to_id=target,
                                relation="hosts",
                                dataset=dataset_key,
                            )

                    for target_gid in rels.get("hosted_by") or []:
                        target = node_id_by_gid.get(target_gid)
                        if target and target != node_id:
                            edge_index = _emit_ifc_edge(
                                rel_rows,
                                emitted_ifc_edges,
                                edge_index=edge_index,
                                from_id=node_id,
                                to_id=target,
                                relation="hosted_by",
                                dataset=dataset_key,
                            )

                    for target_gid in rels.get("ifc_connected_to") or []:
                        target = node_id_by_gid.get(target_gid)
                        if target and target != node_id:
                            edge_index = _emit_ifc_edge(
                                rel_rows,
                                emitted_ifc_edges,
                                edge_index=edge_index,
                                from_id=node_id,
                                to_id=target,
                                relation="ifc_connected_to",
                                dataset=dataset_key,
                            )
                            edge_index = _emit_ifc_edge(
                                rel_rows,
                                emitted_ifc_edges,
                                edge_index=edge_index,
                                from_id=target,
                                to_id=node_id,
                                relation="ifc_connected_to",
                                dataset=dataset_key,
                            )

                    for target_gid in rels.get("typed_by") or []:
                        target = node_id_by_gid.get(target_gid)
                        if target and target != node_id:
                            edge_index = _emit_ifc_edge(
                                rel_rows,
                                emitted_ifc_edges,
                                edge_index=edge_index,
                                from_id=node_id,
                                to_id=target,
                                relation="typed_by",
                                dataset=dataset_key,
                            )

                    for target_gid in rels.get("path_connected_to") or []:
                        target = node_id_by_gid.get(target_gid)
                        if target and target != node_id:
                            edge_index = _emit_ifc_edge(
                                rel_rows,
                                emitted_ifc_edges,
                                edge_index=edge_index,
                                from_id=node_id,
                                to_id=target,
                                relation="path_connected_to",
                                dataset=dataset_key,
                            )
                            edge_index = _emit_ifc_edge(
                                rel_rows,
                                emitted_ifc_edges,
                                edge_index=edge_index,
                                from_id=target,
                                to_id=node_id,
                                relation="path_connected_to",
                                dataset=dataset_key,
                            )

                    for target_gid in rels.get("space_bounded_by") or []:
                        target = node_id_by_gid.get(target_gid)
                        if target and target != node_id:
                            edge_index = _emit_ifc_edge(
                                rel_rows,
                                emitted_ifc_edges,
                                edge_index=edge_index,
                                from_id=node_id,
                                to_id=target,
                                relation="space_bounded_by",
                                dataset=dataset_key,
                            )

                    for target_gid in rels.get("bounds_space") or []:
                        target = node_id_by_gid.get(target_gid)
                        if target and target != node_id:
                            edge_index = _emit_ifc_edge(
                                rel_rows,
                                emitted_ifc_edges,
                                edge_index=edge_index,
                                from_id=node_id,
                                to_id=target,
                                relation="bounds_space",
                                dataset=dataset_key,
                            )

                    context_specs = (
                        (
                            "belongs_to_system",
                            "System",
                            "IfcSystem",
                        ),
                        (
                            "in_zone",
                            "Zone",
                            "IfcZone",
                        ),
                        (
                            "classified_as",
                            "Classification",
                            "IfcClassificationReference",
                        ),
                    )
                    for rel_key, kind, class_name in context_specs:
                        for raw_label in rels.get(rel_key) or []:
                            if not isinstance(raw_label, str):
                                continue
                            label = _normalize_context_label(raw_label)
                            if not label:
                                continue
                            context_node_id = _context_node_id(
                                kind,
                                dataset_key,
                                label,
                                namespaced=namespaced_ids,
                            )
                            if context_node_id not in emitted_context_nodes:
                                emitted_context_nodes.add(context_node_id)
                                node_rows.append(
                                    _context_node_row(
                                        context_node_id,
                                        label=label,
                                        class_=class_name,
                                        dataset=dataset_key,
                                    )
                                )
                            edge_index = _emit_ifc_edge(
                                rel_rows,
                                emitted_ifc_edges,
                                edge_index=edge_index,
                                from_id=node_id,
                                to_id=context_node_id,
                                relation=rel_key,
                                dataset=dataset_key,
                            )

                    if len(node_rows) >= batch_size:
                        node_rows = _flush_node_rows(session, node_rows)
                    if len(rel_rows) >= batch_size:
                        rel_rows = _flush_rel_rows(session, rel_rows)

            node_rows = _flush_node_rows(session, node_rows)
            rel_rows = _flush_rel_rows(session, rel_rows)

            for jsonl_path in resolved_paths:
                dataset_key = _dataset_key_from_path(jsonl_path)
                derived_graph = _build_element_geometry_catalog(
                    jsonl_path,
                    namespaced_ids=namespaced_ids,
                )
                if derived_graph.number_of_nodes() == 0:
                    LOG.info(
                        "Skipping derived-edge pass for %s: "
                        "no geometry-bearing elements",
                        dataset_key,
                    )
                    continue

                LOG.info("Deriving spatial/topology edges for %s", dataset_key)
                for node_ids in _spatial_partition_node_sets(derived_graph):
                    if len(node_ids) < 2:
                        continue
                    spatial_graph = derived_graph.subgraph(node_ids)
                    add_spatial_adjacency(spatial_graph)
                    edge_index = _append_derived_rows_from_graph(
                        rel_rows,
                        spatial_graph,
                        edge_index=edge_index,
                        emitted_edges=emitted_derived_edges,
                    )
                _add_topology_facts_pruned(derived_graph)
                edge_index = _append_derived_rows_from_graph(
                    rel_rows,
                    derived_graph,
                    edge_index=edge_index,
                    emitted_edges=emitted_derived_edges,
                )
                rel_rows = _flush_rel_rows(session, rel_rows)
    finally:
        driver.close()


def import_graph_catalog(
    graph: GraphCatalog,
    *,
    batch_size: int = 2000,
    database: str | None = None,
    uri: str | None = None,
    username: str | None = None,
    password: str | None = None,
) -> None:
    _load_dotenv()
    if GraphDatabase is None:
        raise RuntimeError("neo4j package not installed")

    uri = uri or (os.environ.get("NEO4J_URI") or "").strip()
    username = username or (os.environ.get("NEO4J_USERNAME") or "").strip()
    password = password or (os.environ.get("NEO4J_PASSWORD") or "").strip()
    database = database or (os.environ.get("NEO4J_DATABASE") or "").strip() or None
    if not uri or not username or not password:
        raise RuntimeError("NEO4J_URI/NEO4J_USERNAME/NEO4J_PASSWORD not set")

    # Import is parity-first: build the canonical in-memory graph, then project
    # it into Neo4j.
    driver = GraphDatabase.driver(uri, auth=(username, password))
    try:
        with driver.session(database=database) as session:
            # Truncate first to keep imports deterministic and avoid collisions.
            session.run(CLEAR_DB)
            session.run(CREATE_CONSTRAINT)

            node_rows: list[dict[str, Any]] = []
            for node_id, data in graph.nodes(data=True):
                props = data.get("properties") or {}
                if not isinstance(props, dict):
                    props = {}
                prop_scalars = _extract_property_scalars(props)
                node_rows.append(
                    {
                        "node_id": node_id,
                        "label": data.get("label"),
                        "class_": data.get("class_"),
                        "node_kind": data.get("node_kind"),
                        "dataset": data.get("dataset"),
                        "properties_json": _encode_json(props),
                        "payload_json": _encode_json(data.get("payload") or {}),
                        "geometry_json": _encode_json(_geometry_payload(data)),
                        **prop_scalars,
                    }
                )
                if len(node_rows) >= batch_size:
                    session.run(INSERT_NODES, rows=node_rows)
                    node_rows = []

            if node_rows:
                session.run(INSERT_NODES, rows=node_rows)

            rel_rows: list[dict[str, Any]] = []
            # Preserve original edge ordering for parallel relations.
            edge_index = 0
            edge_iter = graph.edges(keys=True, data=True)

            for u, v, _k, data in edge_iter:
                rel_rows.append(
                    {
                        "from": u,
                        "to": v,
                        "relation": data.get("relation"),
                        "source": data.get("source"),
                        "dataset": data.get("dataset")
                        or graph.nodes[u].get("dataset")
                        or graph.nodes[v].get("dataset"),
                        "distance": data.get("distance"),
                        "vertical_gap": data.get("vertical_gap"),
                        "overlap_area_xy": data.get("overlap_area_xy"),
                        "intersection_volume": data.get("intersection_volume"),
                        "contact_area": data.get("contact_area"),
                        "edge_index": edge_index,
                    }
                )
                edge_index += 1
                if len(rel_rows) >= batch_size:
                    session.run(INSERT_RELS, rows=rel_rows)
                    rel_rows = []

            if rel_rows:
                session.run(INSERT_RELS, rows=rel_rows)

    finally:
        driver.close()


def _resolve_jsonl_paths(jsonl_dir: Path | None, dataset: str | None) -> list[Path]:
    if jsonl_dir is None:
        jsonl_dir = Path("output")
    jsonl_dir = jsonl_dir.expanduser().resolve()
    if dataset:
        candidate = jsonl_dir / f"{dataset}.jsonl"
        if not candidate.is_file():
            raise FileNotFoundError(f"JSONL file not found: {candidate}")
        return [candidate]
    paths = sorted(jsonl_dir.glob("*.jsonl"))
    if not paths:
        raise FileNotFoundError(f"No .jsonl files found in {jsonl_dir}")
    return paths


def main() -> None:
    ap = argparse.ArgumentParser(description="Import IFC graph into Neo4j.")
    ap.add_argument(
        "--jsonl-dir",
        type=Path,
        default=None,
        help="Directory containing .jsonl files (default: ./output).",
    )
    ap.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="JSONL stem to load (e.g. Building-Architecture).",
    )
    ap.add_argument(
        "--batch-size",
        type=int,
        default=2000,
        help="Batch size for Neo4j UNWIND inserts.",
    )
    args = ap.parse_args()

    jsonl_paths = _resolve_jsonl_paths(args.jsonl_dir, args.dataset)
    LOG.info(
        "Importing %d JSONL file(s) into Neo4j with direct streaming",
        len(jsonl_paths),
    )
    import_jsonl_files(jsonl_paths, batch_size=args.batch_size, payload_mode="full")
    LOG.info("Neo4j import complete")


import_networkx_graph = import_graph_catalog


if __name__ == "__main__":
    main()
