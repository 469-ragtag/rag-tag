from __future__ import annotations

import math
from typing import Any

import numpy as np

ORIENTATION_ANGLE_TOLERANCE_DEG = 15.0
PERPENDICULAR_ANGLE_TOLERANCE_DEG = 15.0
SUPPORT_GAP_TOLERANCE = 0.05
FACING_MAX_DISTANCE = 2.0
CONTAINMENT_TOLERANCE = 1e-6
MAX_MESH_POINTS = 256


def centroid_from_node(node_data: dict[str, Any]) -> tuple[float, float, float] | None:
    geometry = node_data.get("geometry")
    if isinstance(geometry, (list, tuple)) and len(geometry) == 3:
        try:
            return (float(geometry[0]), float(geometry[1]), float(geometry[2]))
        except (TypeError, ValueError):
            return None
    bbox = bbox_from_node(node_data)
    if bbox is None:
        return None
    return tuple((bbox[0][i] + bbox[1][i]) / 2.0 for i in range(3))


def bbox_from_node(
    node_data: dict[str, Any],
) -> tuple[tuple[float, float, float], tuple[float, float, float]] | None:
    raw = node_data.get("bbox")
    if (
        not isinstance(raw, (list, tuple))
        or len(raw) != 2
        or not isinstance(raw[0], (list, tuple))
        or not isinstance(raw[1], (list, tuple))
        or len(raw[0]) != 3
        or len(raw[1]) != 3
    ):
        return None
    try:
        return (
            (float(raw[0][0]), float(raw[0][1]), float(raw[0][2])),
            (float(raw[1][0]), float(raw[1][1]), float(raw[1][2])),
        )
    except (TypeError, ValueError):
        return None


def mesh_from_node(
    node_data: dict[str, Any],
) -> tuple[list[tuple[float, float, float]], list[tuple[int, int, int]]] | None:
    mesh = node_data.get("mesh")
    if not isinstance(mesh, (list, tuple)) or len(mesh) != 2:
        return None
    vertices, faces = mesh
    if not isinstance(vertices, list) or not isinstance(faces, list):
        return None
    try:
        parsed_vertices = [tuple(float(v) for v in point) for point in vertices]
        parsed_faces = [tuple(int(i) for i in face) for face in faces]
    except (TypeError, ValueError):
        return None
    return parsed_vertices, parsed_faces


def distance_between_points(
    a: tuple[float, float, float],
    b: tuple[float, float, float],
) -> float:
    return float(np.linalg.norm(np.array(a, dtype=float) - np.array(b, dtype=float)))


def distance_between_bboxes(
    a: tuple[tuple[float, float, float], tuple[float, float, float]],
    b: tuple[tuple[float, float, float], tuple[float, float, float]],
) -> float:
    amin, amax = a
    bmin, bmax = b
    axis_gaps = [max(bmin[i] - amax[i], amin[i] - bmax[i], 0.0) for i in range(3)]
    return float(np.linalg.norm(np.array(axis_gaps, dtype=float)))


def mesh_min_vertex_distance(
    mesh_a: tuple[list[tuple[float, float, float]], list[tuple[int, int, int]]] | None,
    mesh_b: tuple[list[tuple[float, float, float]], list[tuple[int, int, int]]] | None,
    *,
    max_points_per_mesh: int = MAX_MESH_POINTS,
) -> float | None:
    if mesh_a is None or mesh_b is None:
        return None
    verts_a, _faces_a = mesh_a
    verts_b, _faces_b = mesh_b
    if not verts_a or not verts_b:
        return None
    try:
        arr_a = np.asarray(verts_a, dtype=float)
        arr_b = np.asarray(verts_b, dtype=float)
    except Exception:
        return None
    if arr_a.ndim != 2 or arr_a.shape[1] != 3:
        return None
    if arr_b.ndim != 2 or arr_b.shape[1] != 3:
        return None

    if len(arr_a) > max_points_per_mesh:
        idx = np.linspace(0, len(arr_a) - 1, num=max_points_per_mesh, dtype=int)
        arr_a = arr_a[idx]
    if len(arr_b) > max_points_per_mesh:
        idx = np.linspace(0, len(arr_b) - 1, num=max_points_per_mesh, dtype=int)
        arr_b = arr_b[idx]

    delta = arr_a[:, None, :] - arr_b[None, :, :]
    d2 = np.sum(delta * delta, axis=2)
    return float(np.sqrt(np.min(d2)))


def bbox_xy_overlap_area(
    a: tuple[tuple[float, float, float], tuple[float, float, float]],
    b: tuple[tuple[float, float, float], tuple[float, float, float]],
) -> float:
    ax0, ay0, _ = a[0]
    ax1, ay1, _ = a[1]
    bx0, by0, _ = b[0]
    bx1, by1, _ = b[1]
    overlap_x = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    overlap_y = max(0.0, min(ay1, by1) - max(ay0, by0))
    return float(overlap_x * overlap_y)


def bbox_intersection_volume(
    a: tuple[tuple[float, float, float], tuple[float, float, float]],
    b: tuple[tuple[float, float, float], tuple[float, float, float]],
) -> float:
    overlap = [
        max(0.0, min(a[1][i], b[1][i]) - max(a[0][i], b[0][i])) for i in range(3)
    ]
    return float(overlap[0] * overlap[1] * overlap[2])


def bbox_volume(
    bbox: tuple[tuple[float, float, float], tuple[float, float, float]],
) -> float:
    return float(
        max(0.0, bbox[1][0] - bbox[0][0])
        * max(0.0, bbox[1][1] - bbox[0][1])
        * max(0.0, bbox[1][2] - bbox[0][2])
    )


def bbox_vertical_overlap(
    a: tuple[tuple[float, float, float], tuple[float, float, float]],
    b: tuple[tuple[float, float, float], tuple[float, float, float]],
) -> float:
    return max(0.0, min(a[1][2], b[1][2]) - max(a[0][2], b[0][2]))


def vertical_gap_between_bboxes(
    a: tuple[tuple[float, float, float], tuple[float, float, float]],
    b: tuple[tuple[float, float, float], tuple[float, float, float]],
) -> float:
    if a[0][2] > b[1][2]:
        return float(a[0][2] - b[1][2])
    if b[0][2] > a[1][2]:
        return float(b[0][2] - a[1][2])
    return 0.0


def bbox_contains(
    outer: tuple[tuple[float, float, float], tuple[float, float, float]],
    inner: tuple[tuple[float, float, float], tuple[float, float, float]],
    *,
    tolerance: float = CONTAINMENT_TOLERANCE,
) -> bool:
    return all(
        outer[0][i] - tolerance <= inner[0][i] <= inner[1][i] <= outer[1][i] + tolerance
        for i in range(3)
    )


def dominant_horizontal_axis(
    node_data: dict[str, Any],
) -> tuple[float, float] | None:
    obb = node_data.get("obb")
    if isinstance(obb, dict):
        axes = obb.get("axes")
        extents = obb.get("extents")
        if (
            isinstance(axes, (list, tuple))
            and isinstance(extents, (list, tuple))
            and len(extents) == 3
        ):
            best_axis: tuple[float, float] | None = None
            best_score = 0.0
            for axis, extent in zip(axes, extents, strict=False):
                if not isinstance(axis, (list, tuple)) or len(axis) != 3:
                    continue
                vec = np.array([float(axis[0]), float(axis[1])], dtype=float)
                score = float(np.linalg.norm(vec)) * max(float(extent), 0.0)
                if score > best_score:
                    best_score = score
                    best_axis = _normalize_2d(vec)
            if best_axis is not None:
                return best_axis

    footprint = node_data.get("footprint_polygon")
    if isinstance(footprint, list) and len(footprint) >= 2:
        points: list[tuple[float, float]] = []
        for point in footprint:
            if isinstance(point, (list, tuple)) and len(point) == 2:
                points.append((float(point[0]), float(point[1])))
        axis = _dominant_axis_from_points(points)
        if axis is not None:
            return axis

    bbox = bbox_from_node(node_data)
    if bbox is None:
        return None
    dx = float(bbox[1][0] - bbox[0][0])
    dy = float(bbox[1][1] - bbox[0][1])
    if dx <= 0.0 and dy <= 0.0:
        return None
    return (1.0, 0.0) if dx >= dy else (0.0, 1.0)


def axis_angle_deg(
    axis_a: tuple[float, float] | None,
    axis_b: tuple[float, float] | None,
) -> float | None:
    if axis_a is None or axis_b is None:
        return None
    dot = abs(float(axis_a[0] * axis_b[0] + axis_a[1] * axis_b[1]))
    dot = max(-1.0, min(1.0, dot))
    return float(math.degrees(math.acos(dot)))


def compare_nodes_geometry(
    node_a: dict[str, Any],
    node_b: dict[str, Any],
    *,
    edge_metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    bbox_a = bbox_from_node(node_a)
    bbox_b = bbox_from_node(node_b)
    centroid_a = centroid_from_node(node_a)
    centroid_b = centroid_from_node(node_b)

    direction_vector: tuple[float, float, float] | None = None
    if centroid_a is not None and centroid_b is not None:
        direction_vector = tuple(float(centroid_b[i] - centroid_a[i]) for i in range(3))

    overlap_area_xy: float | None = None
    vertical_gap: float | None = None
    intersection_volume: float | None = None
    containment_ratio: float | None = None
    inside_or_contains: str | None = None

    if bbox_a is not None and bbox_b is not None:
        overlap_area_xy = bbox_xy_overlap_area(bbox_a, bbox_b)
        vertical_gap = vertical_gap_between_bboxes(bbox_a, bbox_b)
        intersection_volume = bbox_intersection_volume(bbox_a, bbox_b)
        if bbox_contains(bbox_a, bbox_b):
            inside_or_contains = "a_contains_b"
            inner_volume = bbox_volume(bbox_b)
            containment_ratio = 1.0 if inner_volume > 0.0 else None
        elif bbox_contains(bbox_b, bbox_a):
            inside_or_contains = "b_contains_a"
            inner_volume = bbox_volume(bbox_a)
            containment_ratio = 1.0 if inner_volume > 0.0 else None
        elif intersection_volume > 0.0:
            smaller = min(bbox_volume(bbox_a), bbox_volume(bbox_b))
            containment_ratio = (
                float(intersection_volume / smaller) if smaller > 0.0 else None
            )

    axis_a = dominant_horizontal_axis(node_a)
    axis_b = dominant_horizontal_axis(node_b)
    angle = axis_angle_deg(axis_a, axis_b)
    parallel_score: float | None = None
    perpendicular_score: float | None = None
    if axis_a is not None and axis_b is not None:
        dot = abs(float(axis_a[0] * axis_b[0] + axis_a[1] * axis_b[1]))
        parallel_score = max(0.0, min(1.0, dot))
        perpendicular_score = max(0.0, min(1.0, 1.0 - dot))

    facing_score = facing_score_between_nodes(
        node_a,
        node_b,
        axis_a=axis_a,
        axis_b=axis_b,
    )
    support_relation, support_score = support_relation_between_nodes(node_a, node_b)

    surface_distance: float | None = None
    source = "heuristic_bbox"
    verified = False
    distance_method = "bbox"

    if edge_metrics:
        if edge_metrics.get("relation") in {"intersects_3d", "touches_surface"}:
            surface_distance = 0.0
        elif edge_metrics.get("distance") is not None:
            try:
                surface_distance = float(edge_metrics["distance"])
                distance_method = str(edge_metrics.get("distance_method") or "edge")
            except (TypeError, ValueError):
                surface_distance = None
        if edge_metrics.get("intersection_volume") is not None:
            try:
                intersection_volume = float(edge_metrics["intersection_volume"])
            except (TypeError, ValueError):
                pass
        if edge_metrics.get("contact_area") is not None:
            try:
                contact_area = float(edge_metrics["contact_area"])
            except (TypeError, ValueError):
                contact_area = None
        else:
            contact_area = None
        verified = bool(edge_metrics.get("verified"))
        source = str(
            edge_metrics.get("distance_method") or edge_metrics.get("source") or source
        )
    else:
        contact_area = None

    if surface_distance is None:
        mesh_distance = mesh_min_vertex_distance(
            mesh_from_node(node_a),
            mesh_from_node(node_b),
        )
        if mesh_distance is not None:
            surface_distance = mesh_distance
            source = "mesh_vertices"
            distance_method = "mesh_vertices"
        elif bbox_a is not None and bbox_b is not None:
            surface_distance = distance_between_bboxes(bbox_a, bbox_b)
            source = "bbox"
            distance_method = "bbox"
        elif centroid_a is not None and centroid_b is not None:
            surface_distance = distance_between_points(centroid_a, centroid_b)
            source = "centroid"
            distance_method = "centroid"

    return {
        "surface_distance": surface_distance,
        "nearest_points": None,
        "direction_vector": direction_vector,
        "intersection_volume": intersection_volume,
        "contact_area": contact_area,
        "overlap_area_xy": overlap_area_xy,
        "vertical_gap": vertical_gap,
        "axis_angle_deg": angle,
        "parallel_score": parallel_score,
        "perpendicular_score": perpendicular_score,
        "facing_score": facing_score,
        "support_relation": support_relation,
        "support_score": support_score,
        "inside_or_contains": inside_or_contains,
        "containment_ratio": containment_ratio,
        "source": source,
        "distance_method": distance_method,
        "verified": verified,
    }


def support_relation_between_nodes(
    node_a: dict[str, Any],
    node_b: dict[str, Any],
    *,
    gap_tolerance: float = SUPPORT_GAP_TOLERANCE,
) -> tuple[str | None, float | None]:
    bbox_a = bbox_from_node(node_a)
    bbox_b = bbox_from_node(node_b)
    if bbox_a is None or bbox_b is None:
        return None, None
    overlap_area = bbox_xy_overlap_area(bbox_a, bbox_b)
    if overlap_area <= 0.0:
        return None, None

    footprint_a = max(
        (bbox_a[1][0] - bbox_a[0][0]) * (bbox_a[1][1] - bbox_a[0][1]),
        0.0,
    )
    footprint_b = max(
        (bbox_b[1][0] - bbox_b[0][0]) * (bbox_b[1][1] - bbox_b[0][1]),
        0.0,
    )
    smaller_footprint = min(footprint_a, footprint_b)
    overlap_ratio = overlap_area / smaller_footprint if smaller_footprint > 0.0 else 0.0

    if 0.0 <= bbox_b[0][2] - bbox_a[1][2] <= gap_tolerance:
        gap = float(bbox_b[0][2] - bbox_a[1][2])
        gap_score = 1.0 - min(1.0, gap / max(gap_tolerance, 1e-9))
        return "a_supports_b", float(max(0.0, min(1.0, overlap_ratio * gap_score)))
    if 0.0 <= bbox_a[0][2] - bbox_b[1][2] <= gap_tolerance:
        gap = float(bbox_a[0][2] - bbox_b[1][2])
        gap_score = 1.0 - min(1.0, gap / max(gap_tolerance, 1e-9))
        return "b_supports_a", float(max(0.0, min(1.0, overlap_ratio * gap_score)))
    return None, None


def facing_score_between_nodes(
    node_a: dict[str, Any],
    node_b: dict[str, Any],
    *,
    axis_a: tuple[float, float] | None = None,
    axis_b: tuple[float, float] | None = None,
    max_distance: float = FACING_MAX_DISTANCE,
) -> float | None:
    axis_a = axis_a or dominant_horizontal_axis(node_a)
    axis_b = axis_b or dominant_horizontal_axis(node_b)
    if axis_a is None or axis_b is None:
        return None

    centroid_a = centroid_from_node(node_a)
    centroid_b = centroid_from_node(node_b)
    bbox_a = bbox_from_node(node_a)
    bbox_b = bbox_from_node(node_b)
    if centroid_a is None or centroid_b is None or bbox_a is None or bbox_b is None:
        return None

    separation = np.array(
        [float(centroid_b[0] - centroid_a[0]), float(centroid_b[1] - centroid_a[1])],
        dtype=float,
    )
    separation_len = float(np.linalg.norm(separation))
    if separation_len <= 1e-9 or separation_len > max_distance:
        return None

    separation_axis = _normalize_2d(separation)
    if separation_axis is None:
        return None

    dot = abs(float(axis_a[0] * axis_b[0] + axis_a[1] * axis_b[1]))
    normal_alignment = math.sqrt(
        max(
            0.0,
            1.0
            - float(
                abs(separation_axis[0] * axis_a[0] + separation_axis[1] * axis_a[1])
            )
            ** 2,
        )
    )
    z_overlap = bbox_vertical_overlap(bbox_a, bbox_b)
    min_height = min(
        max(0.0, bbox_a[1][2] - bbox_a[0][2]),
        max(0.0, bbox_b[1][2] - bbox_b[0][2]),
    )
    z_factor = min(1.0, z_overlap / min_height) if min_height > 0.0 else 0.0
    distance_factor = 1.0 - min(1.0, separation_len / max_distance)
    score = dot * normal_alignment * max(z_factor, 0.25) * max(distance_factor, 0.1)
    return float(max(0.0, min(1.0, score)))


def is_parallel(
    node_a: dict[str, Any],
    node_b: dict[str, Any],
) -> tuple[bool, float | None]:
    angle = axis_angle_deg(
        dominant_horizontal_axis(node_a),
        dominant_horizontal_axis(node_b),
    )
    if angle is None:
        return False, None
    return angle <= ORIENTATION_ANGLE_TOLERANCE_DEG, angle


def is_perpendicular(
    node_a: dict[str, Any], node_b: dict[str, Any]
) -> tuple[bool, float | None]:
    angle = axis_angle_deg(
        dominant_horizontal_axis(node_a),
        dominant_horizontal_axis(node_b),
    )
    if angle is None:
        return False, None
    return abs(90.0 - angle) <= PERPENDICULAR_ANGLE_TOLERANCE_DEG, angle


def _normalize_2d(vec: np.ndarray) -> tuple[float, float] | None:
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-9:
        return None
    normalized = vec / norm
    return (float(normalized[0]), float(normalized[1]))


def _dominant_axis_from_points(
    points: list[tuple[float, float]],
) -> tuple[float, float] | None:
    if len(points) < 2:
        return None
    arr = np.asarray(points, dtype=float)
    centered = arr - arr.mean(axis=0)
    if centered.shape[0] < 2:
        return None
    try:
        covariance = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    except Exception:
        return None
    if len(eigenvalues) < 2:
        return None
    axis = eigenvectors[:, int(np.argmax(eigenvalues))]
    return _normalize_2d(np.asarray(axis, dtype=float))
