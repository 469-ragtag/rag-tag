from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import rag_tag.parser.ifc_geometry_parse as geometry_module
import rag_tag.parser.ifc_to_jsonl as ifc_to_jsonl_module
from rag_tag.parser.ifc_geometry_parse import ElementGeometry
from rag_tag.parser.ifc_relationships import empty_relation_block


class _FakeElement:
    GlobalId = "wall-guid"
    Name = "Wall A"
    Description = "Example wall"
    ObjectType = "Basic Wall"
    Tag = "W-01"
    PredefinedType = "STANDARD"

    def id(self) -> int:
        return 101

    def is_a(self) -> str:
        return "IfcWall"


def test_get_element_geometry_data_uses_single_shape_pass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    call_count = 0

    def fake_create_shape(_settings, _element):
        nonlocal call_count
        call_count += 1
        return SimpleNamespace(
            geometry=SimpleNamespace(
                verts=[
                    0.0,
                    0.0,
                    0.0,
                    2.0,
                    0.0,
                    0.0,
                    0.0,
                    2.0,
                    4.0,
                ],
                faces=[0, 1, 2],
            )
        )

    monkeypatch.setattr(
        geometry_module.ifcopenshell.geom,
        "create_shape",
        fake_create_shape,
    )

    geometry = geometry_module.get_element_geometry_data(_FakeElement(), object())

    assert call_count == 1
    assert geometry.vertices is not None
    assert geometry.faces is not None
    assert geometry.centroid is not None
    assert geometry.bbox is not None
    assert geometry.vertices.shape == (3, 3)
    assert geometry.faces.shape == (1, 3)
    assert np.allclose(geometry.centroid, np.array([2 / 3, 2 / 3, 4 / 3]))
    assert np.allclose(geometry.bbox[0], np.array([0.0, 0.0, 0.0]))
    assert np.allclose(geometry.bbox[1], np.array([2.0, 2.0, 4.0]))


def test_extract_element_preserves_geometry_payload_from_single_pass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    geometry_calls = 0
    geometry = ElementGeometry(
        vertices=np.array(
            [
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [0.0, 2.0, 4.0],
            ],
            dtype=float,
        ),
        faces=np.array([[0, 1, 2]], dtype=int),
        centroid=np.array([2 / 3, 2 / 3, 4 / 3], dtype=float),
        bbox=(
            np.array([0.0, 0.0, 0.0], dtype=float),
            np.array([2.0, 2.0, 4.0], dtype=float),
        ),
    )

    def fake_geometry_data(_element, _settings):
        nonlocal geometry_calls
        geometry_calls += 1
        return geometry

    def fake_get_psets(_element, *, psets_only: bool = False, qtos_only: bool = False):
        if psets_only:
            return {
                "Pset_WallCommon": {"Reference": "WallRef", "id": 10},
                "CustomSet": {"CustomProp": 7},
            }
        if qtos_only:
            return {"Qto_WallBaseQuantities": {"Length": 3.5}}
        return {}

    registry = SimpleNamespace(
        normalize_class=lambda _raw_class: {
            "canonical": "IfcWall",
            "ancestors": ["IfcBuildingElement"],
        }
    )

    monkeypatch.setattr(
        ifc_to_jsonl_module,
        "get_element_geometry_data",
        fake_geometry_data,
    )
    monkeypatch.setattr(
        ifc_to_jsonl_module,
        "compute_footprint_polygon_2d",
        lambda _vertices: np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]], dtype=float),
    )
    monkeypatch.setattr(
        ifc_to_jsonl_module,
        "compute_oriented_bbox",
        lambda _vertices: {
            "center": np.array([1.0, 1.0, 2.0], dtype=float),
            "axes": np.eye(3, dtype=float),
            "extents": np.array([1.0, 1.0, 2.0], dtype=float),
            "corners_xy": np.array(
                [[2.0, 2.0], [2.0, 0.0], [0.0, 0.0], [0.0, 2.0]],
                dtype=float,
            ),
        },
    )
    monkeypatch.setattr(
        ifc_to_jsonl_module,
        "get_element_local_placement_matrix",
        lambda _element: np.eye(4, dtype=float),
    )
    monkeypatch.setattr(
        ifc_to_jsonl_module.ifc_element,
        "get_type",
        lambda _element: SimpleNamespace(Name="Wall Type"),
    )
    monkeypatch.setattr(
        ifc_to_jsonl_module.ifc_element,
        "get_container",
        lambda _element: None,
    )
    monkeypatch.setattr(
        ifc_to_jsonl_module.ifc_element,
        "get_psets",
        fake_get_psets,
    )
    monkeypatch.setattr(
        ifc_to_jsonl_module.ifc_element,
        "get_materials",
        lambda _element: [SimpleNamespace(Name="Concrete")],
    )

    record = ifc_to_jsonl_module.extract_element(
        _FakeElement(),
        {"IfcWall": {"ValidPsets": ["Pset_WallCommon"]}},
        object(),
        registry,
        schema_supports_ontology=True,
        relationship_index={"wall-guid": empty_relation_block()},
    )

    assert geometry_calls == 1
    assert record["GlobalId"] == "wall-guid"
    assert record["IfcType"] == "IfcWall"
    assert record["TypeName"] == "Wall Type"
    assert record["Materials"] == ["Concrete"]
    assert record["PropertySets"]["Official"] == {
        "Pset_WallCommon": {"Reference": "WallRef"}
    }
    assert record["PropertySets"]["Custom"] == {"CustomSet": {"CustomProp": 7}}
    assert record["Quantities"] == {"Qto_WallBaseQuantities": {"Length": 3.5}}
    assert record["Geometry"]["Centroid"] == pytest.approx([2 / 3, 2 / 3, 4 / 3])
    assert record["Geometry"]["BoundingBox"]["min"] == pytest.approx([0.0, 0.0, 0.0])
    assert record["Geometry"]["BoundingBox"]["max"] == pytest.approx([2.0, 2.0, 4.0])
    assert np.allclose(
        np.asarray(record["Geometry"]["MeshVertices"], dtype=float),
        np.asarray([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 4.0]], dtype=float),
    )
    assert record["Geometry"]["MeshFaces"] == [[0, 1, 2]]
    assert np.allclose(
        np.asarray(record["Geometry"]["FootprintPolygon2D"], dtype=float),
        np.asarray([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]], dtype=float),
    )
    assert np.allclose(
        np.asarray(record["Geometry"]["LocalPlacementMatrix"], dtype=float),
        np.eye(4),
    )
    assert record["Geometry"]["OrientedBoundingBox"]["center"] == pytest.approx(
        [1.0, 1.0, 2.0]
    )
