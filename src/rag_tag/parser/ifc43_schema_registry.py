"""Registry for IFC class normalization and expected standard properties.

Uses embedded IFC4.3 mappings with optional ancestry augmentation from RDF.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TypedDict

import ifcopenshell

logger = logging.getLogger(__name__)


class NormalizedClassResult(TypedDict):
    """Normalized class metadata returned by ``normalize_class``."""

    raw: str  # Raw class from ``obj.is_a()``.
    canonical: str  # Nearest ancestor with known standard psets.
    base: str  # Compatibility alias for canonical.
    # Ancestors ordered from nearest parent toward IfcRoot.
    ancestors: list[str]


# IFC4.3 ADD2 reference property and quantity sets by canonical class.
STANDARD_PSETS: dict[str, dict[str, list[str]]] = {
    "IfcWall": {
        "Pset_WallCommon": [
            "Reference",  # internal reference code
            "AcousticRating",  # sound insulation rating
            "FireRating",  # e.g. REI 60
            "Combustible",  # is the material combustible?
            "SurfaceSpreadOfFlame",
            "ThermalTransmittance",  # U-value
            "IsExternal",  # outer wall or inner wall?
            "ExtendToStructure",  # does it go all the way to the slab?
            "LoadBearing",
            "Compartmentation",  # acts as a fire compartment boundary?
        ],
        "Qto_WallBaseQuantities": [
            "Length",
            "Width",
            "Height",
            "GrossFootprintArea",
            "NetFootprintArea",
            "GrossSideArea",
            "NetSideArea",
            "GrossVolume",
            "NetVolume",
        ],
    },
    "IfcSlab": {
        "Pset_SlabCommon": [
            "Reference",
            "AcousticRating",
            "FireRating",
            "Combustible",
            "SurfaceSpreadOfFlame",
            "ThermalTransmittance",
            "IsExternal",
            "LoadBearing",
            "PitchAngle",  # for sloped slabs / roofs
        ],
        "Qto_SlabBaseQuantities": [
            "Depth",
            "Perimeter",
            "GrossArea",
            "NetArea",
            "GrossVolume",
            "NetVolume",
        ],
    },
    "IfcDoor": {
        "Pset_DoorCommon": [
            "Reference",
            "FireRating",
            "AcousticRating",
            "SecurityRating",
            "IsExternal",
            "HandicapAccessible",
            "ThermalTransmittance",
            "WaterTightnessRating",
        ],
        "Qto_DoorBaseQuantities": [
            "Width",
            "Height",
            "Perimeter",
            "Area",
        ],
    },
    "IfcWindow": {
        "Pset_WindowCommon": [
            "Reference",
            "FireRating",
            "AcousticRating",
            "ThermalTransmittance",
            "IsExternal",
            "SecurityRating",
            "SolarHeatGainCoefficient",
            "MeanSolarTransmittance",
            "GlazingAreaFraction",
        ],
        "Qto_WindowBaseQuantities": [
            "Width",
            "Height",
            "Perimeter",
            "Area",
        ],
    },
    "IfcColumn": {
        "Pset_ColumnCommon": [
            "Reference",
            "LoadBearing",
            "FireRating",
            "IsExternal",
        ],
        "Qto_ColumnBaseQuantities": [
            "Length",
            "CrossSectionArea",
            "GrossVolume",
            "NetVolume",
            "GrossSurfaceArea",
            "NetSurfaceArea",
        ],
    },
    "IfcBeam": {
        "Pset_BeamCommon": [
            "Reference",
            "LoadBearing",
            "FireRating",
            "IsExternal",
            "Span",
        ],
        "Qto_BeamBaseQuantities": [
            "Length",
            "CrossSectionArea",
            "GrossVolume",
            "NetVolume",
            "GrossSurfaceArea",
            "NetSurfaceArea",
        ],
    },
    "IfcRoof": {
        "Pset_RoofCommon": [
            "Reference",
            "FireRating",
            "IsExternal",
            "ThermalTransmittance",
            "AcousticRating",
        ],
        "Qto_RoofBaseQuantities": [
            "GrossArea",
            "NetArea",
            "GrossVolume",
            "NetVolume",
        ],
    },
    "IfcStair": {
        "Pset_StairCommon": [
            "Reference",
            "FireRating",
            "IsExternal",
            "HandicapAccessible",
            "NumberOfRiser",
            "NumberOfTreads",
            "RiserHeight",
            "TreadLength",
        ],
        "Qto_StairBaseQuantities": [
            "Length",
            "GrossArea",
            "NetArea",
            "GrossVolume",
            "NetVolume",
        ],
    },
    "IfcRamp": {
        "Pset_RampCommon": [
            "Reference",
            "FireRating",
            "IsExternal",
            "HandicapAccessible",
        ],
        "Qto_RampBaseQuantities": [
            "Length",
            "GrossArea",
            "NetArea",
            "GrossVolume",
            "NetVolume",
        ],
    },
    "IfcFurniture": {
        # No standard Qto for furniture — dimensions come from the Pset
        "Pset_FurnitureTypeCommon": [
            "Reference",
            "NominalLength",
            "NominalWidth",
            "NominalHeight",
            "Style",
        ],
    },
    "IfcBuildingElementProxy": {
        # Catch-all for things that don't fit a proper IFC class yet
        "Pset_BuildingElementProxyCommon": [
            "Reference",
            "LoadBearing",
            "FireRating",
            "IsExternal",
        ],
    },
    "IfcCovering": {
        "Pset_CoveringCommon": [
            "Reference",
            "AcousticRating",
            "FireRating",
            "FlammabilityRating",
            "SurfaceSpreadOfFlame",
            "ThermalTransmittance",
            "IsExternal",
        ],
        "Qto_CoveringBaseQuantities": [
            "GrossArea",
            "NetArea",
            "Width",
        ],
    },
    "IfcRailing": {
        "Pset_RailingCommon": [
            "Reference",
            "IsExternal",
        ],
    },
    # MEP (mechanical / electrical / plumbing) elements
    # These tend to have sparse standard Psets compared to structural elements
    "IfcFlowSegment": {
        "Pset_FlowSegmentCommon": ["Reference"],
    },
    "IfcFlowTerminal": {
        "Pset_FlowTerminalCommon": ["Reference"],
    },
    "IfcFlowFitting": {
        "Pset_FlowFittingCommon": ["Reference"],
    },
    "IfcDuctSegment": {
        "Pset_DuctSegmentCommon": [
            "Reference",
            "Shape",
            "NominalDiameter",
            "NominalWidth",
            "NominalHeight",
        ],
        "Qto_DuctSegmentBaseQuantities": [
            "Length",
            "GrossWeight",
        ],
    },
    "IfcPipeSegment": {
        "Pset_PipeSegmentCommon": [
            "Reference",
            "Shape",
            "NominalDiameter",
        ],
        "Qto_PipeSegmentBaseQuantities": [
            "Length",
            "GrossWeight",
        ],
    },
    "IfcMember": {
        "Pset_MemberCommon": [
            "Reference",
            "LoadBearing",
            "FireRating",
            "IsExternal",
            "Span",
        ],
        "Qto_MemberBaseQuantities": [
            "Length",
            "GrossVolume",
            "NetVolume",
            "GrossSurfaceArea",
            "NetSurfaceArea",
        ],
    },
    "IfcPlate": {
        "Pset_PlateCommon": [
            "Reference",
            "AcousticRating",
            "FireRating",
            "IsExternal",
        ],
        "Qto_PlateBaseQuantities": [
            "GrossArea",
            "NetArea",
            "GrossVolume",
            "NetVolume",
        ],
    },
    "IfcFooting": {
        "Pset_FootingCommon": [
            "Reference",
            "LoadBearing",
            "IsExternal",
        ],
    },
    "IfcPile": {
        "Pset_PileCommon": [
            "Reference",
            "LoadBearing",
            "FireRating",
            "IsExternal",
        ],
    },
    # Spatial / organisational elements
    "IfcBuildingStorey": {
        "Pset_BuildingStoreyCommon": [
            "Reference",
            "EntranceLevel",  # is this the entrance floor?
            "AboveGround",
            "SprinklerProtection",
            "SprinklerProtectionType",
        ],
        "Qto_BuildingStoreyBaseQuantities": [
            "GrossFloorArea",
            "NetFloorArea",
            "GrossPerimeter",
        ],
    },
    "IfcSpace": {
        "Pset_SpaceCommon": [
            "Reference",
            "IsExternal",
            "GrossPlannedArea",
            "NetPlannedArea",
            "PubliclyAccessible",
            "HandicapAccessible",
        ],
        "Qto_SpaceBaseQuantities": [
            "Height",
            "NetFloorArea",
            "GrossFloorArea",
            "NetCeilingArea",
            "GrossCeilingArea",
            "NetWallArea",
            "GrossWallArea",
        ],
    },
    "IfcZone": {
        "Pset_ZoneCommon": [
            "Reference",
            "Category",
            "NetPlannedArea",
        ],
    },
    "IfcBuilding": {
        "Pset_BuildingCommon": [
            "Reference",
            "NumberOfStoreys",
            "OccupancyType",
            "IsLandmarked",
            "MainFireUse",
            "AncillaryFireUse",
        ],
        "Qto_BuildingBaseQuantities": [
            "GrossFloorArea",
            "NetFloorArea",
            "GrossVolume",
        ],
    },
    "IfcSite": {
        "Pset_SiteCommon": [
            "Reference",
            "BuildableArea",
            "TotalArea",
            "BuildingHeightLimit",
        ],
    },
    "IfcProject": {
        "Pset_ProjectCommon": [
            "Reference",
        ],
    },
}


def _normalize_schema_name(name: str) -> str:
    """Normalize schema labels for ``ifcopenshell.schema_by_name``."""
    n = name.upper().replace(" ", "")
    if n.startswith("IFC4X3"):
        return "IFC4X3"
    if n.startswith("IFC4"):
        return "IFC4"
    if n.startswith("IFC2X3"):
        return "IFC2X3"
    return name


class IFC43SchemaRegistry:
    """Resolve IFC class ancestry and standard property expectations."""

    def __init__(
        self,
        snapshot_path: Path | None = None,
        schema_name: str | None = None,
    ) -> None:
        self._schema_name = _normalize_schema_name(schema_name or "IFC4")

        # class name -> ancestors from nearest parent to IfcRoot
        self._hierarchy: dict[str, list[str]] = {}

        # Keep instance-local copy so optional augmentation never mutates globals.
        self._psets: dict[str, dict[str, list[str]]] = {
            cls: dict(psets) for cls, psets in STANDARD_PSETS.items()
        }

        self._build_hierarchy()

        if snapshot_path and snapshot_path.exists():
            self._load_from_rdf(snapshot_path)
        elif snapshot_path:
            logger.warning(
                "bSDD snapshot not found at %s — using embedded schema only. "
                "Run `uv run rag-tag-refresh-ifc43-rdf` to download it.",
                snapshot_path,
            )

    def _build_hierarchy(self) -> None:
        """Build ancestor chains from the local ifcopenshell schema."""
        try:
            schema = ifcopenshell.schema_by_name(self._schema_name)
        except Exception as exc:
            logger.warning(
                "Could not load IFC schema '%s': %s. "
                "Class hierarchy will be empty — normalization won't work properly.",
                self._schema_name,
                exc,
            )
            return

        def _get_ancestor_chain(entity) -> list[str]:
            """Return ancestors from direct parent upward."""
            ancestors: list[str] = []
            try:
                current = entity.supertype()
                while current is not None:
                    ancestors.append(current.name())
                    current = current.supertype()
            except Exception:
                pass
            return ancestors

        # Seed canonical classes first.
        for class_name in STANDARD_PSETS:
            try:
                entity = schema.declaration_by_name(class_name)
                self._hierarchy[class_name] = _get_ancestor_chain(entity)
            except Exception:
                self._hierarchy.setdefault(class_name, [])

        # Then add any additional declarations exposed by this ifcopenshell build.
        try:
            for decl in schema.declarations():
                name = decl.name()
                if name.startswith("Ifc") and name not in self._hierarchy:
                    self._hierarchy[name] = _get_ancestor_chain(decl)
        except (AttributeError, TypeError):
            pass

        logger.debug(
            "Built IFC hierarchy from schema '%s': %d classes indexed",
            self._schema_name,
            len(self._hierarchy),
        )

    def _load_from_rdf(self, path: Path) -> None:
        """Augment hierarchy with ``rdfs:subClassOf`` edges from RDF."""
        try:
            import rdflib
            import rdflib.namespace
        except ImportError:
            logger.warning(
                "rdflib is not installed so we can't read the bSDD Turtle file. "
                "The embedded STANDARD_PSETS will still be used — "
                "install rdflib with `uv add rdflib` to enable RDF augmentation."
            )
            return

        try:
            g = rdflib.Graph()
            g.parse(str(path))
            RDFS = rdflib.namespace.RDFS

            added = 0
            for cls_node, _, parent_node in g.triples((None, RDFS.subClassOf, None)):
                cls_name = str(cls_node).rsplit("/", 1)[-1].rsplit("#", 1)[-1]
                parent_name = str(parent_node).rsplit("/", 1)[-1].rsplit("#", 1)[-1]

                if not (cls_name.startswith("Ifc") and parent_name.startswith("Ifc")):
                    continue

                chain = self._hierarchy.setdefault(cls_name, [])
                if parent_name not in chain:
                    # NOTE: Keep nearest parent first.
                    chain.insert(0, parent_name)
                    added += 1

            logger.info(
                "Loaded bSDD RDF from %s: %d triples, %d new relationships added",
                path,
                len(g),
                added,
            )
        except Exception as exc:
            logger.warning("Failed to parse bSDD RDF file at %s: %s", path, exc)

    def normalize_class(self, raw_class: str) -> NormalizedClassResult:
        """Map a raw class to the nearest canonical class with known psets."""
        ancestors = self._hierarchy.get(raw_class, [])

        canonical = raw_class
        if raw_class not in self._psets:
            for ancestor in ancestors:
                if ancestor in self._psets:
                    canonical = ancestor
                    break

        return NormalizedClassResult(
            raw=raw_class,
            canonical=canonical,
            base=canonical,
            ancestors=ancestors,
        )

    def expected_properties_for_class(self, ifc_class: str) -> set[str]:
        """Return standard ``Pset.Property`` names for a class lineage."""
        result: set[str] = set()

        classes_to_check = [ifc_class, *self._hierarchy.get(ifc_class, [])]
        for cls in classes_to_check:
            if cls in self._psets:
                for pset_name, props in self._psets[cls].items():
                    for prop in props:
                        result.add(f"{pset_name}.{prop}")

        return result

    def is_known_property(self, pset: str, prop: str) -> bool:
        """Return whether a pset/property pair exists in known IFC mappings."""
        for class_psets in self._psets.values():
            if pset in class_psets and prop in class_psets[pset]:
                return True
        return False


# Cache by ``(snapshot_path, schema_name)`` for schema-safe reuse.
_registry_cache: dict[str, IFC43SchemaRegistry] = {}


def get_registry(
    snapshot_path: Path | None = None,
    schema_name: str | None = None,
) -> IFC43SchemaRegistry:
    """Return a cached registry keyed by snapshot path and schema name."""
    path_key = str(snapshot_path) if snapshot_path is not None else ""
    schema_key = schema_name or ""
    key = f"{path_key}::{schema_key}"
    if key not in _registry_cache:
        _registry_cache[key] = IFC43SchemaRegistry(snapshot_path, schema_name)
    return _registry_cache[key]
