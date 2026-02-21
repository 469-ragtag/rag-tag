"""
ifc43_schema_registry.py

The problem this solves
-----------------------
When we parse an IFC file, we can only see the property sets (Psets) that are
*actually attached* to each element.  If an architect forgot to add
Pset_WallCommon to a wall, we get no fire rating, no thermal transmittance,
nothing.  Different models end up with totally different CSV columns, which
makes comparison really difficult.

The IFC 4.3 standard says "every IfcWall SHOULD have Pset_WallCommon" —
but the standard can't enforce that.  So we do it here.

This module gives us a schema registry that knows:
  - which properties *should* exist for every IFC class (from the spec)
  - the class hierarchy (IfcWallStandardCase → IfcWall → IfcBuildingElement…)
    so we can normalise subclasses back to their canonical parent

Then ifc_to_csv.py uses this registry to always include those property
columns in the output CSV, even if the IFC model left them empty.

Where the schema data comes from
---------------------------------
Primary source: the STANDARD_PSETS dict below — hand-curated from the IFC 4.3
ADD2 spec (https://ifc43-docs.buildingsmart.org/).  This is always available.

Optional augmentation: a local bSDD / IFC-OWL Turtle (.ttl) snapshot.
bSDD is buildingSMART's data dictionary — it's basically the authoritative
online version of the schema.  If we have a downloaded copy of it, we can
cross-check and augment the hierarchy.

To download a fresh copy of the snapshot run:
    uv run rag-tag-refresh-ifc43-rdf \\
        --url <snapshot-url> \\
        --out output/metadata/bsdd/ifc43.ttl

The snapshot lives in output/metadata/bsdd/ifc43.ttl by default.
You can override this with the --bsdd-rdf-path flag or BSDD_IFC43_RDF_PATH env var.

If the snapshot isn't there, no problem — we just use the embedded dict and
ifcopenshell's built-in schema.  Nothing crashes.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TypedDict

import ifcopenshell

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result type for normalize_class()
# ---------------------------------------------------------------------------

class NormalizedClassResult(TypedDict):
    """
    What we know about an IFC class after looking it up in the hierarchy.

    For example, if the IFC file has IfcWallStandardCase (a subclass of
    IfcWall), we want to map it back to IfcWall so we know which Psets
    to expect and how to group it with other walls.
    """
    raw: str        # exactly what obj.is_a() returned, e.g. "IfcWallStandardCase"
    canonical: str  # nearest ancestor we have Psets for, e.g. "IfcWall"
    base: str       # same as canonical for now — used for grouping in the CSV
    # full chain: ["IfcWall", "IfcBuildingElement", ..., "IfcRoot"]
    ancestors: list[str]


# ---------------------------------------------------------------------------
# Standard property set definitions
#
# Format: { IFC class → { pset/qto name → [property names] } }
#
# Every entry here is defined in the IFC 4.3 ADD2 specification.
# We list the properties that *should* be on each element type according
# to the standard — even if the model doesn't actually have them set.
#
# Pset_* = property sets (text / boolean / enum values)
# Qto_*  = quantity sets (numeric measurements)
# ---------------------------------------------------------------------------

STANDARD_PSETS: dict[str, dict[str, list[str]]] = {

    "IfcWall": {
        "Pset_WallCommon": [
            "Reference",           # internal reference code
            "AcousticRating",      # sound insulation rating
            "FireRating",          # e.g. REI 60
            "Combustible",         # is the material combustible?
            "SurfaceSpreadOfFlame",
            "ThermalTransmittance",  # U-value
            "IsExternal",          # outer wall or inner wall?
            "ExtendToStructure",   # does it go all the way to the slab?
            "LoadBearing",
            "Compartmentation",    # acts as a fire compartment boundary?
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
            "EntranceLevel",   # is this the entrance floor?
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


# ---------------------------------------------------------------------------
# Small helper — IFC schema version strings aren't always consistent
# ---------------------------------------------------------------------------

def _normalize_schema_name(name: str) -> str:
    """
    Make sure the schema name is in the format ifcopenshell expects.
    IFC files can say "IFC4", "IFC4x3", "IFC4X3_ADD2" etc.
    We just want the major version so we can call schema_by_name().
    """
    n = name.upper().replace(" ", "")
    if n.startswith("IFC4X3"):
        return "IFC4X3"
    if n.startswith("IFC4"):
        return "IFC4"
    if n.startswith("IFC2X3"):
        return "IFC2X3"
    return name


# ---------------------------------------------------------------------------
# The registry class
# ---------------------------------------------------------------------------

class IFC43SchemaRegistry:
    """
    Knows the IFC class hierarchy and which Psets/properties belong to each class.

    On creation it:
      1. Builds the class hierarchy from ifcopenshell's built-in schema
         (always works, no internet needed)
      2. Optionally loads a local bSDD Turtle file to augment the hierarchy
         with extra rdfs:subClassOf relationships (requires rdflib)

    Usage:
        registry = IFC43SchemaRegistry()
        result = registry.normalize_class("IfcWallStandardCase")
        # result["canonical"] → "IfcWall"

        props = registry.expected_properties_for_class("IfcWall")
        # props → {"Pset_WallCommon.FireRating", "Pset_WallCommon.IsExternal", ...}
    """

    def __init__(
        self,
        snapshot_path: Path | None = None,
        schema_name: str | None = None,
    ) -> None:
        self._schema_name = _normalize_schema_name(schema_name or "IFC4")

        # class name → list of ancestor names, ordered from nearest to IfcRoot
        # e.g. "IfcWall" → ["IfcBuildingElement", "IfcElement", ..., "IfcRoot"]
        self._hierarchy: dict[str, list[str]] = {}

        # working copy of STANDARD_PSETS — we keep it separate so RDF augmentation
        # can add to it without touching the module-level dict
        self._psets: dict[str, dict[str, list[str]]] = {
            cls: dict(psets) for cls, psets in STANDARD_PSETS.items()
        }

        self._build_hierarchy()

        # if a bSDD RDF snapshot was provided and the file actually exists, load it
        if snapshot_path and snapshot_path.exists():
            self._load_from_rdf(snapshot_path)
        elif snapshot_path:
            logger.warning(
                "bSDD snapshot not found at %s — using embedded schema only. "
                "Run `uv run rag-tag-refresh-ifc43-rdf` to download it.",
                snapshot_path,
            )

    def _build_hierarchy(self) -> None:
        """
        Walk the IFC schema using ifcopenshell and record the full ancestor
        chain for every class.

        ifcopenshell ships with the IFC schema built in, so we can ask it
        "what is the supertype of IfcWall?" without any network call.
        We walk up the chain until we hit None (which is IfcRoot's supertype).
        """
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
            """Walk up the supertype chain and collect all ancestor names."""
            ancestors: list[str] = []
            try:
                current = entity.supertype()
                while current is not None:
                    ancestors.append(current.name())
                    current = current.supertype()
            except Exception:
                pass  # some abstract types don't have supertypes
            return ancestors

        # First make sure every class we care about (from STANDARD_PSETS) is in there
        for class_name in STANDARD_PSETS:
            try:
                entity = schema.declaration_by_name(class_name)
                self._hierarchy[class_name] = _get_ancestor_chain(entity)
            except Exception:
                # class might not exist in this schema version — that's fine
                self._hierarchy.setdefault(class_name, [])

        # Then try to get all other IFC declarations too (subclasses like
        # IfcWallStandardCase). Not all ifcopenshell builds expose this.
        try:
            for decl in schema.declarations():
                name = decl.name()
                if name.startswith("Ifc") and name not in self._hierarchy:
                    self._hierarchy[name] = _get_ancestor_chain(decl)
        except (AttributeError, TypeError):
            pass  # older ifcopenshell builds don't have this — no problem

        logger.debug(
            "Built IFC hierarchy from schema '%s': %d classes indexed",
            self._schema_name,
            len(self._hierarchy),
        )

    def _load_from_rdf(self, path: Path) -> None:
        """
        Augment the hierarchy using a local bSDD / IFC-OWL Turtle file.

        The IFC OWL ontology uses rdfs:subClassOf to encode the class hierarchy.
        We parse those triples and add any parent relationships we don't already
        have from ifcopenshell.

        This is optional — if rdflib isn't installed or the file is bad,
        we just log a warning and carry on.
        """
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
            # look for triples like: IfcWallStandardCase rdfs:subClassOf IfcWall
            for cls_node, _, parent_node in g.triples((None, RDFS.subClassOf, None)):
                # URIs look like ".../IfcWall" — we just want the local name
                cls_name = str(cls_node).rsplit("/", 1)[-1].rsplit("#", 1)[-1]
                parent_name = str(parent_node).rsplit("/", 1)[-1].rsplit("#", 1)[-1]

                # only care about IFC classes
                if not (cls_name.startswith("Ifc") and parent_name.startswith("Ifc")):
                    continue

                chain = self._hierarchy.setdefault(cls_name, [])
                if parent_name not in chain:
                    # insert at front — it's the most direct parent
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

    # -------------------------------------------------------------------------
    # Public methods
    # -------------------------------------------------------------------------

    def normalize_class(self, raw_class: str) -> NormalizedClassResult:
        """
        Given a raw IFC class name, return its canonical form and full ancestry.

        Some IFC files use subclasses like IfcWallStandardCase instead of IfcWall.
        The standard Pset definitions are tied to IfcWall though, so we need to
        walk up the hierarchy to find the right one.

        If the class is totally unknown, we just return it as-is.

        Example:
            normalize_class("IfcWallStandardCase")
            → { raw: "IfcWallStandardCase", canonical: "IfcWall", base: "IfcWall", ... }
        """
        ancestors = self._hierarchy.get(raw_class, [])

        # if this class itself is in STANDARD_PSETS, it's already canonical
        canonical = raw_class
        if raw_class not in self._psets:
            # otherwise walk up the ancestor chain until we find a known one
            for ancestor in ancestors:
                if ancestor in self._psets:
                    canonical = ancestor
                    break

        return NormalizedClassResult(
            raw=raw_class,
            canonical=canonical,
            base=canonical,  # base and canonical are the same for now
            ancestors=ancestors,
        )

    def expected_properties_for_class(self, ifc_class: str) -> set[str]:
        """
        Return all "Pset.Property" strings that the spec says this class should have.

        We check the class itself AND all its ancestors, because a class can
        inherit Psets from its parent (though in practice the IFC spec defines
        them per-class, not via inheritance).

        The result is a set of strings like:
            {"Pset_WallCommon.FireRating", "Pset_WallCommon.IsExternal", ...}

        These are used in ifc_to_csv.py to always include these columns
        in the output, even if the model doesn't have the values set.
        """
        result: set[str] = set()

        # check this class and every ancestor
        classes_to_check = [ifc_class, *self._hierarchy.get(ifc_class, [])]
        for cls in classes_to_check:
            if cls in self._psets:
                for pset_name, props in self._psets[cls].items():
                    for prop in props:
                        result.add(f"{pset_name}.{prop}")

        return result

    def is_known_property(self, pset: str, prop: str) -> bool:
        """
        Check if a Pset+property combination is part of the IFC standard.

        Useful for separating "known standard properties" from custom/vendor
        properties that show up in some IFC files.  Unknown Psets are still
        extracted by ifc_to_csv.py — we just don't add empty columns for them.

        Example:
            is_known_property("Pset_WallCommon", "FireRating") → True
            is_known_property("Pset_VendorCustom", "SomeField") → False
        """
        for class_psets in self._psets.values():
            if pset in class_psets and prop in class_psets[pset]:
                return True
        return False


# ---------------------------------------------------------------------------
# Module-level cache so we only parse the schema once per process
# ---------------------------------------------------------------------------

# keyed by str(snapshot_path) so different paths get different instances
_registry_cache: dict[str | None, IFC43SchemaRegistry] = {}


def get_registry(
    snapshot_path: Path | None = None,
    schema_name: str | None = None,
) -> IFC43SchemaRegistry:
    """
    Get a cached registry instance.

    Parsing the ifcopenshell schema and (optionally) the RDF file takes a
    moment, so we cache the result and reuse it for every class we process
    in the same run.  Subsequent calls with the same path are instant.
    """
    key = str(snapshot_path) if snapshot_path is not None else None
    if key not in _registry_cache:
        _registry_cache[key] = IFC43SchemaRegistry(snapshot_path, schema_name)
    return _registry_cache[key]
