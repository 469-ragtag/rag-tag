"""IFC 4.3 schema registry loaded from a local bSDD RDF snapshot."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

from rdflib import BNode, Graph, Literal, URIRef
from rdflib.namespace import RDFS

logger = logging.getLogger(__name__)

_IFC_CLASS_RE = re.compile(r"^Ifc[A-Za-z0-9_]+$")
_PSET_RE = re.compile(r"^(Pset|Qto)_[A-Za-z0-9_]+$")
_PROPERTY_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]+$")

_RDF_SUFFIX_TO_FORMAT = {
    ".ttl": "turtle",
    ".rdf": "xml",
    ".owl": "xml",
    ".nt": "nt",
    ".jsonld": "json-ld",
}

_BASE_CLASS_PRIORITY = (
    "IfcBuildingElement",
    "IfcDistributionElement",
    "IfcSpatialElement",
    "IfcElement",
    "IfcProduct",
    "IfcObject",
    "IfcRoot",
)


@dataclass(frozen=True)
class NormalizedClassResult:
    """Normalization result for an IFC class."""

    raw_class: str
    canonical_class: str
    base_class: str
    ancestors: tuple[str, ...]


class IFC43SchemaRegistry:
    """Provides class hierarchy and Pset/property definitions from RDF."""

    def __init__(self, rdf_path: Path | None = None) -> None:
        self._rdf_path = rdf_path
        self._enabled = rdf_path is not None and rdf_path.is_file()
        self._graph: Graph | None = None

        self._known_classes: set[str] = set()
        self._lower_to_class: dict[str, str] = {}
        self._parent_map: dict[str, set[str]] = {}
        self._ancestor_cache: dict[str, set[str]] = {}

        self._pset_properties: dict[str, set[str]] = {}
        self._class_psets: dict[str, set[str]] = {}

        if not self._enabled:
            if rdf_path:
                logger.warning(
                    "bSDD IFC 4.3 RDF snapshot not found at %s. "
                    "Using legacy extraction mode.",
                    rdf_path,
                )
            return

        self._graph = self._load_graph(rdf_path)
        if self._graph is None:
            self._enabled = False
            return

        self._build_class_hierarchy()
        self._build_pset_property_definitions()

        logger.info(
            "Loaded IFC43 registry from %s (%d classes, %d psets, %d class-pset links)",
            rdf_path,
            len(self._known_classes),
            len(self._pset_properties),
            sum(len(psets) for psets in self._class_psets.values()),
        )

    @property
    def enabled(self) -> bool:
        return self._enabled

    def normalize_class(self, raw_class: str) -> NormalizedClassResult:
        """Normalize class naming and derive a stable base class."""
        canonical = self._canonicalize_class_name(raw_class)
        ancestors = sorted(self._ancestors_of(canonical))
        base_class = self._pick_base_class(canonical, ancestors)
        return NormalizedClassResult(
            raw_class=raw_class,
            canonical_class=canonical,
            base_class=base_class,
            ancestors=tuple(ancestors),
        )

    def expected_properties_for_class(self, raw_or_normalized_class: str) -> set[str]:
        """Return expected Pset.Property keys for the class and its ancestors."""
        if not self._enabled:
            return set()

        canonical = self._canonicalize_class_name(raw_or_normalized_class)
        lineage = self._ancestors_of(canonical) | {canonical}

        keys: set[str] = set()
        for cls_name in lineage:
            for pset_name in self._class_psets.get(cls_name, set()):
                for prop_name in self._pset_properties.get(pset_name, set()):
                    keys.add(f"{pset_name}.{prop_name}")
        return keys

    def is_known_property(self, pset_name: str, prop_name: str) -> bool:
        return prop_name in self._pset_properties.get(pset_name, set())

    def _load_graph(self, rdf_path: Path) -> Graph | None:
        graph = Graph()
        rdf_format = _RDF_SUFFIX_TO_FORMAT.get(rdf_path.suffix.lower())
        try:
            graph.parse(str(rdf_path), format=rdf_format)
            return graph
        except Exception as exc:
            logger.warning(
                "Failed to parse bSDD IFC 4.3 RDF snapshot at %s: %s. "
                "Using legacy extraction mode.",
                rdf_path,
                exc,
            )
            return None

    def _build_class_hierarchy(self) -> None:
        if self._graph is None:
            return

        for child, _, parent in self._graph.triples((None, RDFS.subClassOf, None)):
            child_name = self._extract_ifc_class_name(child)
            parent_name = self._extract_ifc_class_name(parent)
            if not child_name or not parent_name:
                continue
            self._known_classes.add(child_name)
            self._known_classes.add(parent_name)
            self._parent_map.setdefault(child_name, set()).add(parent_name)

        for cls in self._known_classes:
            self._lower_to_class[cls.lower()] = cls

    def _build_pset_property_definitions(self) -> None:
        if self._graph is None:
            return

        class_nodes: dict[URIRef | BNode | Literal, str] = {}
        pset_nodes: dict[URIRef | BNode | Literal, str] = {}
        bnode_classes: dict[BNode, set[str]] = {}
        bnode_psets: dict[BNode, set[str]] = {}

        for subj, pred, obj in self._graph:
            subj_class = self._extract_ifc_class_name(subj)
            obj_class = self._extract_ifc_class_name(obj)
            subj_pset = self._extract_pset_name(subj)
            obj_pset = self._extract_pset_name(obj)

            if subj_class:
                class_nodes[subj] = subj_class
            if obj_class:
                class_nodes[obj] = obj_class
            if subj_pset:
                pset_nodes[subj] = subj_pset
            if obj_pset:
                pset_nodes[obj] = obj_pset

            if isinstance(subj, BNode) and obj_class:
                bnode_classes.setdefault(subj, set()).add(obj_class)
            if isinstance(obj, BNode) and subj_class:
                bnode_classes.setdefault(obj, set()).add(subj_class)
            if isinstance(subj, BNode) and obj_pset:
                bnode_psets.setdefault(subj, set()).add(obj_pset)
            if isinstance(obj, BNode) and subj_pset:
                bnode_psets.setdefault(obj, set()).add(subj_pset)

            if subj_class and obj_pset:
                self._class_psets.setdefault(subj_class, set()).add(obj_pset)
            if obj_class and subj_pset:
                self._class_psets.setdefault(obj_class, set()).add(subj_pset)

            if subj_pset:
                self._collect_pset_property_candidate(subj_pset, pred)
                self._collect_pset_property_candidate(subj_pset, obj)
            if obj_pset:
                self._collect_pset_property_candidate(obj_pset, pred)
                self._collect_pset_property_candidate(obj_pset, subj)

        for bnode, classes in bnode_classes.items():
            psets = bnode_psets.get(bnode, set())
            for cls_name in classes:
                self._class_psets.setdefault(cls_name, set()).update(psets)

    def _collect_pset_property_candidate(
        self, pset_name: str, term: URIRef | BNode | Literal
    ) -> None:
        prop_name = self._extract_property_name(term)
        if not prop_name:
            return
        self._pset_properties.setdefault(pset_name, set()).add(prop_name)

    def _canonicalize_class_name(self, class_name: str) -> str:
        if class_name in self._known_classes:
            return class_name

        normalized = class_name.strip()
        if normalized.lower().startswith("ifc"):
            candidate = "Ifc" + normalized[3:]
        else:
            candidate = "Ifc" + normalized

        mapped = self._lower_to_class.get(candidate.lower())
        return mapped or candidate

    def _ancestors_of(self, class_name: str) -> set[str]:
        if class_name in self._ancestor_cache:
            return set(self._ancestor_cache[class_name])

        parents = self._parent_map.get(class_name, set())
        ancestors: set[str] = set(parents)
        for parent in parents:
            ancestors.update(self._ancestors_of(parent))

        self._ancestor_cache[class_name] = set(ancestors)
        return ancestors

    def _pick_base_class(self, canonical: str, ancestors: list[str]) -> str:
        lineage = set(ancestors)
        lineage.add(canonical)
        for candidate in _BASE_CLASS_PRIORITY:
            if candidate in lineage:
                return candidate
        return canonical

    def _extract_ifc_class_name(self, term: URIRef | BNode | Literal) -> str | None:
        token = self._extract_token(term)
        if token and _IFC_CLASS_RE.match(token):
            return token
        return None

    def _extract_pset_name(self, term: URIRef | BNode | Literal) -> str | None:
        token = self._extract_token(term)
        if token and _PSET_RE.match(token):
            return token
        return None

    def _extract_property_name(self, term: URIRef | BNode | Literal) -> str | None:
        token = self._extract_token(term)
        if not token:
            return None
        if _IFC_CLASS_RE.match(token) or _PSET_RE.match(token):
            return None
        if token.startswith("http") or token.startswith("urn"):
            return None
        if token in {
            "Class",
            "PropertySet",
            "Property",
            "Quantity",
            "type",
            "subClassOf",
            "label",
            "comment",
            "domain",
            "range",
            "hasPropertySet",
            "hasProperty",
            "applicableClass",
            "relatedProperty",
        }:
            return None
        if _PROPERTY_RE.match(token):
            return token
        return None

    def _extract_token(self, term: URIRef | BNode | Literal) -> str | None:
        if isinstance(term, URIRef):
            value = str(term).rstrip("/")
            token = value.rsplit("#", 1)[-1].rsplit("/", 1)[-1]
            return token if token else None
        if isinstance(term, Literal):
            text = str(term).strip()
            if text and len(text) <= 80 and " " not in text:
                return text
        return None
