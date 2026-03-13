# One-pass IFC relationship index builder.
#
# Extracts explicit IFC semantic relationships from an open IFC model and
# returns a mapping of GlobalId -> RelationBlock.  Every extracted array is
# deduplicated and stable-sorted to guarantee deterministic JSONL output.
#
# Extracted relationships
# -----------------------
# hosts            – elements this element hosts
#                    (via IfcRelVoidsElement / IfcRelFillsElement)
# hosted_by        – elements that host this element
# ifc_connected_to – explicit IFC element connections
#                    (IfcRelConnectsElements subtypes +
#                     IfcRelConnectsStructuralElement for IFC2X3)
# belongs_to_system – system names from IfcRelAssignsToGroup + IfcSystem
# in_zone           – zone names from IfcRelAssignsToGroup + IfcZone
# classified_as     – classification labels from IfcRelAssociatesClassification
# typed_by          – type object GlobalIds from IfcRelDefinesByType

from __future__ import annotations

import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

# Type alias so callers can annotate against the exact shape.
RelationBlock = dict[str, list[str]]

# Canonical key ordering — used for empty blocks and for the sort step.
_BLOCK_KEYS: tuple[str, ...] = (
    "hosts",
    "hosted_by",
    "ifc_connected_to",
    "path_connected_to",
    "space_bounded_by",
    "bounds_space",
    "belongs_to_system",
    "in_zone",
    "classified_as",
    "typed_by",
)


def empty_relation_block() -> RelationBlock:
    """Return a fresh empty relation block (all arrays empty)."""
    return {k: [] for k in _BLOCK_KEYS}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _safe_gid(entity: object) -> str | None:
    """Extract GlobalId as a plain string; silently return None on any failure."""
    try:
        gid = getattr(entity, "GlobalId", None)
        return str(gid) if gid else None
    except Exception:
        return None


def _stable(lst: list[str]) -> list[str]:
    """Deduplicate and stable-sort a list of strings."""
    return sorted(set(lst))


def _classification_label(ref_or_cls: object) -> str | None:
    """
    Build a stable, human-readable label from an IfcClassificationReference
    or IfcClassification entity.

    Format for IfcClassificationReference:
        ``<SourceName>:<Identification>:<RefName>``
        (empty parts are omitted; parts joined with ``:``)

    Format for IfcClassification:
        ``<Name>``

    Returns None when no useful label can be extracted.
    """
    try:
        ifc_type: str = ref_or_cls.is_a()  # type: ignore[union-attr]
    except Exception:
        return None

    if ifc_type == "IfcClassificationReference":
        # IFC4+ uses Identification; IFC2X3 uses ItemReference
        ident: str | None = None
        for attr in ("Identification", "ItemReference"):
            try:
                val = getattr(ref_or_cls, attr, None)
                if val:
                    ident = str(val)
                    break
            except Exception:
                pass

        name: str | None = None
        try:
            n = getattr(ref_or_cls, "Name", None)
            name = str(n) if n else None
        except Exception:
            pass

        source_name: str | None = None
        try:
            src = getattr(ref_or_cls, "ReferencedSource", None)
            if src is not None:
                sn = getattr(src, "Name", None)
                source_name = str(sn) if sn else None
        except Exception:
            pass

        parts = [p for p in (source_name, ident, name) if p]
        return ":".join(parts) if parts else None

    if ifc_type == "IfcClassification":
        try:
            n = getattr(ref_or_cls, "Name", None)
            return str(n) if n else None
        except Exception:
            return None

    return None


def _group_label(group: object) -> str | None:
    """
    Extract a stable label for an IfcGroup / IfcSystem / IfcZone.

    Prefers the group's Name attribute; falls back to its GlobalId so the
    label is never None for a valid group entity.
    """
    try:
        name = getattr(group, "Name", None)
        if name:
            return str(name)
        gid = getattr(group, "GlobalId", None)
        return str(gid) if gid else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Per-relationship-type index passes
# ---------------------------------------------------------------------------


def _index_voids(model: object, raw: dict) -> None:
    """
    IfcRelVoidsElement: RelatingBuildingElement hosts RelatedOpeningElement.

    The parent element gets the opening's GlobalId added to its ``hosts``
    array; the opening gets the parent's GlobalId in its ``hosted_by`` array.
    """
    try:
        for rel in model.by_type("IfcRelVoidsElement"):  # type: ignore[union-attr]
            try:
                parent = rel.RelatingBuildingElement
                opening = rel.RelatedOpeningElement
                if parent is None or opening is None:
                    continue
                pgid = _safe_gid(parent)
                ogid = _safe_gid(opening)
                if pgid and ogid:
                    raw[pgid]["hosts"].append(ogid)
                    raw[ogid]["hosted_by"].append(pgid)
            except Exception as exc:
                logger.debug("IfcRelVoidsElement record skipped: %s", exc)
    except Exception as exc:
        logger.debug("IfcRelVoidsElement iteration skipped: %s", exc)


def _index_fills(model: object, raw: dict) -> None:
    """
    IfcRelFillsElement: RelatingOpeningElement hosts RelatedBuildingElement.

    Doors and windows are the typical 'fillings' of openings.
    """
    try:
        for rel in model.by_type("IfcRelFillsElement"):  # type: ignore[union-attr]
            try:
                opening = rel.RelatingOpeningElement
                filling = rel.RelatedBuildingElement
                if opening is None or filling is None:
                    continue
                ogid = _safe_gid(opening)
                fgid = _safe_gid(filling)
                if ogid and fgid:
                    raw[ogid]["hosts"].append(fgid)
                    raw[fgid]["hosted_by"].append(ogid)
            except Exception as exc:
                logger.debug("IfcRelFillsElement record skipped: %s", exc)
    except Exception as exc:
        logger.debug("IfcRelFillsElement iteration skipped: %s", exc)


def _index_connects_elements(model: object, raw: dict) -> None:
    """
    Explicit element connections → ``ifc_connected_to`` (bidirectional).

    Covers:
    - IfcRelConnectsElements (and all its subtypes, e.g.
      IfcRelConnectsPathElements, IfcRelConnectsWithRealizingElements,
      IfcRelConnectsPortToElement) — handled via ifcopenshell's
      by_type() inheritance support.
    - IfcRelConnectsStructuralElement (IFC2X3 / some IFC4 structural
      models): uses RelatingElement + RelatedStructuralMember.

    Self-loops (same GlobalId on both sides) are silently dropped.
    """
    for rel_class in (
        "IfcRelConnectsElements",
        "IfcRelConnectsStructuralElement",
    ):
        try:
            for rel in model.by_type(rel_class):  # type: ignore[union-attr]
                try:
                    a = getattr(rel, "RelatingElement", None)
                    b = getattr(rel, "RelatedElement", None)
                    # IFC2X3 structural variant uses a different attribute name
                    if b is None:
                        b = getattr(rel, "RelatedStructuralMember", None)
                    if a is None or b is None:
                        continue
                    agid = _safe_gid(a)
                    bgid = _safe_gid(b)
                    if agid and bgid and agid != bgid:
                        raw[agid]["ifc_connected_to"].append(bgid)
                        raw[bgid]["ifc_connected_to"].append(agid)
                except Exception as exc:
                    logger.debug("%s record skipped: %s", rel_class, exc)
        except Exception as exc:
            logger.debug("%s iteration skipped: %s", rel_class, exc)


def _index_connects_path_elements(model: object, raw: dict) -> None:
    """IfcRelConnectsPathElements -> ``path_connected_to`` (bidirectional)."""
    try:
        for rel in model.by_type("IfcRelConnectsPathElements"):  # type: ignore[union-attr]
            try:
                a = getattr(rel, "RelatingElement", None)
                b = getattr(rel, "RelatedElement", None)
                if a is None or b is None:
                    continue
                agid = _safe_gid(a)
                bgid = _safe_gid(b)
                if agid and bgid and agid != bgid:
                    raw[agid]["path_connected_to"].append(bgid)
                    raw[bgid]["path_connected_to"].append(agid)
            except Exception as exc:
                logger.debug("IfcRelConnectsPathElements record skipped: %s", exc)
    except Exception as exc:
        logger.debug("IfcRelConnectsPathElements iteration skipped: %s", exc)


def _index_space_boundaries(model: object, raw: dict) -> None:
    """IfcRelSpaceBoundary -> ``space_bounded_by`` / ``bounds_space``."""
    try:
        for rel in model.by_type("IfcRelSpaceBoundary"):  # type: ignore[union-attr]
            try:
                space = getattr(rel, "RelatingSpace", None)
                element = getattr(rel, "RelatedBuildingElement", None)
                if space is None or element is None:
                    continue
                sgid = _safe_gid(space)
                egid = _safe_gid(element)
                if sgid and egid and sgid != egid:
                    raw[sgid]["space_bounded_by"].append(egid)
                    raw[egid]["bounds_space"].append(sgid)
            except Exception as exc:
                logger.debug("IfcRelSpaceBoundary record skipped: %s", exc)
    except Exception as exc:
        logger.debug("IfcRelSpaceBoundary iteration skipped: %s", exc)


def _index_groups(model: object, raw: dict) -> None:
    """
    IfcRelAssignsToGroup → ``belongs_to_system`` or ``in_zone``.

    In IFC4 / IFC4X3, IfcZone is a *subtype* of IfcSystem.  We therefore
    check for IfcZone first (more specific) so that zone membership is
    not misclassified as system membership.

    Generic IfcGroup instances (neither system nor zone) are ignored.
    """
    try:
        for rel in model.by_type("IfcRelAssignsToGroup"):  # type: ignore[union-attr]
            try:
                group = rel.RelatingGroup
                if group is None:
                    continue
                label = _group_label(group)
                if not label:
                    continue

                # IfcZone IS-A IfcSystem in IFC4/IFC4X3 — check Zone first.
                if group.is_a("IfcZone"):
                    key = "in_zone"
                elif group.is_a("IfcSystem"):
                    key = "belongs_to_system"
                else:
                    continue

                for obj in getattr(rel, "RelatedObjects", None) or []:
                    try:
                        gid = _safe_gid(obj)
                        if gid:
                            raw[gid][key].append(label)
                    except Exception as exc:
                        logger.debug("Group member skipped: %s", exc)
            except Exception as exc:
                logger.debug("IfcRelAssignsToGroup record skipped: %s", exc)
    except Exception as exc:
        logger.debug("IfcRelAssignsToGroup iteration skipped: %s", exc)


def _index_classifications(model: object, raw: dict) -> None:
    """
    IfcRelAssociatesClassification → ``classified_as``.

    The label format is ``<SourceName>:<Identification>:<RefName>``
    (parts joined only when non-empty) for IfcClassificationReference,
    or ``<Name>`` for a bare IfcClassification.
    """
    try:
        for rel in model.by_type(  # type: ignore[union-attr]
            "IfcRelAssociatesClassification"
        ):
            try:
                cls_ref = rel.RelatingClassification
                if cls_ref is None:
                    continue
                label = _classification_label(cls_ref)
                if not label:
                    continue
                for obj in getattr(rel, "RelatedObjects", None) or []:
                    try:
                        gid = _safe_gid(obj)
                        if gid:
                            raw[gid]["classified_as"].append(label)
                    except Exception as exc:
                        logger.debug("Classification member skipped: %s", exc)
            except Exception as exc:
                logger.debug("IfcRelAssociatesClassification record skipped: %s", exc)
    except Exception as exc:
        logger.debug("IfcRelAssociatesClassification iteration skipped: %s", exc)


def _index_type_definitions(model: object, raw: dict) -> None:
    """IfcRelDefinesByType -> ``typed_by`` from occurrence to type GlobalId."""
    try:
        for rel in model.by_type("IfcRelDefinesByType"):  # type: ignore[union-attr]
            try:
                relating_type = getattr(rel, "RelatingType", None)
                if relating_type is None:
                    continue
                type_gid = _safe_gid(relating_type)
                if not type_gid:
                    continue
                for obj in getattr(rel, "RelatedObjects", None) or []:
                    gid = _safe_gid(obj)
                    if gid and gid != type_gid:
                        raw[gid]["typed_by"].append(type_gid)
            except Exception as exc:
                logger.debug("IfcRelDefinesByType record skipped: %s", exc)
    except Exception as exc:
        logger.debug("IfcRelDefinesByType iteration skipped: %s", exc)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def build_relationship_index(model: object) -> dict[str, RelationBlock]:
    """
    Build a one-pass relationship index over an open IFC model.

    All five relationship passes run in sequence; each pass accumulates raw
    (possibly duplicate) entries into a working dict keyed by GlobalId.
    After all passes the dict is frozen by deduplicating and stable-sorting
    every array, which guarantees deterministic output across reruns.

    Parameters
    ----------
    model:
        An already-open ``ifcopenshell.file`` instance.

    Returns
    -------
    dict[str, RelationBlock]
        Mapping ``GlobalId -> RelationBlock``.  Only elements touched by at
        least one relationship appear as keys.  Use::

            index.get(gid, empty_relation_block())

        for safe per-element access.
    """
    raw: dict[str, dict[str, list[str]]] = defaultdict(empty_relation_block)

    _index_voids(model, raw)
    _index_fills(model, raw)
    _index_connects_elements(model, raw)
    _index_connects_path_elements(model, raw)
    _index_space_boundaries(model, raw)
    _index_groups(model, raw)
    _index_classifications(model, raw)
    _index_type_definitions(model, raw)

    # Deduplicate and stable-sort every array for determinism.
    result: dict[str, RelationBlock] = {
        gid: {k: _stable(v) for k, v in block.items()} for gid, block in raw.items()
    }

    logger.info(
        "Relationship index built: %d element(s) with at least one relation.",
        len(result),
    )
    return result
