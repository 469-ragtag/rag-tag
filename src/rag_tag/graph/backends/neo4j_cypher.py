"""Cypher snippets for Neo4j graph backend."""

from __future__ import annotations

CREATE_CONSTRAINT = (
    "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Node) REQUIRE n.node_id IS UNIQUE"
)

CLEAR_DB = "MATCH (n) DETACH DELETE n"

INSERT_NODES = """
UNWIND $rows AS row
MERGE (n:Node {node_id: row.node_id})
SET n.label = row.label,
    n.class_ = row.class_,
    n.node_kind = row.node_kind,
    n.dataset = row.dataset,
    // Full properties/payload are stored as JSON strings (Neo4j disallows maps).
    n.properties_json = row.properties_json,
    n.payload_json = row.payload_json,
    n.geometry_json = row.geometry_json,
    n.global_id = row.global_id,
    n.express_id = row.express_id,
    n.name = row.name,
    n.class_raw = row.class_raw,
    n.ifc_type = row.ifc_type,
    n.level = row.level,
    n.type_name = row.type_name,
    n.predefined_type = row.predefined_type,
    n.object_type = row.object_type,
    n.tag = row.tag,
    n.description = row.description
"""

INSERT_RELS = """
UNWIND $rows AS row
MATCH (a:Node {node_id: row.from})
MATCH (b:Node {node_id: row.to})
CREATE (a)-[r:REL]->(b)
SET r.relation = row.relation,
    r.source = row.source,
    r.dataset = row.dataset,
    r.distance = row.distance,
    r.vertical_gap = row.vertical_gap,
    r.overlap_area_xy = row.overlap_area_xy,
    r.intersection_volume = row.intersection_volume,
    r.contact_area = row.contact_area,
    // Preserve parallel edge ordering deterministically.
    r.edge_index = row.edge_index
"""

MATCH_NODE_BY_ID = """
MATCH (n:Node {node_id: $node_id})
WHERE size($datasets) = 0 OR n.dataset IN $datasets
RETURN n
"""

MATCH_STOREY_BY_ID = """
MATCH (n:Node {node_id: $node_id})
WHERE toLower(n.class_) = 'ifcbuildingstorey'
  AND (size($datasets) = 0 OR n.dataset IN $datasets)
RETURN n
"""

MATCH_STOREY_BY_LABEL = """
MATCH (n:Node)
WHERE toLower(n.class_) = 'ifcbuildingstorey'
  AND toLower(n.label) = toLower($label)
  AND (size($datasets) = 0 OR n.dataset IN $datasets)
RETURN n
"""

MATCH_DESCENDANTS_CONTAINS = """
MATCH (s:Node {node_id: $node_id})
WHERE size($datasets) = 0 OR s.dataset IN $datasets
MATCH p=(s)-[:REL*1..]->(n:Node)
WHERE all(r in relationships(p) WHERE r.relation = 'contains')
  AND (
    size($datasets) = 0
    OR all(node in nodes(p) WHERE node.dataset IN $datasets)
  )
RETURN DISTINCT n
"""

MATCH_CLASS = """
MATCH (n:Node)
WHERE toLower(n.class_) = toLower($class_name)
  AND (size($datasets) = 0 OR n.dataset IN $datasets)
RETURN n
"""

MATCH_SPATIAL_NEIGHBORS = """
MATCH (n:Node {node_id: $node_id})-[r:REL]-(m:Node)
WHERE r.relation IN $relations
  AND (size($datasets) = 0 OR n.dataset IN $datasets)
  AND (size($datasets) = 0 OR m.dataset IN $datasets)
  AND (size($datasets) = 0 OR r.dataset IN $datasets)
RETURN m, r
ORDER BY r.edge_index
"""

MATCH_TOPOLOGY_NEIGHBORS = """
MATCH (n:Node {node_id: $node_id})-[r:REL]-(m:Node)
WHERE r.relation IN $relations
  AND (size($datasets) = 0 OR n.dataset IN $datasets)
  AND (size($datasets) = 0 OR m.dataset IN $datasets)
  AND (size($datasets) = 0 OR r.dataset IN $datasets)
RETURN m, r
ORDER BY r.edge_index
"""

MATCH_OUTGOING_RELATIONS = """
MATCH (n:Node)-[r:REL]->(m:Node)
WHERE n.node_id IN $frontier AND r.relation IN $relations
  AND (size($datasets) = 0 OR n.dataset IN $datasets)
  AND (size($datasets) = 0 OR m.dataset IN $datasets)
  AND (size($datasets) = 0 OR r.dataset IN $datasets)
RETURN n.node_id AS from_id, m, r
ORDER BY r.edge_index
"""

MATCH_OUTGOING_RELATION_EXACT = """
MATCH (n:Node)-[r:REL]->(m:Node)
WHERE n.node_id IN $frontier AND r.relation = $relation
  AND (size($datasets) = 0 OR n.dataset IN $datasets)
  AND (size($datasets) = 0 OR m.dataset IN $datasets)
  AND (size($datasets) = 0 OR r.dataset IN $datasets)
RETURN n.node_id AS from_id, m, r
ORDER BY r.edge_index
"""

MATCH_BY_NODE_IDS = """
MATCH (n:Node)
WHERE n.node_id IN $ids
  AND (size($datasets) = 0 OR n.dataset IN $datasets)
RETURN n
"""

MATCH_ALL_NODES = """
MATCH (n:Node)
WHERE size($datasets) = 0 OR n.dataset IN $datasets
RETURN n
"""

MATCH_ALL_RELS = """
MATCH (a:Node)-[r:REL]->(b:Node)
WHERE (size($datasets) = 0 OR a.dataset IN $datasets)
  AND (size($datasets) = 0 OR b.dataset IN $datasets)
  AND (size($datasets) = 0 OR r.dataset IN $datasets)
RETURN a.node_id AS from_id, b.node_id AS to_id, r
ORDER BY r.edge_index
"""
