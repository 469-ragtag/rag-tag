# IfcOpenShell Python Skill for Graph-RAG IFC Parser

## Overview

IfcOpenShell is an open-source (LGPL 3) library for working with Industry Foundation Classes (IFC) files - the ISO standard for BIM data exchange. This skill focuses on using IfcOpenShell Python API to build robust parsers and graph representations for spatial/topological queries in Graph-RAG systems.

**Latest Version**: 0.8.4 (as of February 2026)  
**Supported IFC Schemas**: IFC2X3 TC1, IFC4 Add2 TC1, IFC4x1, IFC4x2, IFC4x3 Add2  
**Python Compatibility**: Python 3.7+

---

## Core Architecture

### Three-Layer System

IfcOpenShell provides three primary interaction layers:

1. **Core Parser** (`ifcopenshell.file`): Low-level file operations, entity queries
2. **Utility Functions** (`ifcopenshell.util.*`): High-level data extraction helpers
3. **API Layer** (`ifcopenshell.api.*`): High-level authoring and editing functions

For **parsing and graph building**, focus on layers 1 and 2.

---

## Installation & Setup

```python
# Install via pip
pip install ifcopenshell --break-system-packages  # Required on some systems

# Basic import pattern
import ifcopenshell
import ifcopenshell.util.element
import ifcopenshell.util.placement
import ifcopenshell.util.unit
import ifcopenshell.geom

# Open IFC file
model = ifcopenshell.open('path/to/model.ifc')

# Check schema version
print(model.schema)  # e.g., 'IFC4'
print(model.schema_version)  # e.g., (4, 0, 2, 1)
```

---

## Core Parsing Patterns

### 1. Entity Retrieval

```python
# By type - returns list of entity_instance objects
walls = model.by_type('IfcWall')
all_products = model.by_type('IfcProduct')

# By ID (STEP ID)
element = model.by_id(123)

# By GUID (IFC GlobalId)
element = model.by_guid('2MLFd4X2f0jRq28Dvww1Vm')

# Subscripting (equivalent to above)
element = model[123]  # By ID
element = model['2MLFd4X2f0jRq28Dvww1Vm']  # By GUID

# Check type hierarchy
wall.is_a('IfcWall')  # True
wall.is_a('IfcBuildingElement')  # True
wall.is_a('IfcProduct')  # True
```

**CRITICAL**: `by_type()` only returns direct instances of that class. It does NOT include subclasses by default. For example, `by_type('IfcElement')` will NOT return `IfcWall` instances.

### 2. Attribute Access

```python
# Direct attribute access
print(wall.Name)
print(wall.GlobalId)
print(wall.Description)

# Check for optional attributes
if hasattr(wall, 'Name'):
    print(wall.Name)

# Positional access (attributes have fixed order per IFC schema)
print(wall[0])  # First attribute (usually GlobalId)
print(wall[2])  # Third attribute (usually Name)

# Get attribute metadata
print(wall.attribute_name(0))  # 'GlobalId'
print(wall.attribute_type(0))  # 'STRING'
```

### 3. Comprehensive Entity Info

```python
# Get all attributes as dictionary
info = element.get_info()
# Returns: {'id', 'type', 'GlobalId', 'Name', 'Description', ...}

# Recursive retrieval (includes referenced entities)
info = element.get_info(recursive=True)

# Exclude STEP ID
info = element.get_info(include_identifier=False)

# Ignore specific attributes
info = element.get_info(ignore=['OwnerHistory'])
```

---

## Relationship Handling

IFC uses **objectified relationships** - relationships are entities themselves (IfcRel\* classes).

### Understanding Inverse Attributes

Inverse attributes are NOT stored in the entity itself - they're computed by following relationships:

```python
# Forward relationship (direct attribute)
wall.OwnerHistory  # Returns IfcOwnerHistory entity

# Inverse relationship (computed)
wall.IsDefinedBy  # Returns tuple of IfcRelDefinesByProperties relationships

# Common inverse attributes:
# - IsDefinedBy: Property sets
# - ContainedInStructure: Spatial container
# - ConnectedTo / ConnectedFrom: Physical connections
# - FillsVoids / HasOpenings: Voids and openings
# - HasAssignments: Group/system assignments
# - Decomposes / IsDecomposedBy: Aggregation hierarchy
```

### Key Relationship Types

#### 1. Spatial Containment (IfcRelContainedInSpatialStructure)

```python
import ifcopenshell.util.element as util_element

# Get spatial container (Building Storey, Space, etc.)
container = util_element.get_container(wall)

# Get all elements in a container
storey = model.by_type('IfcBuildingStorey')[0]
elements_in_storey = []
for rel in storey.ContainsElements or []:
    elements_in_storey.extend(rel.RelatedElements)
```

**IMPORTANT**: Elements can have both primary and secondary containment. Windows/walls spanning multiple storeys use primary containment for the lowest storey and references for others.

#### 2. Property Sets (IfcRelDefinesByProperties)

```python
# HIGH-LEVEL METHOD (RECOMMENDED)
psets = util_element.get_psets(wall)
# Returns: {'Pset_WallCommon': {'id': 123, 'FireRating': '2HR', ...}, ...}

# Get single pset
pset = util_element.get_pset(wall, 'Pset_WallCommon')

# Inherit from type
psets = util_element.get_psets(wall, should_inherit=True)

# LOW-LEVEL METHOD
for definition in wall.IsDefinedBy:
    related_data = definition.RelatingPropertyDefinition
    if related_data.is_a('IfcPropertySet'):
        print(f"Property Set: {related_data.Name}")
        for prop in related_data.HasProperties:
            if prop.is_a('IfcPropertySingleValue'):
                print(f"  {prop.Name}: {prop.NominalValue.wrappedValue}")
```

#### 3. Type Relationships (IfcRelDefinesByType)

```python
# Get type object
wall_type = util_element.get_type(wall)

# Get all occurrences of a type
wall_type = model.by_type('IfcWallType')[0]
occurrences = util_element.get_types(wall_type)
```

#### 4. Material Associations (IfcRelAssociatesMaterial)

```python
# Get material
material = util_element.get_material(wall)

# Material can be:
# - IfcMaterial (single material)
# - IfcMaterialLayerSet (layered, e.g., walls)
# - IfcMaterialConstituentSet (constituents)
# - IfcMaterialProfileSet (profiles, e.g., beams)
```

#### 5. Decomposition (IfcRelAggregates / IfcRelNests)

```python
# Get parts of an aggregate
components = util_element.get_components(element)

# Get aggregate parent
aggregate = util_element.get_aggregate(element)

# Get full decomposition tree
decomposition = util_element.get_decomposition(element, is_recursive=True)
```

---

## Spatial Geometry & Placement

### 1. Object Placement Matrices

```python
import ifcopenshell.util.placement as util_placement

# Get 4x4 transformation matrix (global coordinates)
matrix = util_placement.get_local_placement(wall.ObjectPlacement)
# Returns numpy array:
# [[ x_x, y_x, z_x, x ],
#  [ x_y, y_y, z_y, y ],
#  [ x_z, y_z, z_z, z ],
#  [ 0.0, 0.0, 0.0, 1 ]]

# Extract position
position = matrix[:3, 3]  # [x, y, z]

# Extract rotation
rotation_matrix = matrix[:3, :3]

# Get storey elevation
elevation = util_placement.get_storey_elevation(storey)
```

**CRITICAL**: Placement is hierarchical. Element placement is relative to its parent spatial structure. Use `get_local_placement()` which handles the chain automatically.

### 2. Geometry Extraction

```python
import ifcopenshell.geom

# Create geometry settings
settings = ifcopenshell.geom.settings()
settings.set('deflection-tolerance', 1e-3)  # Mesh quality
settings.set('use-world-coords', True)  # Global vs local coordinates
settings.set('apply-default-materials', True)

# Extract single shape
shape = ifcopenshell.geom.create_shape(settings, wall)

# Access geometry data
faces = shape.geometry.faces  # Triangle indices: [f1v1, f1v2, f1v3, ...]
verts = shape.geometry.verts  # Vertex coords: [v1x, v1y, v1z, v2x, ...]
materials = shape.geometry.materials  # Material info
material_ids = shape.geometry.material_ids  # Material per face

# Reshape for easier processing
vertices = [[verts[i], verts[i+1], verts[i+2]]
            for i in range(0, len(verts), 3)]
triangles = [[faces[i], faces[i+1], faces[i+2]]
             for i in range(0, len(faces), 3)]
```

### 3. Batch Geometry Processing (Faster)

```python
import multiprocessing

# Use iterator for large models
settings = ifcopenshell.geom.settings()
iterator = ifcopenshell.geom.iterator(
    settings,
    model,
    num_threads=multiprocessing.cpu_count()
)

if iterator.initialize():
    while True:
        shape = iterator.get()
        element = model.by_guid(shape.guid)
        # Process geometry...
        if not iterator.next():
            break
```

### 4. Filter Geometry by Type

```python
# Only process specific types
settings = ifcopenshell.geom.settings()
iterator = ifcopenshell.geom.iterator(
    settings,
    model,
    include=['IfcWall', 'IfcSlab', 'IfcColumn']
)

# Exclude types
iterator = ifcopenshell.geom.iterator(
    settings,
    model,
    exclude=['IfcOpeningElement', 'IfcSpace']
)
```

---

## Advanced Spatial Queries

### 1. Geometry Trees for Spatial Analysis

IfcOpenShell provides two tree types for efficient spatial queries:

```python
import ifcopenshell.geom

# Setup tree settings
tree_settings = ifcopenshell.geom.settings()
tree_settings.set('DISABLE_OPENING_SUBTRACTIONS', True)  # Faster for spaces
tree_settings.set('DISABLE_TRIANGULATION', True)  # Use native geometry

# Build tree
iterator = ifcopenshell.geom.iterator(
    tree_settings,
    model,
    include=['IfcSpace']
)
tree = ifcopenshell.geom.tree()
tree.add_iterator(iterator)

# Point query - find space containing point
location = (10.0, 5.0, 2.0)  # x, y, z in meters
spaces = tree.select(location)

# Extended search (with tolerance)
spaces = tree.select(location, extend=0.5)  # 0.5m radius

# Box query - find elements intersecting bounding box
elements = tree.select_box(wall)

# Elements completely within box
elements = tree.select_box(wall, completely_within=True)

# Extend bounding box
elements = tree.select_box(wall, extend=5.0)

# Ray casting
ray_origin = (0, 0, 0)
ray_direction = (1, 0, 0)
hits = tree.select_ray(ray_origin, ray_direction)
```

### 2. Clash Detection

```python
# Setup for clash detection
tree = ifcopenshell.geom.tree()
tree.add_iterator(iterator)

# Define element groups
group_a = model.by_type('IfcWall')
group_b = model.by_type('IfcDuct')

# Intersection clashes
clashes = tree.clash_intersection_many(group_a, group_b)

# Clearance clashes (elements too close)
clashes = tree.clash_clearance_many(
    group_a,
    group_b,
    clearance=0.1,  # 0.1m minimum distance
    check_all=True
)

# Collision detection (surface touching)
clashes = tree.clash_collision_many(group_a, group_b)
```

---

## Building Graph Representations

### 1. Extract Core Node Data

```python
def extract_element_data(element):
    """Extract comprehensive data for graph node."""
    data = {
        'id': element.id(),
        'guid': element.GlobalId,
        'type': element.is_a(),
        'name': getattr(element, 'Name', None),
        'description': getattr(element, 'Description', None),
    }

    # Add properties
    try:
        data['properties'] = ifcopenshell.util.element.get_psets(element)
    except:
        data['properties'] = {}

    # Add type info
    try:
        element_type = ifcopenshell.util.element.get_type(element)
        if element_type:
            data['type_name'] = element_type.Name
            data['type_properties'] = ifcopenshell.util.element.get_psets(element_type)
    except:
        pass

    # Add material
    try:
        material = ifcopenshell.util.element.get_material(element)
        if material:
            data['material'] = material.Name if hasattr(material, 'Name') else str(material)
    except:
        pass

    # Add spatial placement
    try:
        if hasattr(element, 'ObjectPlacement') and element.ObjectPlacement:
            matrix = ifcopenshell.util.placement.get_local_placement(element.ObjectPlacement)
            data['position'] = matrix[:3, 3].tolist()
            data['placement_matrix'] = matrix.tolist()
    except:
        pass

    return data
```

### 2. Extract Relationship Edges

```python
def extract_relationships(element):
    """Extract all relationships for an element."""
    relationships = {
        'spatial_container': None,
        'type': None,
        'material': None,
        'connected_to': [],
        'connected_from': [],
        'fills_voids': [],
        'has_openings': [],
        'decomposed_by': [],
        'part_of': None,
        'assigned_to_groups': [],
        'assigned_to_systems': [],
    }

    # Spatial container
    try:
        container = ifcopenshell.util.element.get_container(element)
        if container:
            relationships['spatial_container'] = container.id()
    except:
        pass

    # Type
    try:
        element_type = ifcopenshell.util.element.get_type(element)
        if element_type:
            relationships['type'] = element_type.id()
    except:
        pass

    # Connections (IfcRelConnectsElements)
    try:
        for rel in getattr(element, 'ConnectedTo', []):
            relationships['connected_to'].append({
                'element_id': rel.RelatedElement.id(),
                'connection_type': rel.is_a(),
            })
        for rel in getattr(element, 'ConnectedFrom', []):
            relationships['connected_from'].append({
                'element_id': rel.RelatingElement.id(),
                'connection_type': rel.is_a(),
            })
    except:
        pass

    # Openings
    try:
        for rel in getattr(element, 'HasOpenings', []):
            relationships['has_openings'].append(rel.RelatedOpeningElement.id())
    except:
        pass

    # Fill voids
    try:
        for rel in getattr(element, 'FillsVoids', []):
            relationships['fills_voids'].append(rel.RelatingOpeningElement.id())
    except:
        pass

    # Aggregation
    try:
        aggregate = ifcopenshell.util.element.get_aggregate(element)
        if aggregate:
            relationships['part_of'] = aggregate.id()

        components = ifcopenshell.util.element.get_components(element)
        relationships['decomposed_by'] = [c.id() for c in components]
    except:
        pass

    return relationships
```

### 3. Build NetworkX Graph

```python
import networkx as nx

def build_ifc_graph(model, include_geometry=False):
    """Build comprehensive NetworkX graph from IFC model."""
    G = nx.DiGraph()

    # Get all products (physical objects)
    products = model.by_type('IfcProduct')

    # Add nodes
    for product in products:
        node_data = extract_element_data(product)

        # Optionally add geometry
        if include_geometry:
            try:
                settings = ifcopenshell.geom.settings()
                shape = ifcopenshell.geom.create_shape(settings, product)
                node_data['geometry'] = {
                    'faces': shape.geometry.faces,
                    'verts': shape.geometry.verts,
                }
            except:
                pass

        G.add_node(product.id(), **node_data)

    # Add edges
    for product in products:
        relationships = extract_relationships(product)

        # Spatial containment
        if relationships['spatial_container']:
            G.add_edge(
                relationships['spatial_container'],
                product.id(),
                rel_type='contains'
            )

        # Type relationship
        if relationships['type']:
            G.add_edge(
                product.id(),
                relationships['type'],
                rel_type='instance_of'
            )

        # Physical connections
        for conn in relationships['connected_to']:
            G.add_edge(
                product.id(),
                conn['element_id'],
                rel_type='connects_to',
                connection_type=conn['connection_type']
            )

        # Openings
        for opening_id in relationships['has_openings']:
            G.add_edge(
                product.id(),
                opening_id,
                rel_type='has_opening'
            )

        # Aggregation
        if relationships['part_of']:
            G.add_edge(
                relationships['part_of'],
                product.id(),
                rel_type='aggregates'
            )

    return G
```

---

## Advanced Query Patterns

### 1. Selector Syntax (Query Language)

```python
from ifcopenshell.util.selector import Selector

selector = Selector()

# By GUID
element = selector.parse(model, '#2MLFd4X2f0jRq28Dvww1Vm')

# By type
walls = selector.parse(model, '.IfcWall')

# By property value
walls = selector.parse(model, '.IfcWall[Pset_WallCommon.FireRating = "2HR"]')

# By quantity
slabs = selector.parse(model, '.IfcSlab[Qto_SlabBaseQuantities.NetVolume > "10"]')

# By name pattern (case-sensitive)
elements = selector.parse(model, '.IfcSlab[Name *= "Precast"]')

# By spatial containment
# Find all elements in a specific space
elements = selector.parse(model, '.IfcElement[@IfcSpace.Name = "Office"]')

# Complex queries
query = '''
.IfcWall[Pset_WallCommon.LoadBearing = TRUE]
[Qto_WallBaseQuantities.Length > "5"]
'''
load_bearing_walls = selector.parse(model, query)
```

### 2. Traversing Entity Graphs

```python
# Get all referenced entities (forward relationships)
sub_entities = model.traverse(wall)

# Limit depth
sub_entities = model.traverse(wall, max_levels=2)

# Get inverse references
inverse_refs = model.get_inverse(wall)

# Count inverses (faster than len(get_inverse))
count = model.get_total_inverses(wall)

# Get inverse with indices (which attributes reference this)
refs_with_attrs = model.get_inverse(wall, allow_duplicate=True, with_attribute_indices=True)
```

---

## Performance Optimization

### 1. Batch Operations

```python
# Cache frequently accessed data
element_cache = {}
for element in model.by_type('IfcWall'):
    element_cache[element.id()] = {
        'psets': ifcopenshell.util.element.get_psets(element),
        'container': ifcopenshell.util.element.get_container(element),
    }

# Use batch removal for deleting elements
ifcopenshell.util.element.batch_remove_deep2(model)
ifcopenshell.util.element.remove_deep2(model, element1)
ifcopenshell.util.element.remove_deep2(model, element2)
model = ifcopenshell.util.element.unbatch_remove_deep2(model)
```

### 2. Geometry Processing

```python
# Use multiprocessing
import multiprocessing
iterator = ifcopenshell.geom.iterator(
    settings,
    model,
    num_threads=multiprocessing.cpu_count()
)

# Disable expensive operations
settings.set('DISABLE_OPENING_SUBTRACTIONS', True)
settings.set('DISABLE_TRIANGULATION', True)

# Adjust mesh quality
settings.set('deflection-tolerance', 1e-2)  # Lower = faster but less accurate
```

### 3. Lazy Loading

```python
# Don't extract all data upfront
def get_element_data_lazy(element_id):
    """Lazy data extraction - only when needed."""
    element = model.by_id(element_id)
    return extract_element_data(element)

# Build minimal graph first, enrich on query
G = nx.DiGraph()
for product in model.by_type('IfcProduct'):
    G.add_node(product.id(), type=product.is_a())
# Add relationships...
# Then enrich nodes as needed during queries
```

---

## Common Pitfalls & Solutions

### 1. Missing Relationships

**Problem**: Some elements don't have expected relationships (e.g., no spatial container).

```python
# Always check before accessing
container = ifcopenshell.util.element.get_container(element)
if container is None:
    # Element is not spatially contained
    # Check if it's a spatial element itself
    if element.is_a('IfcSpatialElement'):
        # It's a space/storey/building - no container needed
        pass
```

### 2. Optional Attributes

**Problem**: Accessing optional attributes can raise AttributeError.

```python
# WRONG
name = element.Name  # Might fail if Name is not set

# RIGHT
name = getattr(element, 'Name', 'Unnamed')

# OR
if hasattr(element, 'Name'):
    name = element.Name
```

### 3. Empty Collections

**Problem**: Inverse attributes might return None instead of empty collection.

```python
# WRONG
for rel in element.IsDefinedBy:  # Might be None
    ...

# RIGHT
for rel in getattr(element, 'IsDefinedBy', []) or []:
    ...
```

### 4. Type Hierarchy

**Problem**: `by_type()` doesn't return subclasses.

```python
# WRONG - Won't get all building elements
elements = model.by_type('IfcBuildingElement')

# RIGHT - Get all products (includes walls, doors, windows, etc.)
elements = model.by_type('IfcProduct')
# Then filter by hierarchy
building_elements = [e for e in elements if e.is_a('IfcBuildingElement')]
```

### 5. Units and Coordinate Systems

**Problem**: IFC files can use different units.

```python
import ifcopenshell.util.unit

# Always convert to SI units
unit_scale = ifcopenshell.util.unit.calculate_unit_scale(model)
# If model uses mm, unit_scale = 0.001
# Multiply all dimensions by unit_scale to get meters

# Get coordinates in project units
matrix = ifcopenshell.util.placement.get_local_placement(element.ObjectPlacement)
position_si = matrix[:3, 3] * unit_scale  # Now in meters
```

### 6. Shared Entities

**Problem**: Some entities (like OwnerHistory) are shared by many elements.

```python
# get_inverse() will be VERY slow for shared entities
# Use get_total_inverses() first to check
count = model.get_total_inverses(owner_history)
if count > 1000:
    # Don't enumerate - too many references
    pass
else:
    refs = model.get_inverse(owner_history)
```

---

## Integration with Graph-RAG

### 1. Tool Function Design

```python
def find_elements_by_type(model, element_type: str) -> dict:
    """Tool: Find all elements of a specific type."""
    try:
        elements = model.by_type(element_type)
        return {
            'status': 'success',
            'data': [{
                'id': e.id(),
                'guid': e.GlobalId,
                'name': getattr(e, 'Name', None),
                'type': e.is_a(),
            } for e in elements],
            'count': len(elements)
        }
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e),
            'data': []
        }

def find_spatial_neighbors(model, element_id: int, distance: float = 5.0) -> dict:
    """Tool: Find elements within distance of target element."""
    try:
        element = model.by_id(element_id)

        # Get element position
        matrix = ifcopenshell.util.placement.get_local_placement(
            element.ObjectPlacement
        )
        position = matrix[:3, 3]

        # Build tree for spatial query
        settings = ifcopenshell.geom.settings()
        iterator = ifcopenshell.geom.iterator(settings, model)
        tree = ifcopenshell.geom.tree()
        tree.add_iterator(iterator)

        # Find neighbors
        neighbors = tree.select(position, extend=distance)

        return {
            'status': 'success',
            'data': [{
                'id': n.id(),
                'guid': n.guid,
                'type': model.by_id(n.id()).is_a(),
            } for n in neighbors if n.id() != element_id],
        }
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e),
            'data': []
        }

def get_element_relationships(model, element_id: int) -> dict:
    """Tool: Get all relationships for an element."""
    try:
        element = model.by_id(element_id)
        return {
            'status': 'success',
            'data': extract_relationships(element)
        }
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e),
            'data': {}
        }
```

### 2. Handling Large Models

```python
def parse_ifc_incremental(ifc_path: str, output_dir: str):
    """Parse large IFC files incrementally to avoid memory issues."""
    model = ifcopenshell.open(ifc_path)

    # Process by type
    element_types = ['IfcWall', 'IfcSlab', 'IfcColumn', 'IfcBeam', 'IfcDoor', 'IfcWindow']

    for elem_type in element_types:
        elements = model.by_type(elem_type)

        # Process in chunks
        chunk_size = 1000
        for i in range(0, len(elements), chunk_size):
            chunk = elements[i:i+chunk_size]

            # Extract data
            chunk_data = [extract_element_data(e) for e in chunk]

            # Write to file
            output_file = f"{output_dir}/{elem_type}_{i}.json"
            with open(output_file, 'w') as f:
                json.dump(chunk_data, f)

        # Clear processed elements from memory
        del elements
```

### 3. Validation & Error Handling

```python
def validate_ifc_element(element) -> tuple[bool, list[str]]:
    """Validate element has required data for graph."""
    errors = []

    # Check GUID
    if not hasattr(element, 'GlobalId'):
        errors.append('Missing GlobalId')

    # Check type
    if not element.is_a():
        errors.append('Invalid type')

    # Check spatial placement for physical elements
    if element.is_a('IfcProduct'):
        if not hasattr(element, 'ObjectPlacement') or not element.ObjectPlacement:
            errors.append('Missing ObjectPlacement')

    # Check spatial container
    try:
        container = ifcopenshell.util.element.get_container(element)
        if container is None and element.is_a('IfcElement'):
            errors.append('No spatial container')
    except:
        errors.append('Error checking spatial container')

    return len(errors) == 0, errors
```

---

## Best Practices for Parser Development

### 1. Schema-Agnostic Code

```python
# GOOD - Works with IFC2X3, IFC4, IFC4x3
def get_element_properties(element):
    try:
        return ifcopenshell.util.element.get_psets(element)
    except:
        return {}

# BAD - Assumes specific schema
def get_wall_properties(wall):
    return wall.IsDefinedBy[0].RelatingPropertyDefinition.HasProperties
```

### 2. Robust Error Handling

```python
def safe_extract(element, operation, default=None):
    """Wrapper for safe extraction with fallback."""
    try:
        return operation(element)
    except Exception as e:
        logger.warning(f"Failed to extract from {element.id()}: {e}")
        return default

# Usage
psets = safe_extract(
    element,
    lambda e: ifcopenshell.util.element.get_psets(e),
    default={}
)
```

### 3. Logging & Debugging

```python
import logging

# Enable IfcOpenShell logging
logging.basicConfig(level=logging.INFO)

# Log parsing progress
logger = logging.getLogger('ifc_parser')

def parse_model_with_logging(model):
    logger.info(f"Parsing model: {model.schema}")
    logger.info(f"Total entities: {len(list(model))}")

    products = model.by_type('IfcProduct')
    logger.info(f"Products found: {len(products)}")

    # Log validation issues
    invalid_elements = []
    for product in products:
        valid, errors = validate_ifc_element(product)
        if not valid:
            invalid_elements.append({
                'id': product.id(),
                'errors': errors
            })

    logger.warning(f"Invalid elements: {len(invalid_elements)}")
    return invalid_elements
```

### 4. Testing with Multiple Models

```python
def test_parser(ifc_files: list[str]):
    """Test parser across multiple IFC files."""
    results = []

    for ifc_file in ifc_files:
        try:
            model = ifcopenshell.open(ifc_file)

            # Run parser
            graph = build_ifc_graph(model)

            # Collect stats
            results.append({
                'file': ifc_file,
                'schema': model.schema,
                'nodes': graph.number_of_nodes(),
                'edges': graph.number_of_edges(),
                'status': 'success'
            })
        except Exception as e:
            results.append({
                'file': ifc_file,
                'status': 'error',
                'error': str(e)
            })

    return results
```

---

## Advanced Features

### 1. Working with Custom Property Sets

```python
# Check if pset exists
psets = ifcopenshell.util.element.get_psets(element)
if 'CustomPset_MyData' in psets:
    custom_data = psets['CustomPset_MyData']
    value = custom_data.get('CustomProperty')

# Handle nested properties
def get_nested_property(psets, path):
    """Get property using dot notation: 'PsetName.PropertyName'"""
    parts = path.split('.')
    value = psets
    for part in parts:
        if isinstance(value, dict):
            value = value.get(part)
        else:
            return None
    return value

value = get_nested_property(psets, 'Pset_WallCommon.FireRating')
```

### 2. Quantity Extraction

```python
# Get quantities (similar to properties)
quantities = ifcopenshell.util.element.get_psets(element)

for definition in element.IsDefinedBy:
    related_data = definition.RelatingPropertyDefinition
    if related_data.is_a('IfcElementQuantity'):
        print(f"Quantity Set: {related_data.Name}")
        for quantity in related_data.Quantities:
            if quantity.is_a('IfcQuantityLength'):
                print(f"  {quantity.Name}: {quantity.LengthValue}")
            elif quantity.is_a('IfcQuantityArea'):
                print(f"  {quantity.Name}: {quantity.AreaValue}")
            elif quantity.is_a('IfcQuantityVolume'):
                print(f"  {quantity.Name}: {quantity.VolumeValue}")
```

### 3. Working with Classifications

```python
import ifcopenshell.util.classification

# Get classification references
references = ifcopenshell.util.classification.get_references(element)

for ref in references:
    # ref is (IfcClassificationReference, code, name)
    classification_ref, code, name = ref

    # Get classification system
    system = ifcopenshell.util.classification.get_classification(classification_ref)
    print(f"System: {system.Name}")
    print(f"Code: {code}")
    print(f"Name: {name}")
```

### 4. Material Layer Information

```python
def get_material_layers(element):
    """Extract detailed material layer information."""
    material = ifcopenshell.util.element.get_material(element)

    if not material:
        return None

    if material.is_a('IfcMaterialLayerSet'):
        layers = []
        for layer in material.MaterialLayers:
            layers.append({
                'material': layer.Material.Name,
                'thickness': layer.LayerThickness,
                'priority': getattr(layer, 'Priority', None),
                'description': getattr(layer, 'Description', None),
            })
        return layers

    return None
```

---

## Graph Query Examples

### 1. Find Rooms Adjacent to Element

```python
def find_adjacent_spaces(model, element_id):
    """Find spaces adjacent to an element (e.g., wall)."""
    element = model.by_id(element_id)

    # Get element geometry
    settings = ifcopenshell.geom.settings()
    shape = ifcopenshell.geom.create_shape(settings, element)

    # Build space tree
    iterator = ifcopenshell.geom.iterator(
        settings,
        model,
        include=['IfcSpace']
    )
    tree = ifcopenshell.geom.tree()
    tree.add_iterator(iterator)

    # Find touching/near spaces
    spaces = tree.select_box(shape, extend=0.1)

    return [model.by_id(s.id()) for s in spaces]
```

### 2. Calculate Element Distances

```python
def calculate_distance(element1, element2):
    """Calculate distance between two elements."""
    # Get positions
    pos1 = ifcopenshell.util.placement.get_local_placement(
        element1.ObjectPlacement
    )[:3, 3]
    pos2 = ifcopenshell.util.placement.get_local_placement(
        element2.ObjectPlacement
    )[:3, 3]

    # Euclidean distance
    import numpy as np
    return np.linalg.norm(pos1 - pos2)
```

### 3. Find Load Path

```python
def find_structural_load_path(model, element):
    """Trace structural load path from element to foundation."""
    path = [element]
    current = element

    while True:
        # Check if element is supported by something
        supported_by = None

        # Check spatial container (e.g., on a storey)
        container = ifcopenshell.util.element.get_container(current)
        if container and container.is_a('IfcBuildingStorey'):
            # Find elements below
            below = find_elements_below(model, current, container)
            if below:
                supported_by = below[0]

        # Check physical connections
        for rel in getattr(current, 'ConnectedTo', []):
            if rel.RelatedElement.is_a('IfcColumn') or \
               rel.RelatedElement.is_a('IfcFooting'):
                supported_by = rel.RelatedElement
                break

        if supported_by:
            path.append(supported_by)
            current = supported_by
            if current.is_a('IfcFooting'):
                break  # Reached foundation
        else:
            break  # No more support

    return path
```

---

## Summary: Key Takeaways

### Parser Design Checklist

✅ **Use utility functions** (`ifcopenshell.util.element`) for common operations  
✅ **Handle missing/optional attributes** with `getattr()` and `hasattr()`  
✅ **Check for None** on inverse attributes before iteration  
✅ **Convert units** to SI using `ifcopenshell.util.unit`  
✅ **Build spatial trees** for geometric queries  
✅ **Use iterators** for large models with multiprocessing  
✅ **Validate data** before adding to graph  
✅ **Log errors** and track invalid elements  
✅ **Test with multiple IFC schemas** (IFC2X3, IFC4, IFC4x3)  
✅ **Cache frequently accessed data** to avoid repeated lookups

### Common Data Patterns

| Data Type      | Extraction Method                                             |
| -------------- | ------------------------------------------------------------- |
| Properties     | `util.element.get_psets(element)`                             |
| Type           | `util.element.get_type(element)`                              |
| Material       | `util.element.get_material(element)`                          |
| Container      | `util.element.get_container(element)`                         |
| Position       | `util.placement.get_local_placement(element.ObjectPlacement)` |
| Geometry       | `geom.create_shape(settings, element)`                        |
| Classification | `util.classification.get_references(element)`                 |

### Performance Tips

⚡ Use `by_type()` instead of iterating all entities  
⚡ Cache property sets and avoid repeated `get_psets()` calls  
⚡ Use geometry iterators with multiprocessing for large models  
⚡ Disable opening subtractions for faster geometry  
⚡ Build spatial trees once, reuse for multiple queries  
⚡ Use `get_total_inverses()` before `get_inverse()` for shared entities

---

## References

- **Official Docs**: https://docs.ifcopenshell.org/
- **GitHub**: https://github.com/IfcOpenShell/IfcOpenShell
- **PyPI**: https://pypi.org/project/ifcopenshell/
- **Community**: https://community.osarch.org/
- **IFC Schema**: https://standards.buildingsmart.org/

---

This skill document provides comprehensive guidance for using IfcOpenShell to build robust IFC parsers and graph representations for Graph-RAG systems. The patterns and practices here are based on real-world usage and community best practices as of February 2026.
