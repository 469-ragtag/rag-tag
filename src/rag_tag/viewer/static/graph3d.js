const LINE_VERTEX_SHADER = `
attribute vec3 a_position;
attribute vec3 a_color;
uniform mat4 u_matrix;
varying vec3 v_color;
void main() {
  gl_Position = u_matrix * vec4(a_position, 1.0);
  v_color = a_color;
}
`;

const LINE_FRAGMENT_SHADER = `
precision mediump float;
varying vec3 v_color;
void main() {
  gl_FragColor = vec4(v_color, 0.88);
}
`;

const POINT_VERTEX_SHADER = `
attribute vec3 a_position;
attribute vec3 a_color;
attribute float a_size;
uniform mat4 u_matrix;
varying vec3 v_color;
void main() {
  gl_Position = u_matrix * vec4(a_position, 1.0);
  gl_PointSize = a_size;
  v_color = a_color;
}
`;

const POINT_FRAGMENT_SHADER = `
precision mediump float;
varying vec3 v_color;
void main() {
  vec2 centered = gl_PointCoord - vec2(0.5, 0.5);
  if (dot(centered, centered) > 0.25) {
    discard;
  }
  gl_FragColor = vec4(v_color, 1.0);
}
`;

const DEG_TO_RAD = Math.PI / 180;
const DISTINCT_RELATION_PALETTE = [
  "#3b82f6",
  "#ef4444",
  "#22c55e",
  "#f59e0b",
  "#a855f7",
  "#06b6d4",
  "#f97316",
  "#84cc16",
  "#ec4899",
  "#14b8a6",
  "#e11d48",
  "#8b5cf6",
  "#10b981",
  "#f43f5e",
  "#38bdf8",
  "#d946ef",
];

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function normalizeHexColor(value) {
  const match = String(value || "")
    .trim()
    .match(/^#([0-9a-f]{6})$/i);
  return match ? `#${match[1].toLowerCase()}` : null;
}

function hslToHex(hue, saturation, lightness) {
  const normalizedHue = ((hue % 360) + 360) % 360;
  const saturationUnit = Math.max(0, Math.min(100, saturation)) / 100;
  const lightnessUnit = Math.max(0, Math.min(100, lightness)) / 100;
  const chroma = (1 - Math.abs(2 * lightnessUnit - 1)) * saturationUnit;
  const huePrime = normalizedHue / 60;
  const secondary = chroma * (1 - Math.abs((huePrime % 2) - 1));
  let red = 0;
  let green = 0;
  let blue = 0;

  if (huePrime >= 0 && huePrime < 1) {
    red = chroma;
    green = secondary;
  } else if (huePrime < 2) {
    red = secondary;
    green = chroma;
  } else if (huePrime < 3) {
    green = chroma;
    blue = secondary;
  } else if (huePrime < 4) {
    green = secondary;
    blue = chroma;
  } else if (huePrime < 5) {
    red = secondary;
    blue = chroma;
  } else {
    red = chroma;
    blue = secondary;
  }

  const match = lightnessUnit - chroma / 2;
  const channelToHex = (channel) =>
    Math.round((channel + match) * 255)
      .toString(16)
      .padStart(2, "0");

  return `#${channelToHex(red)}${channelToHex(green)}${channelToHex(blue)}`;
}

function distinctRelationColor(index) {
  if (index < DISTINCT_RELATION_PALETTE.length) {
    return DISTINCT_RELATION_PALETTE[index];
  }
  return hslToHex(index * 137.508, 74, 58);
}

function buildDistinctRelationColors(manifest) {
  const relationNames = manifest.relation_names || [];
  if (!relationNames.length) {
    return { ...(manifest.relation_colors || {}) };
  }

  const existing = manifest.relation_colors || {};
  const usedColors = new Set();
  const normalized = {};

  relationNames.forEach((relation, index) => {
    const currentColor = normalizeHexColor(existing[relation]);
    if (currentColor && !usedColors.has(currentColor)) {
      normalized[relation] = currentColor;
      usedColors.add(currentColor);
      return;
    }

    let candidate = distinctRelationColor(index);
    let offset = 0;
    while (usedColors.has(candidate)) {
      offset += 1;
      candidate = distinctRelationColor(index + offset);
    }
    normalized[relation] = candidate;
    usedColors.add(candidate);
  });

  return normalized;
}

function hexToRgb(hex) {
  const normalized = hex.replace("#", "");
  if (normalized.length !== 6) {
    return [0.45, 0.48, 0.6];
  }
  return [
    parseInt(normalized.slice(0, 2), 16) / 255,
    parseInt(normalized.slice(2, 4), 16) / 255,
    parseInt(normalized.slice(4, 6), 16) / 255,
  ];
}

function vec3Subtract(a, b) {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

function vec3Length(value) {
  return Math.hypot(value[0], value[1], value[2]);
}

function vec3Normalize(value) {
  const length = vec3Length(value) || 1;
  return [value[0] / length, value[1] / length, value[2] / length];
}

function vec3Cross(a, b) {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}

function mat4Multiply(a, b) {
  const out = new Float32Array(16);
  for (let row = 0; row < 4; row += 1) {
    for (let col = 0; col < 4; col += 1) {
      out[col * 4 + row] =
        a[0 * 4 + row] * b[col * 4 + 0] +
        a[1 * 4 + row] * b[col * 4 + 1] +
        a[2 * 4 + row] * b[col * 4 + 2] +
        a[3 * 4 + row] * b[col * 4 + 3];
    }
  }
  return out;
}

function mat4Perspective(fovy, aspect, near, far) {
  const f = 1 / Math.tan(fovy / 2);
  const nf = 1 / (near - far);
  const out = new Float32Array(16);
  out[0] = f / aspect;
  out[5] = f;
  out[10] = (far + near) * nf;
  out[11] = -1;
  out[14] = 2 * far * near * nf;
  return out;
}

function mat4LookAt(eye, center, up) {
  const z = vec3Normalize(vec3Subtract(eye, center));
  const x = vec3Normalize(vec3Cross(up, z));
  const y = vec3Cross(z, x);
  const out = new Float32Array(16);
  out[0] = x[0];
  out[1] = y[0];
  out[2] = z[0];
  out[4] = x[1];
  out[5] = y[1];
  out[6] = z[1];
  out[8] = x[2];
  out[9] = y[2];
  out[10] = z[2];
  out[12] = -(x[0] * eye[0] + x[1] * eye[1] + x[2] * eye[2]);
  out[13] = -(y[0] * eye[0] + y[1] * eye[1] + y[2] * eye[2]);
  out[14] = -(z[0] * eye[0] + z[1] * eye[1] + z[2] * eye[2]);
  out[15] = 1;
  return out;
}

function transformPoint(matrix, point) {
  const x = point[0];
  const y = point[1];
  const z = point[2];
  const w =
    matrix[3] * x + matrix[7] * y + matrix[11] * z + matrix[15];
  const clipX =
    matrix[0] * x + matrix[4] * y + matrix[8] * z + matrix[12];
  const clipY =
    matrix[1] * x + matrix[5] * y + matrix[9] * z + matrix[13];
  const clipZ =
    matrix[2] * x + matrix[6] * y + matrix[10] * z + matrix[14];
  return {
    clipX,
    clipY,
    clipZ,
    clipW: w,
  };
}

function createShader(gl, type, source) {
  const shader = gl.createShader(type);
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    const message = gl.getShaderInfoLog(shader);
    gl.deleteShader(shader);
    throw new Error(message || "Failed to compile shader");
  }
  return shader;
}

function createProgram(gl, vertexSource, fragmentSource) {
  const program = gl.createProgram();
  const vertexShader = createShader(gl, gl.VERTEX_SHADER, vertexSource);
  const fragmentShader = createShader(gl, gl.FRAGMENT_SHADER, fragmentSource);
  gl.attachShader(program, vertexShader);
  gl.attachShader(program, fragmentShader);
  gl.linkProgram(program);
  gl.deleteShader(vertexShader);
  gl.deleteShader(fragmentShader);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    const message = gl.getProgramInfoLog(program);
    gl.deleteProgram(program);
    throw new Error(message || "Failed to link program");
  }
  return program;
}

function bufferData(gl, target, data, usage = gl.STATIC_DRAW) {
  const buffer = gl.createBuffer();
  gl.bindBuffer(target, buffer);
  gl.bufferData(target, data, usage);
  return buffer;
}

function fetchJson(url) {
  return fetch(url, { cache: "no-store" }).then((response) => {
    if (!response.ok) {
      throw new Error(`${url}: ${response.status}`);
    }
    return response.json();
  });
}

function fetchArrayBuffer(url) {
  return fetch(url, { cache: "no-store" }).then((response) => {
    if (!response.ok) {
      throw new Error(`${url}: ${response.status}`);
    }
    return response.arrayBuffer();
  });
}

function createNodeHighlightArrays(nodes, color, size) {
  if (!nodes.length) {
    return {
      positions: new Float32Array(0),
      colors: new Float32Array(0),
      sizes: new Float32Array(0),
    };
  }
  const positions = new Float32Array(nodes.length * 3);
  const colors = new Float32Array(nodes.length * 3);
  const sizes = new Float32Array(nodes.length);
  for (let index = 0; index < nodes.length; index += 1) {
    positions.set(nodes[index], index * 3);
    colors.set(color, index * 3);
    sizes[index] = size;
  }
  return { positions, colors, sizes };
}

function appendColoredLine(positions, colors, start, end, color) {
  positions.push(...start, ...end);
  colors.push(...color, ...color);
}

function niceGridStep(rawValue) {
  const safeValue = Math.max(rawValue, 0.01);
  const exponent = Math.floor(Math.log10(safeValue));
  const base = 10 ** exponent;
  const normalized = safeValue / base;
  if (normalized <= 1) {
    return base;
  }
  if (normalized <= 2) {
    return 2 * base;
  }
  if (normalized <= 5) {
    return 5 * base;
  }
  return 10 * base;
}

function snapDown(value, step) {
  return Math.floor(value / step) * step;
}

function snapUp(value, step) {
  return Math.ceil(value / step) * step;
}

export class WebGLGraphView {
  constructor({
    canvas,
    overlayCanvas,
    stageMetaElement,
    selectionMetaElement,
  }) {
    this.canvas = canvas;
    this.overlayCanvas = overlayCanvas;
    this.stageMetaElement = stageMetaElement;
    this.selectionMetaElement = selectionMetaElement;
    this.gl = canvas.getContext("webgl", {
      antialias: true,
      alpha: false,
      powerPreference: "high-performance",
    });
    if (!this.gl) {
      throw new Error("WebGL is not available in this browser.");
    }

    this.overlayContext = overlayCanvas.getContext("2d");
    this.lineProgram = createProgram(
      this.gl,
      LINE_VERTEX_SHADER,
      LINE_FRAGMENT_SHADER
    );
    this.pointProgram = createProgram(
      this.gl,
      POINT_VERTEX_SHADER,
      POINT_FRAGMENT_SHADER
    );

    this.nodeLayer = null;
    this.highlightLayer = null;
    this.edgeLayers = new Map();
    this.worldReferenceLayer = null;
    this.worldReferenceKey = "";
    this.bboxLayer = null;
    this.meshLayer = null;
    this.manifest = null;
    this.assetVersion = Date.now().toString();
    this.nodeMeta = [];
    this.nodeMetaById = new Map();
    this.nodePositions = null;
    this.nodeColors = null;
    this.nodeSizes = null;
    this.bounds = { min: [0, 0, 0], max: [0, 0, 0], center: [0, 0, 0], radius: 1 };
    this.mode = "all";
    this.showNodes = true;
    this.showEdges = true;
    this.showEdgeLabels = false;
    this.showBBoxes = false;
    this.showMeshes = false;
    this.visibleRelations = new Set();
    this.hiddenRelations = new Set();
    this.selectedNodeId = null;
    this.hoveredNodeId = null;
    this.onSelectionChangeHandler = null;
    this.onHoverChangeHandler = null;
    this.projectedNodeCache = [];
    this.dragState = null;
    this.viewProjectionMatrix = mat4Multiply(
      mat4Perspective(45 * DEG_TO_RAD, 1, 0.1, 1000),
      mat4LookAt([0, 0, 10], [0, 0, 0], [0, 0, 1])
    );
    this.camera = {
      yaw: -35 * DEG_TO_RAD,
      pitch: 28 * DEG_TO_RAD,
      distance: 24,
      target: [0, 0, 0],
    };
    this.defaultCamera = {
      yaw: this.camera.yaw,
      pitch: this.camera.pitch,
      distance: this.camera.distance,
      target: [...this.camera.target],
    };
    this.needsRender = false;

    this._bindEvents();
    this.resize();
  }

  _bindEvents() {
    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.resizeObserver.observe(this.canvas);
    this.canvas.addEventListener("pointerdown", (event) => this._onPointerDown(event));
    this.canvas.addEventListener("pointermove", (event) => this._onPointerMove(event));
    this.canvas.addEventListener("pointerup", (event) => this._onPointerUp(event));
    this.canvas.addEventListener("pointerleave", () => this._clearHover());
    this.canvas.addEventListener("wheel", (event) => this._onWheel(event), {
      passive: false,
    });
  }

  onSelectionChange(callback) {
    this.onSelectionChangeHandler = callback;
  }

  onHoverChange(callback) {
    this.onHoverChangeHandler = callback;
  }

  async loadFromManifestUrl(manifestUrl) {
    this.manifest = await fetchJson(manifestUrl);
    this.manifest.relation_colors = buildDistinctRelationColors(this.manifest);
    if (this.manifest.legend?.entries) {
      this.manifest.legend.entries = this.manifest.legend.entries.map((entry) => {
        const relation =
          entry.relation_id || entry.relation || entry.label || "related_to";
        return {
          ...entry,
          swatch: this.manifest.relation_colors?.[relation] || entry.swatch || "#4b5563",
        };
      });
    }
    this.assetVersion = String(this.manifest.build_id || Date.now());
    this.mode = this.manifest.render_defaults?.mode || "all";
    this.showNodes = this.manifest.render_defaults?.nodes !== false;
    this.showEdges = this.manifest.render_defaults?.edges !== false;
    this.showEdgeLabels = Boolean(this.manifest.render_defaults?.edge_labels);
    this.showBBoxes = Boolean(this.manifest.render_defaults?.bboxes);
    this.showMeshes = Boolean(this.manifest.render_defaults?.meshes);
    this.visibleRelations = new Set(this.manifest.viewer_modes?.all || []);
    this.hiddenRelations.clear();

    const [nodeMeta, nodeBuffer] = await Promise.all([
      fetchJson(this._assetUrl(this.manifest.files.node_meta.path)),
      fetchArrayBuffer(this._assetUrl(this.manifest.files.nodes.path)),
    ]);
    this.nodeMeta = nodeMeta;
    this.nodeMetaById = new Map(nodeMeta.map((item) => [item.id, item]));
    this.nodePositions = new Float32Array(nodeBuffer);
    this.nodeColors = new Float32Array(this.nodeMeta.length * 3);
    this.nodeSizes = new Float32Array(this.nodeMeta.length);
    this.projectedNodeCache = new Array(this.nodeMeta.length);
    this.nodeMeta.forEach((node, index) => {
      const color = hexToRgb(
        this.manifest.node_groups?.[node.class_name]?.color || "#22c55e"
      );
      this.nodeColors.set(color, index * 3);
      this.nodeSizes[index] = 6;
      node.position = [
        this.nodePositions[index * 3],
        this.nodePositions[index * 3 + 1],
        this.nodePositions[index * 3 + 2],
      ];
      node.index = index;
    });
    this.bounds = this._computeBounds();
    this.defaultCamera = {
      yaw: -35 * DEG_TO_RAD,
      pitch: 28 * DEG_TO_RAD,
      distance: Math.max(this.bounds.radius * 2.4, 24),
      target: [...this.bounds.center],
    };
    this.camera = {
      yaw: this.defaultCamera.yaw,
      pitch: this.defaultCamera.pitch,
      distance: this.defaultCamera.distance,
      target: [...this.defaultCamera.target],
    };
    this.worldReferenceKey = "";
    this.nodeLayer = this._createPointLayer(
      this.nodePositions,
      this.nodeColors,
      this.nodeSizes
    );

    await Promise.all(
      Object.entries(this.manifest.files.edges).map(([category, config]) =>
        this._loadEdgeCategory(category, config)
      )
    );
    this.setMode(this.mode, { suppressRender: true });
    this._updateHighlightLayer();
    this._setStageMeta(
      `${this.manifest.node_count} nodes • ${this.manifest.edge_count} edges`
    );
    this.requestRender();
  }

  _assetUrl(relativePath) {
    return `/debug/graph-assets/${relativePath}?v=${encodeURIComponent(this.assetVersion)}`;
  }

  _computeBounds() {
    const min = [...(this.manifest.bounds?.min || [0, 0, 0])];
    const max = [...(this.manifest.bounds?.max || [0, 0, 0])];
    const center = [
      (min[0] + max[0]) / 2,
      (min[1] + max[1]) / 2,
      (min[2] + max[2]) / 2,
    ];
    const radius =
      Math.max(
        Math.hypot(max[0] - min[0], max[1] - min[1], max[2] - min[2]) / 2,
        1
      ) || 1;
    return { min, max, center, radius };
  }

  async _loadEdgeCategory(category, config) {
    const buffer = await fetchArrayBuffer(this._assetUrl(config.path));
    const view = new DataView(buffer);
    const stride = config.stride_bytes || 12;
    const relationNames = this.manifest.relation_names || [];
    const sourceKinds = this.manifest.source_kinds || [];
    const buckets = new Map();

    for (let offset = 0; offset < view.byteLength; offset += stride) {
      const sourceIndex = view.getUint32(offset, true);
      const targetIndex = view.getUint32(offset + 4, true);
      const relationIndex = view.getUint16(offset + 8, true);
      const sourceKindIndex = view.getUint8(offset + 10, true);
      const relation = relationNames[relationIndex] || "related_to";
      const sourceKind = sourceKinds[sourceKindIndex] || "unknown";
      const bucket =
        buckets.get(relation) ||
        {
          positions: [],
          colors: [],
          midpoints: [],
          sourceKinds: new Set(),
          count: 0,
          category,
          relation,
        };
      const sourceOffset = sourceIndex * 3;
      const targetOffset = targetIndex * 3;
      const source = [
        this.nodePositions[sourceOffset],
        this.nodePositions[sourceOffset + 1],
        this.nodePositions[sourceOffset + 2],
      ];
      const target = [
        this.nodePositions[targetOffset],
        this.nodePositions[targetOffset + 1],
        this.nodePositions[targetOffset + 2],
      ];
      const color = hexToRgb(
        this.manifest.relation_colors?.[relation] || "#6b7280"
      );
      bucket.positions.push(...source, ...target);
      bucket.colors.push(...color, ...color);
      bucket.midpoints.push(
        (source[0] + target[0]) / 2,
        (source[1] + target[1]) / 2,
        (source[2] + target[2]) / 2
      );
      bucket.sourceKinds.add(sourceKind);
      bucket.count += 1;
      buckets.set(relation, bucket);
    }

    for (const [relation, bucket] of buckets.entries()) {
      this.edgeLayers.set(relation, {
        ...bucket,
        edgeCount: bucket.count,
        vertexCount: bucket.positions.length / 3,
        visible: true,
        positionsArray: new Float32Array(bucket.positions),
        colorsArray: new Float32Array(bucket.colors),
        midpointArray: new Float32Array(bucket.midpoints),
      });
    }

    for (const layer of this.edgeLayers.values()) {
      if (!layer.positionBuffer) {
        layer.positionBuffer = bufferData(
          this.gl,
          this.gl.ARRAY_BUFFER,
          layer.positionsArray
        );
        layer.colorBuffer = bufferData(
          this.gl,
          this.gl.ARRAY_BUFFER,
          layer.colorsArray
        );
      }
    }
  }

  _createPointLayer(positions, colors, sizes) {
    return {
      positionBuffer: bufferData(this.gl, this.gl.ARRAY_BUFFER, positions),
      colorBuffer: bufferData(this.gl, this.gl.ARRAY_BUFFER, colors),
      sizeBuffer: bufferData(this.gl, this.gl.ARRAY_BUFFER, sizes),
      count: sizes.length,
    };
  }

  _createLineLayer(positions, colors) {
    return {
      positionBuffer: bufferData(this.gl, this.gl.ARRAY_BUFFER, positions),
      colorBuffer: bufferData(this.gl, this.gl.ARRAY_BUFFER, colors),
      count: positions.length / 3,
    };
  }

  _disposeLineLayer(layer) {
    if (!layer) {
      return;
    }
    if (layer.positionBuffer) {
      this.gl.deleteBuffer(layer.positionBuffer);
    }
    if (layer.colorBuffer) {
      this.gl.deleteBuffer(layer.colorBuffer);
    }
  }

  _ensureWorldReferenceLayer() {
    const [boundsMinX, boundsMinY, boundsMinZ] = this.bounds.min;
    const [boundsMaxX, boundsMaxY, boundsMaxZ] = this.bounds.max;
    const spanX = Math.max(boundsMaxX - boundsMinX, 1);
    const spanY = Math.max(boundsMaxY - boundsMinY, 1);
    const spanZ = Math.max(boundsMaxZ - boundsMinZ, 1);
    const step = niceGridStep(
      Math.max(
        this.camera.distance / 10,
        Math.min(spanX, spanY, spanZ) / 28,
        0.05
      )
    );
    const halfX = Math.min(spanX / 2, Math.max(this.camera.distance * 1.2, step * 4));
    const halfY = Math.min(spanY / 2, Math.max(this.camera.distance * 1.2, step * 4));
    const verticalSpan = Math.min(
      spanZ,
      Math.max(this.camera.distance * 1.45, step * 5)
    );
    const centerX = Math.min(Math.max(this.camera.target[0], boundsMinX), boundsMaxX);
    const centerY = Math.min(Math.max(this.camera.target[1], boundsMinY), boundsMaxY);
    let minX = snapDown(Math.max(boundsMinX, centerX - halfX), step);
    let maxX = snapUp(Math.min(boundsMaxX, centerX + halfX), step);
    let minY = snapDown(Math.max(boundsMinY, centerY - halfY), step);
    let maxY = snapUp(Math.min(boundsMaxY, centerY + halfY), step);
    const minZ = snapDown(boundsMinZ, step);
    let maxZ = snapUp(Math.min(boundsMaxZ, boundsMinZ + verticalSpan), step);

    if (maxX - minX < step) {
      maxX = minX + step;
    }
    if (maxY - minY < step) {
      maxY = minY + step;
    }
    if (maxZ - minZ < step) {
      maxZ = minZ + step;
    }

    const key = [
      step.toFixed(3),
      minX.toFixed(3),
      maxX.toFixed(3),
      minY.toFixed(3),
      maxY.toFixed(3),
      minZ.toFixed(3),
      maxZ.toFixed(3),
    ].join("|");
    if (key === this.worldReferenceKey && this.worldReferenceLayer) {
      return;
    }

    const positions = [];
    const colors = [];
    const frameColor = hexToRgb("#4c566a");
    const gridColor = hexToRgb("#313244");
    const xAxisColor = hexToRgb("#f38ba8");
    const yAxisColor = hexToRgb("#a6e3a1");
    const zAxisColor = hexToRgb("#89b4fa");
    const corners = [
      [minX, minY, minZ],
      [maxX, minY, minZ],
      [maxX, maxY, minZ],
      [minX, maxY, minZ],
      [minX, minY, maxZ],
      [maxX, minY, maxZ],
      [maxX, maxY, maxZ],
      [minX, maxY, maxZ],
    ];
    const edges = [
      [0, 1],
      [1, 2],
      [2, 3],
      [3, 0],
      [4, 5],
      [5, 6],
      [6, 7],
      [7, 4],
      [0, 4],
      [1, 5],
      [2, 6],
      [3, 7],
    ];
    for (const [startIndex, endIndex] of edges) {
      appendColoredLine(
        positions,
        colors,
        corners[startIndex],
        corners[endIndex],
        frameColor
      );
    }

    for (let x = minX; x <= maxX + step * 0.25; x += step) {
      const isAxis = Math.abs(x - minX) < step * 0.1;
      appendColoredLine(
        positions,
        colors,
        [x, minY, minZ],
        [x, maxY, minZ],
        isAxis ? xAxisColor : gridColor
      );
      appendColoredLine(
        positions,
        colors,
        [x, minY, minZ],
        [x, minY, maxZ],
        isAxis ? xAxisColor : gridColor
      );
    }

    for (let y = minY; y <= maxY + step * 0.25; y += step) {
      const isAxis = Math.abs(y - minY) < step * 0.1;
      appendColoredLine(
        positions,
        colors,
        [minX, y, minZ],
        [maxX, y, minZ],
        isAxis ? yAxisColor : gridColor
      );
      appendColoredLine(
        positions,
        colors,
        [minX, y, minZ],
        [minX, y, maxZ],
        isAxis ? yAxisColor : gridColor
      );
    }

    for (let z = minZ; z <= maxZ + step * 0.25; z += step) {
      const isAxis = Math.abs(z - minZ) < step * 0.1;
      appendColoredLine(
        positions,
        colors,
        [minX, minY, z],
        [maxX, minY, z],
        isAxis ? zAxisColor : gridColor
      );
      appendColoredLine(
        positions,
        colors,
        [minX, minY, z],
        [minX, maxY, z],
        isAxis ? zAxisColor : gridColor
      );
    }

    this._disposeLineLayer(this.worldReferenceLayer);
    this.worldReferenceLayer = this._createLineLayer(
      new Float32Array(positions),
      new Float32Array(colors)
    );
    this.worldReferenceKey = key;
  }

  async setOverlayEnabled(name, enabled) {
    if (name === "bbox") {
      this.showBBoxes = enabled;
      if (enabled && !this.bboxLayer) {
        const overlayConfig = this.manifest.files.overlays?.bbox;
        if (overlayConfig?.available && overlayConfig.path) {
          const buffer = await fetchArrayBuffer(this._assetUrl(overlayConfig.path));
          const positions = new Float32Array(buffer);
          const colors = new Float32Array(positions.length);
          const color = hexToRgb("#f59e0b");
          for (let index = 0; index < colors.length; index += 3) {
            colors.set(color, index);
          }
          this.bboxLayer = this._createLineLayer(positions, colors);
        }
      }
    }
    if (name === "mesh") {
      this.showMeshes = enabled;
      if (enabled && !this.meshLayer) {
        const overlayConfig = this.manifest.files.overlays?.mesh;
        if (overlayConfig?.available && overlayConfig.manifest_path) {
          const manifest = await fetchJson(this._assetUrl(overlayConfig.manifest_path));
          if (manifest.available && manifest.path) {
            const buffer = await fetchArrayBuffer(this._assetUrl(manifest.path));
            const positions = new Float32Array(buffer);
            const colors = new Float32Array(positions.length);
            const color = hexToRgb("#89b4fa");
            for (let index = 0; index < colors.length; index += 3) {
              colors.set(color, index);
            }
            this.meshLayer = this._createLineLayer(positions, colors);
          }
        }
      }
    }
    this.requestRender();
  }

  setMode(mode, { suppressRender = false } = {}) {
    this.mode = mode;
    const relationSets = this.manifest?.viewer_modes || {};
    if (mode === "nodes") {
      this.showNodes = true;
      this.showEdges = false;
      this.visibleRelations = new Set();
    } else if (mode === "edges") {
      this.showNodes = false;
      this.showEdges = true;
      this.visibleRelations = new Set(relationSets.all || []);
    } else if (mode === "all") {
      this.showNodes = true;
      this.showEdges = true;
      this.visibleRelations = new Set(relationSets.all || []);
      // Full Graph is the hard reset view, so restore every relation family.
      this.hiddenRelations.clear();
    } else {
      this.showNodes = true;
      this.showEdges = true;
      this.visibleRelations = new Set(relationSets[mode] || []);
    }
    this._setStageMeta(
      `${this.manifest?.node_count || 0} nodes • ${this.getVisibleEdgeCount()} visible edges • mode: ${mode}`
    );
    if (!suppressRender) {
      this.requestRender();
    }
  }

  isRelationVisible(relation) {
    return this.visibleRelations.has(relation) && !this.hiddenRelations.has(relation);
  }

  setRelationVisible(relation, visible) {
    if (!this.edgeLayers.has(relation)) {
      return false;
    }
    if (visible) {
      this.hiddenRelations.delete(relation);
    } else {
      this.hiddenRelations.add(relation);
    }
    this._setStageMeta(
      `${this.manifest?.node_count || 0} nodes • ${this.getVisibleEdgeCount()} visible edges • mode: ${this.mode}`
    );
    this.requestRender();
    return true;
  }

  getRelationLegendEntries() {
    const manifestEntries = this.manifest?.legend?.entries || [];
    if (manifestEntries.length) {
      return manifestEntries.map((entry) => {
        const relation =
          entry.relation_id || entry.relation || entry.label || "related_to";
        const layer = this.edgeLayers.get(relation);
        return {
          relation,
          label:
            entry.label ||
            this.manifest?.relation_labels?.[relation] ||
            relation,
          subtitle:
            entry.subtitle ||
            this.manifest?.relation_explanations?.[relation] ||
            "graph relation",
          count: layer?.edgeCount ?? layer?.count ?? entry.count ?? 0,
          swatch:
            this.manifest?.relation_colors?.[relation] || entry.swatch || "#4b5563",
          visible: this.isRelationVisible(relation),
          available: this.visibleRelations.has(relation),
        };
      });
    }

    const relationNames = this.manifest?.relation_names || [];
    return relationNames
      .map((relation) => {
        const layer = this.edgeLayers.get(relation);
        return {
          relation,
          label: this.manifest?.relation_labels?.[relation] || relation,
          subtitle:
            this.manifest?.relation_explanations?.[relation] || "graph relation",
          count: layer?.edgeCount ?? layer?.count ?? 0,
          swatch: this.manifest?.relation_colors?.[relation] || "#4b5563",
          visible: this.isRelationVisible(relation),
          available: this.visibleRelations.has(relation),
        };
      })
      .sort((left, right) => {
        if (right.count !== left.count) {
          return right.count - left.count;
        }
        return left.label.localeCompare(right.label);
      });
  }

  setEdgeLabelsEnabled(enabled) {
    this.showEdgeLabels = enabled;
    this.requestRender();
  }

  resetView() {
    this.camera = {
      yaw: this.defaultCamera.yaw,
      pitch: this.defaultCamera.pitch,
      distance: this.defaultCamera.distance,
      target: [...this.defaultCamera.target],
    };
    this.requestRender();
  }

  focusNode(nodeId) {
    const node = this.nodeMetaById.get(nodeId);
    if (!node) {
      return false;
    }
    this.camera.target = [...node.position];
    const minDistance = Math.max(this.bounds.radius * 0.15, 2);
    const maxDistance = Math.max(this.bounds.radius * 12, 250);
    const focusDistance = clamp(
      Math.max(this.bounds.radius * 0.22, 8),
      minDistance,
      maxDistance
    );
    this.camera.distance = Math.min(this.camera.distance, focusDistance);
    this.selectNode(nodeId, { focusOnly: false });
    return true;
  }

  focusFirstNodeInClass(className) {
    const node = this.nodeMeta.find((entry) => entry.class_name === className);
    if (!node) {
      return false;
    }
    return this.focusNode(node.id);
  }

  selectNode(nodeId, { focusOnly = false } = {}) {
    if (!this.nodeMetaById.has(nodeId)) {
      return;
    }
    this.selectedNodeId = nodeId;
    this._updateHighlightLayer();
    const node = this.nodeMetaById.get(nodeId);
    this._setSelectionMeta(`${node.label} • ${node.class_name} • ${node.id}`);
    if (this.onSelectionChangeHandler && !focusOnly) {
      this.onSelectionChangeHandler(node);
    }
    this.requestRender();
  }

  getVisibleEdgeCount() {
    if (!this.showEdges) {
      return 0;
    }
    let total = 0;
    for (const [relation, layer] of this.edgeLayers.entries()) {
      if (this.isRelationVisible(relation)) {
        total += layer.edgeCount ?? layer.count ?? 0;
      }
    }
    return total;
  }

  resize() {
    const devicePixelRatio = window.devicePixelRatio || 1;
    const rect = this.canvas.getBoundingClientRect();
    const width = Math.max(1, Math.floor(rect.width * devicePixelRatio));
    const height = Math.max(1, Math.floor(rect.height * devicePixelRatio));
    if (this.canvas.width !== width || this.canvas.height !== height) {
      this.canvas.width = width;
      this.canvas.height = height;
      this.overlayCanvas.width = width;
      this.overlayCanvas.height = height;
      this.overlayCanvas.style.width = `${rect.width}px`;
      this.overlayCanvas.style.height = `${rect.height}px`;
      this.gl.viewport(0, 0, width, height);
      this.requestRender();
    }
  }

  requestRender() {
    if (this.needsRender) {
      return;
    }
    this.needsRender = true;
    requestAnimationFrame(() => {
      this.needsRender = false;
      this.render();
    });
  }

  render() {
    if (!this.nodeLayer || !this.manifest) {
      return;
    }
    this.resize();
    const gl = this.gl;
    const aspect = gl.canvas.width / Math.max(gl.canvas.height, 1);
    const projection = mat4Perspective(42 * DEG_TO_RAD, aspect, 0.1, this.bounds.radius * 8 + 1000);
    const eye = [
      this.camera.target[0] +
        this.camera.distance * Math.cos(this.camera.pitch) * Math.cos(this.camera.yaw),
      this.camera.target[1] +
        this.camera.distance * Math.cos(this.camera.pitch) * Math.sin(this.camera.yaw),
      this.camera.target[2] + this.camera.distance * Math.sin(this.camera.pitch),
    ];
    const view = mat4LookAt(eye, this.camera.target, [0, 0, 1]);
    this.viewProjectionMatrix = mat4Multiply(projection, view);

    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    gl.enable(gl.DEPTH_TEST);
    gl.clearColor(0.066, 0.066, 0.106, 1);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

    this._ensureWorldReferenceLayer();
    if (this.worldReferenceLayer) {
      this._drawLines(this.worldReferenceLayer);
    }
    if (this.showEdges) {
      for (const [relation, layer] of this.edgeLayers.entries()) {
        if (!this.isRelationVisible(relation)) {
          continue;
        }
        this._drawLines(layer);
      }
    }
    if (this.showBBoxes && this.bboxLayer) {
      this._drawLines(this.bboxLayer);
    }
    if (this.showMeshes && this.meshLayer) {
      this._drawLines(this.meshLayer);
    }
    if (this.showNodes) {
      this._drawPoints(this.nodeLayer);
    }
    if (this.highlightLayer) {
      this._drawPoints(this.highlightLayer);
    }
    this._drawOverlay();
  }

  _drawLines(layer) {
    const gl = this.gl;
    gl.useProgram(this.lineProgram);
    const positionLoc = gl.getAttribLocation(this.lineProgram, "a_position");
    const colorLoc = gl.getAttribLocation(this.lineProgram, "a_color");
    const matrixLoc = gl.getUniformLocation(this.lineProgram, "u_matrix");
    gl.uniformMatrix4fv(matrixLoc, false, this.viewProjectionMatrix);
    gl.bindBuffer(gl.ARRAY_BUFFER, layer.positionBuffer);
    gl.enableVertexAttribArray(positionLoc);
    gl.vertexAttribPointer(positionLoc, 3, gl.FLOAT, false, 0, 0);
    gl.bindBuffer(gl.ARRAY_BUFFER, layer.colorBuffer);
    gl.enableVertexAttribArray(colorLoc);
    gl.vertexAttribPointer(colorLoc, 3, gl.FLOAT, false, 0, 0);
    gl.drawArrays(gl.LINES, 0, layer.vertexCount ?? layer.count);
  }

  _drawPoints(layer) {
    const gl = this.gl;
    gl.useProgram(this.pointProgram);
    const positionLoc = gl.getAttribLocation(this.pointProgram, "a_position");
    const colorLoc = gl.getAttribLocation(this.pointProgram, "a_color");
    const sizeLoc = gl.getAttribLocation(this.pointProgram, "a_size");
    const matrixLoc = gl.getUniformLocation(this.pointProgram, "u_matrix");
    gl.uniformMatrix4fv(matrixLoc, false, this.viewProjectionMatrix);
    gl.bindBuffer(gl.ARRAY_BUFFER, layer.positionBuffer);
    gl.enableVertexAttribArray(positionLoc);
    gl.vertexAttribPointer(positionLoc, 3, gl.FLOAT, false, 0, 0);
    gl.bindBuffer(gl.ARRAY_BUFFER, layer.colorBuffer);
    gl.enableVertexAttribArray(colorLoc);
    gl.vertexAttribPointer(colorLoc, 3, gl.FLOAT, false, 0, 0);
    gl.bindBuffer(gl.ARRAY_BUFFER, layer.sizeBuffer);
    gl.enableVertexAttribArray(sizeLoc);
    gl.vertexAttribPointer(sizeLoc, 1, gl.FLOAT, false, 0, 0);
    gl.drawArrays(gl.POINTS, 0, layer.count);
  }

  _drawOverlay() {
    const ctx = this.overlayContext;
    if (!ctx) {
      return;
    }
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, this.overlayCanvas.width, this.overlayCanvas.height);
    ctx.save();
    ctx.scale(window.devicePixelRatio || 1, window.devicePixelRatio || 1);
    ctx.font = '12px "Segoe UI", Tahoma, sans-serif';
    ctx.textBaseline = "top";

    const drawNodeLabel = (nodeId, color) => {
      const node = this.nodeMetaById.get(nodeId);
      if (!node) {
        return;
      }
      const projected = this._projectNode(node.index);
      if (!projected?.visible) {
        return;
      }
      ctx.fillStyle = color;
      ctx.fillText(
        node.label,
        projected.x + 10,
        projected.y - 6
      );
    };

    if (this.showEdgeLabels) {
      this._drawEdgeLabels(ctx);
    }
    if (this.hoveredNodeId && this.hoveredNodeId !== this.selectedNodeId) {
      drawNodeLabel(this.hoveredNodeId, "#89b4fa");
    }
    if (this.selectedNodeId) {
      drawNodeLabel(this.selectedNodeId, "#a6e3a1");
    }
    ctx.restore();
  }

  _drawEdgeLabels(ctx) {
    const maxLabels = 180;
    let drawn = 0;
    for (const [relation, layer] of this.edgeLayers.entries()) {
      if (!this.isRelationVisible(relation) || drawn >= maxLabels) {
        continue;
      }
      const count = layer.edgeCount ?? layer.count ?? 0;
      const step = Math.max(1, Math.ceil(count / Math.max(1, Math.floor(maxLabels / 6))));
      ctx.fillStyle = this.manifest.relation_colors?.[relation] || "#a6adc8";
      const relationLabel = this.manifest.relation_labels?.[relation] || relation;
      for (let index = 0; index < count && drawn < maxLabels; index += step) {
        const projected = this._projectPoint([
          layer.midpointArray[index * 3],
          layer.midpointArray[index * 3 + 1],
          layer.midpointArray[index * 3 + 2],
        ]);
        if (!projected?.visible) {
          continue;
        }
        ctx.fillText(relationLabel, projected.x + 4, projected.y + 4);
        drawn += 1;
      }
    }
  }

  _projectPoint(point) {
    const projected = transformPoint(this.viewProjectionMatrix, point);
    if (projected.clipW <= 0) {
      return { visible: false };
    }
    const ndcX = projected.clipX / projected.clipW;
    const ndcY = projected.clipY / projected.clipW;
    const ndcZ = projected.clipZ / projected.clipW;
    const width = this.canvas.clientWidth || 1;
    const height = this.canvas.clientHeight || 1;
    return {
      visible:
        Math.abs(ndcX) <= 1.15 &&
        Math.abs(ndcY) <= 1.15 &&
        Math.abs(ndcZ) <= 1.1,
      x: ((ndcX + 1) / 2) * width,
      y: ((1 - ndcY) / 2) * height,
      depth: ndcZ,
    };
  }

  _projectNode(index) {
    const cached = this.projectedNodeCache[index];
    if (cached && cached.frameMatrix === this.viewProjectionMatrix) {
      return cached.value;
    }
    const value = this._projectPoint([
      this.nodePositions[index * 3],
      this.nodePositions[index * 3 + 1],
      this.nodePositions[index * 3 + 2],
    ]);
    this.projectedNodeCache[index] = {
      frameMatrix: this.viewProjectionMatrix,
      value,
    };
    return value;
  }

  _updateHighlightLayer() {
    const highlighted = [];
    if (this.selectedNodeId) {
      const node = this.nodeMetaById.get(this.selectedNodeId);
      if (node) {
        highlighted.push(node.position);
      }
    }
    if (this.hoveredNodeId && this.hoveredNodeId !== this.selectedNodeId) {
      const node = this.nodeMetaById.get(this.hoveredNodeId);
      if (node) {
        highlighted.push(node.position);
      }
    }
    const highlightArrays = createNodeHighlightArrays(
      highlighted,
      hexToRgb("#a6e3a1"),
      12
    );
    this.highlightLayer = this._createPointLayer(
      highlightArrays.positions,
      highlightArrays.colors,
      highlightArrays.sizes
    );
  }

  _pickNode(clientX, clientY) {
    const rect = this.canvas.getBoundingClientRect();
    const x = clientX - rect.left;
    const y = clientY - rect.top;
    let best = null;
    let bestDistance = 18;
    for (const node of this.nodeMeta) {
      const projected = this._projectNode(node.index);
      if (!projected.visible) {
        continue;
      }
      const distance = Math.hypot(projected.x - x, projected.y - y);
      if (distance < bestDistance) {
        bestDistance = distance;
        best = node;
      }
    }
    return best;
  }

  _clearHover() {
    if (this.hoveredNodeId !== null) {
      this.hoveredNodeId = null;
      this._updateHighlightLayer();
      if (this.onHoverChangeHandler) {
        this.onHoverChangeHandler(null);
      }
      this.requestRender();
    }
  }

  _onPointerDown(event) {
    this.canvas.setPointerCapture(event.pointerId);
    this.dragState = {
      pointerId: event.pointerId,
      startX: event.clientX,
      startY: event.clientY,
      lastX: event.clientX,
      lastY: event.clientY,
      moved: false,
    };
    this.canvas.classList.add("dragging");
  }

  _onPointerMove(event) {
    if (this.dragState && this.dragState.pointerId === event.pointerId) {
      const dx = event.clientX - this.dragState.lastX;
      const dy = event.clientY - this.dragState.lastY;
      if (Math.abs(dx) + Math.abs(dy) > 0) {
        this.dragState.moved = true;
      }
      this.dragState.lastX = event.clientX;
      this.dragState.lastY = event.clientY;
      this.camera.yaw -= dx * 0.008;
      this.camera.pitch = clamp(this.camera.pitch + dy * 0.008, -1.45, 1.45);
      this.requestRender();
      return;
    }
    const hovered = this._pickNode(event.clientX, event.clientY);
    const hoveredId = hovered?.id || null;
    if (hoveredId !== this.hoveredNodeId) {
      this.hoveredNodeId = hoveredId;
      this._updateHighlightLayer();
      if (this.onHoverChangeHandler) {
        this.onHoverChangeHandler(hovered || null);
      }
      this.requestRender();
    }
  }

  _onPointerUp(event) {
    if (!this.dragState || this.dragState.pointerId !== event.pointerId) {
      return;
    }
    this.canvas.releasePointerCapture(event.pointerId);
    this.canvas.classList.remove("dragging");
    const clickLike =
      !this.dragState.moved &&
      Math.hypot(
        event.clientX - this.dragState.startX,
        event.clientY - this.dragState.startY
      ) < 6;
    this.dragState = null;
    if (clickLike) {
      const picked = this._pickNode(event.clientX, event.clientY);
      if (picked) {
        this.selectNode(picked.id);
      }
    }
  }

  _onWheel(event) {
    event.preventDefault();
    const delta = Math.sign(event.deltaY);
    this.camera.distance = clamp(
      this.camera.distance * (delta > 0 ? 1.08 : 0.92),
      Math.max(this.bounds.radius * 0.15, 2),
      Math.max(this.bounds.radius * 12, 250)
    );
    this.requestRender();
  }

  _setStageMeta(text) {
    if (this.stageMetaElement) {
      this.stageMetaElement.textContent = text;
    }
  }

  _setSelectionMeta(text) {
    if (this.selectionMetaElement) {
      this.selectionMetaElement.textContent = text;
    }
  }
}
