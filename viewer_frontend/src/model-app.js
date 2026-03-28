import * as THREE from "three";
import * as OBC from "@thatopen/components";
import * as OBCF from "@thatopen/components-front";
import * as FRAGS from "@thatopen/fragments";

const LOCAL_WASM_PATH = "/static/vendor/web-ifc/";
const FRAGMENTS_WORKER_URL = "/static/vendor/thatopen/worker.mjs";
const VIEWER_REFRESH_SIGNAL_KEY = "ragtag-viewer-refresh-signal";
const SELECTION_STYLE = "ragtag-selection";
const MODEL_CACHE_DB_NAME = "ragtag-model-cache";
const MODEL_CACHE_DB_VERSION = 1;
const MODEL_CACHE_STORE_NAME = "fragments-models";

const pageStatus = document.getElementById("model-page-status");
const toolbarStatus = document.getElementById("model-toolbar-status");
const modelStage = document.getElementById("model-stage");
const modelViewport = document.getElementById("model-viewport");
const modelEmptyState = document.getElementById("model-empty-state");
const modelRightSidebar = document.getElementById("model-right-sidebar");
const modelSearchPanel = document.getElementById("model-search-panel");
const toggleSearchPanelButton = document.getElementById("toggle-model-search-panel");
const toggleInspectorPanelButton = document.getElementById(
  "toggle-model-inspector-panel"
);
const modelToolResetButton = document.getElementById("model-tool-reset");
const modelToolFullscreenButton = document.getElementById("model-tool-fullscreen");
const searchForm = document.getElementById("model-search-form");
const searchInput = document.getElementById("model-search-input");
const searchClassFilter = document.getElementById("model-search-class-filter");
const searchPageSize = document.getElementById("model-search-page-size");
const searchResetButton = document.getElementById("model-search-reset");
const searchResultsMeta = document.getElementById("model-search-results-meta");
const searchPrevPageButton = document.getElementById("model-search-prev-page");
const searchNextPageButton = document.getElementById("model-search-next-page");
const searchPageStatus = document.getElementById("model-search-page-status");
const searchResults = document.getElementById("model-search-results");
const selectionPanel = document.getElementById("model-selection-panel");
const selectionCaption = document.getElementById("model-selection-caption");
const selectionDetails = document.getElementById("model-selection-details");
const ifcUploadInput = document.getElementById("ifc-upload-input");
const ifcUploadTrigger = document.getElementById("ifc-upload-trigger");
const ifcUploadStatus = document.getElementById("ifc-upload-status");
const ifcUploadProgressRow = document.getElementById("ifc-upload-progress-row");
const ifcUploadProgress = document.getElementById("ifc-upload-progress");
const ifcUploadProgressValue = document.getElementById("ifc-upload-progress-value");

const DEFAULT_UPLOAD_STATUS =
  "Upload an IFC to rebuild both the graph and 3D model views.";

let bootstrapPayload = null;
let ifcUploadBusy = false;
let components = null;
let world = null;
let fragments = null;
let ifcLoader = null;
let highlighter = null;
let currentModel = null;
let searchPanelVisible = true;
let inspectorPanelVisible = true;
let searchState = {
  query: "",
  className: "",
  page: 1,
  pageSize: Number(searchPageSize?.value) || 25,
  total: 0,
  totalPages: 0,
};

function broadcastViewerRefresh(detail = {}) {
  try {
    localStorage.setItem(
      VIEWER_REFRESH_SIGNAL_KEY,
      JSON.stringify({
        ...detail,
        pathname: window.location.pathname,
        timestamp: Date.now(),
      })
    );
  } catch (error) {
    // Ignore storage write failures and continue with the local refresh.
  }
}

function installViewerRefreshListener() {
  window.addEventListener("storage", (event) => {
    if (event.key !== VIEWER_REFRESH_SIGNAL_KEY || !event.newValue || ifcUploadBusy) {
      return;
    }
    window.location.reload();
  });
}

function pretty(value) {
  return JSON.stringify(value, null, 2);
}

async function api(path, options = {}) {
  const headers = new Headers(options.headers || {});
  headers.set("Accept", "application/json");
  if (options.body && !headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }
  const response = await fetch(path, {
    ...options,
    headers,
    cache: "no-store",
  });
  if (!response.ok) {
    const contentType = response.headers.get("content-type") || "";
    if (contentType.includes("application/json")) {
      const payload = await response.json().catch(() => null);
      const detail = payload?.detail ?? payload?.error ?? payload;
      throw new Error(
        typeof detail === "string"
          ? detail
          : pretty(detail || `Request failed with ${response.status}`)
      );
    }
    throw new Error((await response.text()) || `Request failed with ${response.status}`);
  }
  return response.json();
}

function setToolbarStatus(message) {
  if (toolbarStatus) {
    toolbarStatus.textContent = message;
    return;
  }
  if (pageStatus) {
    pageStatus.textContent = message;
  }
}

function setPageStatus(message) {
  if (pageStatus) {
    pageStatus.textContent = message;
  }
}

function setEmptyState(visible) {
  if (modelEmptyState) {
    modelEmptyState.hidden = !visible;
  }
}

function setSelectionEmptyState(empty, caption, details) {
  selectionPanel?.classList.toggle("is-empty", empty);
  if (selectionDetails) {
    selectionDetails.classList.toggle("inspector-empty", empty);
    selectionDetails.textContent = details;
  }
  if (selectionCaption) {
    selectionCaption.textContent = caption;
  }
}

function syncPanelToggleButton(button, visible) {
  if (!button) {
    return;
  }
  button.classList.toggle("active", visible);
  button.setAttribute("aria-pressed", visible ? "true" : "false");
}

function setChipToggleLabel(button, label, visible) {
  if (!button) {
    return;
  }
  button.textContent = `${label}: ${visible ? "On" : "Off"}`;
}

function setSearchPanelVisible(visible) {
  searchPanelVisible = visible;
  if (modelSearchPanel) {
    modelSearchPanel.hidden = !visible;
  }
  syncPanelToggleButton(toggleSearchPanelButton, visible);
  setChipToggleLabel(toggleSearchPanelButton, "Explorer", visible);
  if (modelRightSidebar) {
    modelRightSidebar.hidden = !(searchPanelVisible || inspectorPanelVisible);
  }
}

function setInspectorPanelVisible(visible) {
  inspectorPanelVisible = visible;
  if (selectionPanel) {
    selectionPanel.hidden = !visible;
  }
  syncPanelToggleButton(toggleInspectorPanelButton, visible);
  setChipToggleLabel(toggleInspectorPanelButton, "Inspector", visible);
  if (modelRightSidebar) {
    modelRightSidebar.hidden = !(searchPanelVisible || inspectorPanelVisible);
  }
}

function updateFullscreenButton() {
  const isFullscreen = document.fullscreenElement === modelStage;
  modelToolFullscreenButton?.classList.toggle("active", isFullscreen);
}

function resizeModelViewport() {
  try {
    world?.renderer?.resize?.();
  } catch (error) {
    // Ignore resize failures and let the viewer keep the current frame.
  }
}

function setUploadProgress(progressValue = null) {
  const hasProgress =
    Number.isFinite(progressValue) && progressValue >= 0 && progressValue <= 100;
  if (ifcUploadProgressRow) {
    ifcUploadProgressRow.hidden = !hasProgress;
  }
  if (!hasProgress) {
    if (ifcUploadProgress) {
      ifcUploadProgress.value = 0;
    }
    if (ifcUploadProgressValue) {
      ifcUploadProgressValue.textContent = "";
    }
    return;
  }
  const clamped = Math.max(0, Math.min(100, Math.round(progressValue)));
  if (ifcUploadProgress) {
    ifcUploadProgress.value = clamped;
  }
  if (ifcUploadProgressValue) {
    ifcUploadProgressValue.textContent = `${clamped}%`;
  }
}

function setUploadBusy(busy, statusText = null, progressValue = null) {
  ifcUploadBusy = busy;
  if (ifcUploadTrigger) {
    ifcUploadTrigger.disabled = busy;
    ifcUploadTrigger.textContent = busy ? "Building..." : "Upload IFC";
  }
  if (statusText !== null && ifcUploadStatus) {
    ifcUploadStatus.textContent = statusText;
  }
  setUploadProgress(progressValue);
}

function wait(ms) {
  return new Promise((resolve) => {
    window.setTimeout(resolve, ms);
  });
}

function supportsModelCache() {
  return typeof window !== "undefined" && "indexedDB" in window;
}

function openModelCacheDb() {
  return new Promise((resolve, reject) => {
    if (!supportsModelCache()) {
      resolve(null);
      return;
    }

    const request = window.indexedDB.open(
      MODEL_CACHE_DB_NAME,
      MODEL_CACHE_DB_VERSION
    );
    request.addEventListener("upgradeneeded", () => {
      const db = request.result;
      if (!db.objectStoreNames.contains(MODEL_CACHE_STORE_NAME)) {
        db.createObjectStore(MODEL_CACHE_STORE_NAME, {
          keyPath: "key",
        });
      }
    });
    request.addEventListener("success", () => {
      resolve(request.result);
    });
    request.addEventListener("error", () => {
      reject(request.error || new Error("Failed to open the model cache."));
    });
  });
}

async function withModelCacheStore(mode, callback) {
  const db = await openModelCacheDb();
  if (!db) {
    return null;
  }

  return new Promise((resolve, reject) => {
    const transaction = db.transaction(MODEL_CACHE_STORE_NAME, mode);
    const store = transaction.objectStore(MODEL_CACHE_STORE_NAME);

    transaction.addEventListener("complete", () => {
      db.close();
    });
    transaction.addEventListener("abort", () => {
      db.close();
      reject(transaction.error || new Error("The model cache transaction failed."));
    });
    transaction.addEventListener("error", () => {
      db.close();
      reject(transaction.error || new Error("The model cache transaction failed."));
    });

    Promise.resolve(callback(store)).then(resolve, reject);
  });
}

async function readCachedFragments(cacheKey) {
  if (!cacheKey) {
    return null;
  }
  const record = await withModelCacheStore("readonly", (store) => {
    return new Promise((resolve, reject) => {
      const request = store.get(cacheKey);
      request.addEventListener("success", () => {
        resolve(request.result || null);
      });
      request.addEventListener("error", () => {
        reject(
          request.error || new Error("Failed to read the cached fragments model.")
        );
      });
    });
  });
  const buffer = record?.buffer;
  if (buffer instanceof ArrayBuffer) {
    return buffer;
  }
  if (ArrayBuffer.isView(buffer)) {
    return buffer.buffer.slice(
      buffer.byteOffset,
      buffer.byteOffset + buffer.byteLength
    );
  }
  return null;
}

async function writeCachedFragments(cacheKey, buffer, metadata = {}) {
  if (!cacheKey || !(buffer instanceof ArrayBuffer)) {
    return false;
  }
  await withModelCacheStore("readwrite", (store) => {
    return new Promise((resolve, reject) => {
      const request = store.put({
        key: cacheKey,
        buffer,
        updatedAt: Date.now(),
        ...metadata,
      });
      request.addEventListener("success", () => {
        resolve(true);
      });
      request.addEventListener("error", () => {
        reject(
          request.error || new Error("Failed to save the cached fragments model.")
        );
      });
    });
  });
  return true;
}

async function deleteCachedFragments(cacheKey) {
  if (!cacheKey) {
    return;
  }
  await withModelCacheStore("readwrite", (store) => {
    return new Promise((resolve, reject) => {
      const request = store.delete(cacheKey);
      request.addEventListener("success", () => {
        resolve(true);
      });
      request.addEventListener("error", () => {
        reject(
          request.error || new Error("Failed to delete the cached fragments model.")
        );
      });
    });
  });
}

function hasImportJobId(payload) {
  return Boolean(payload && typeof payload.job_id === "string" && payload.job_id);
}

function isLegacyCompletedImport(payload) {
  return Boolean(
    payload &&
      typeof payload.dataset === "string" &&
      typeof payload.message === "string"
  );
}

function getFirstLocalId(modelIdMap) {
  for (const localIds of Object.values(modelIdMap || {})) {
    for (const localId of localIds || []) {
      return Number(localId);
    }
  }
  return null;
}

function hasSearchableDataset() {
  if (!bootstrapPayload) {
    return false;
  }
  const graph = bootstrapPayload.graph || {};
  return Number(graph.node_count) > 0 || Number(graph.edge_count) > 0;
}

function updateSearchPagination(payload = {}) {
  if (!searchPrevPageButton || !searchNextPageButton || !searchPageStatus) {
    return;
  }
  const page = payload.page || searchState.page || 1;
  const totalPages = payload.total_pages || searchState.totalPages || 0;
  searchPrevPageButton.disabled = !Boolean(payload.has_previous);
  searchNextPageButton.disabled = !Boolean(payload.has_next);
  searchPageStatus.textContent = totalPages
    ? `Page ${page} of ${totalPages}`
    : "No pages";
}

function renderSearchMeta(payload) {
  if (!searchResultsMeta) {
    return;
  }
  const parts = [`${payload.total} matches`];
  if (payload.query) {
    parts.push(`query: "${payload.query}"`);
  } else {
    parts.push("browse mode");
  }
  if (payload.class_name) {
    parts.push(`class: ${payload.class_name}`);
  }
  parts.push(`showing ${payload.results.length} on this page`);
  searchResultsMeta.textContent = parts.join(" | ");
}

function populateSearchClassOptions(classOptions, selectedValue) {
  if (!searchClassFilter) {
    return;
  }
  searchClassFilter.innerHTML = "";

  const allOption = document.createElement("option");
  allOption.value = "";
  allOption.textContent = "All classes";
  searchClassFilter.appendChild(allOption);

  for (const option of classOptions || []) {
    const element = document.createElement("option");
    element.value = option.value;
    element.textContent = `${option.label} (${option.count})`;
    searchClassFilter.appendChild(element);
  }

  searchClassFilter.value = selectedValue || "";
}

function renderSearchUnavailableState(
  message = "Upload an IFC to search the model dataset."
) {
  if (!searchResults || !searchResultsMeta) {
    return;
  }
  populateSearchClassOptions([], "");
  searchResults.innerHTML = "";
  searchResults.textContent = message;
  searchResultsMeta.textContent = "No searchable dataset is active.";
  updateSearchPagination({ page: 1, total_pages: 0 });
}

function showElementDetails(element, nodeId = null) {
  const resolvedId = nodeId || element.id || "no id";
  const title = element.label || resolvedId || "Selected element";
  const className = element.class_ || "Unknown";
  setSelectionEmptyState(
    false,
    `${title} | ${className} | ${resolvedId}`,
    pretty(element)
  );
}

async function resolveSelectionModelIdMap(element) {
  if (!currentModel) {
    throw new Error("The IFC model is not loaded yet.");
  }

  const properties = element?.properties || {};
  const localIds = new Set();
  const expressId = Number(properties.ExpressId);
  if (Number.isInteger(expressId) && expressId > 0) {
    localIds.add(expressId);
  }

  const guid = typeof properties.GlobalId === "string" ? properties.GlobalId : null;
  if (!localIds.size && guid && typeof currentModel.getLocalIdsByGuids === "function") {
    const resolvedIds = await currentModel.getLocalIdsByGuids([guid]);
    for (const localId of resolvedIds || []) {
      if (Number.isInteger(localId) && localId > 0) {
        localIds.add(localId);
      }
    }
  }

  if (!localIds.size || !currentModel.modelId) {
    throw new Error("No IFC local IDs were available for this element.");
  }

  return { [currentModel.modelId]: localIds };
}

async function focusSearchResult(result) {
  try {
    const payload = await api(`/api/element/${encodeURIComponent(result.id)}`);
    const element = payload.element || payload;

    if (!highlighter || !currentModel) {
      showElementDetails(element, result.id);
      setToolbarStatus("Model search ready | showing dataset details only");
      return;
    }

    const modelIdMap = await resolveSelectionModelIdMap(element);
    await highlighter.highlightByID(SELECTION_STYLE, modelIdMap, true, true);
  } catch (error) {
    setSelectionEmptyState(
      false,
      `${result.label || result.id} | failed to focus`,
      `Failed to focus ${result.id}\n${String(error)}`
    );
    setToolbarStatus("Model search failed to focus the selected IFC element");
  }
}

function renderSearchResults(payload) {
  if (!searchResults) {
    return;
  }
  searchResults.innerHTML = "";
  const results = payload.results || [];
  if (!results.length) {
    searchResults.textContent = payload.total
      ? "No results on this page."
      : "No matches found.";
    return;
  }

  for (const result of results) {
    const item = document.createElement("div");
    item.className = "search-result";

    const header = document.createElement("div");
    header.className = "search-result-header";

    const title = document.createElement("div");
    title.className = "search-result-title";
    title.textContent = result.label;

    const button = document.createElement("button");
    button.type = "button";
    button.className = "secondary";
    button.textContent = "Focus";
    button.addEventListener("click", async () => {
      await focusSearchResult(result);
    });

    header.appendChild(title);
    header.appendChild(button);
    item.appendChild(header);

    const meta = document.createElement("div");
    meta.className = "search-result-meta";
    meta.textContent = `${result.class_name || "Unknown"} - ${result.id}`;
    item.appendChild(meta);

    if (Array.isArray(result.geometry)) {
      const geometry = document.createElement("div");
      geometry.className = "search-result-geometry";
      geometry.textContent =
        `Center: ${result.geometry.map((value) => value.toFixed(2)).join(", ")}`;
      item.appendChild(geometry);
    }

    searchResults.appendChild(item);
  }
}

async function runSearch({ resetPage = false, page = null } = {}) {
  if (!searchInput || !searchClassFilter || !searchPageSize) {
    return;
  }
  if (!hasSearchableDataset()) {
    renderSearchUnavailableState();
    return;
  }

  const query = searchInput.value.trim();
  const className = searchClassFilter.value.trim();
  const pageSize = Number(searchPageSize.value) || searchState.pageSize || 25;
  const nextPage = resetPage ? 1 : page || searchState.page || 1;
  const params = new URLSearchParams({
    page: String(nextPage),
    page_size: String(pageSize),
  });
  if (query) {
    params.set("q", query);
  }
  if (className) {
    params.set("class_name", className);
  }

  if (searchResults) {
    searchResults.textContent = "Searching...";
  }
  if (searchResultsMeta) {
    searchResultsMeta.textContent = "Searching node browser...";
  }
  updateSearchPagination({ page: nextPage, total_pages: 0 });

  try {
    const payload = await api(`/api/search?${params.toString()}`);
    searchState = {
      query: payload.query || "",
      className: payload.class_name || "",
      page: payload.page || 1,
      pageSize: payload.page_size || pageSize,
      total: payload.total || 0,
      totalPages: payload.total_pages || 0,
    };
    searchInput.value = searchState.query;
    searchPageSize.value = String(searchState.pageSize);
    populateSearchClassOptions(payload.class_options, searchState.className);
    renderSearchMeta(payload);
    renderSearchResults(payload);
    updateSearchPagination(payload);
  } catch (error) {
    if (searchResults) {
      searchResults.textContent = String(error);
    }
    if (searchResultsMeta) {
      searchResultsMeta.textContent = "Search failed.";
    }
    updateSearchPagination({ page: 1, total_pages: 0 });
  }
}

async function ensureViewerReady() {
  if (world) {
    return;
  }

  components = new OBC.Components();
  const worlds = components.get(OBC.Worlds);
  world = worlds.create();
  world.scene = new OBC.SimpleScene(components);
  world.renderer = new OBC.SimpleRenderer(components, modelViewport);
  world.camera = new OBC.SimpleCamera(components);
  components.init();
  world.scene.setup();
  world.scene.three.background = new THREE.Color("#0b1119");
  world.camera.controls.setLookAt(14, 10, 12, 0, 0, 0);

  const grids = components.get(OBC.Grids);
  grids.create(world);

  const raycasters = components.get(OBC.Raycasters);
  raycasters.get(world);

  fragments = components.get(OBC.FragmentsManager);
  fragments.init(FRAGMENTS_WORKER_URL);
  ifcLoader = components.get(OBC.IfcLoader);

  highlighter = components.get(OBCF.Highlighter);
  highlighter.setup({
    world,
    selectName: SELECTION_STYLE,
    selectEnabled: true,
    autoHighlightOnClick: true,
    autoUpdateFragments: true,
    selectionColor: null,
    selectMaterialDefinition: {
      color: new THREE.Color("#7da8ff"),
      renderedFaces: FRAGS.RenderedFaces.TWO,
      opacity: 1,
      transparent: false,
      preserveOriginalMaterial: true,
      depthTest: true,
    },
  });
  highlighter.zoomToSelection = false;
  highlighter.autoToggle.add(SELECTION_STYLE);

  const selectEvents = highlighter.events[SELECTION_STYLE];
  selectEvents.onHighlight.add(async (modelIdMap) => {
    setToolbarStatus("Selection updated. Fetching normalized element details...");
    await renderSelectionDetails(modelIdMap);
  });
  selectEvents.onClear.add(() => {
    setSelectionEmptyState(
      true,
      "Select an IFC element in the model or from search to inspect its properties.",
      "Select an IFC element in the model or from search to inspect its properties."
    );
    setToolbarStatus(
      currentModel
        ? "Model ready | Orbit, pan, zoom, and click elements to inspect"
        : "Preparing model viewer..."
    );
  });
}

async function fitModelToView() {
  if (!currentModel || !world) {
    return;
  }

  const box = currentModel.box || new THREE.Box3().setFromObject(currentModel.object);
  if (box.isEmpty()) {
    return;
  }
  const sphere = box.getBoundingSphere(new THREE.Sphere());
  const controls = world.camera.controls;
  if (typeof controls.fitToSphere === "function") {
    await controls.fitToSphere(sphere, true);
    return;
  }
  const distance = Math.max(sphere.radius * 2.8, 8);
  await controls.setLookAt(
    sphere.center.x + distance,
    sphere.center.y + distance,
    sphere.center.z + distance,
    sphere.center.x,
    sphere.center.y,
    sphere.center.z,
    true
  );
}

function buildModelRuntimeId(payload) {
  const source =
    payload?.model_fragments_cache_key ||
    payload?.model_ifc_name ||
    payload?.selected_dataset ||
    "ragtag-model";
  const normalized = String(source)
    .trim()
    .replace(/[^A-Za-z0-9._-]+/g, "-")
    .replace(/^-+|-+$/g, "");
  return normalized || "ragtag-model";
}

async function finalizeLoadedModel(datasetName, sourceLabel) {
  if (typeof currentModel.useCamera === "function") {
    currentModel.useCamera(world.camera.three);
  }
  const modelObject = currentModel.object || currentModel;
  world.scene.three.add(modelObject);
  await fragments.core.update(true);
  resizeModelViewport();
  await fitModelToView();

  setPageStatus(
    `${datasetName} loaded. Orbit, pan, zoom, and click IFC elements to inspect them.`
  );
  setToolbarStatus(
    `${bootstrapPayload.model_ifc_name || "IFC model"} ready | loaded from ${sourceLabel} | ${bootstrapPayload.graph.node_count} graph nodes available in the paired runtime`
  );
}

async function cacheCurrentModelFragments(cacheKey) {
  if (!cacheKey || !currentModel || typeof currentModel.getBuffer !== "function") {
    return false;
  }
  try {
    const buffer = await currentModel.getBuffer(false);
    return await writeCachedFragments(cacheKey, buffer, {
      dataset: bootstrapPayload?.selected_dataset || null,
      ifcName: bootstrapPayload?.model_ifc_name || null,
    });
  } catch (error) {
    return false;
  }
}

async function fetchElementPayload(modelIdMap) {
  const localId = getFirstLocalId(modelIdMap);
  if (localId !== null) {
    try {
      return await api(`/api/element/${encodeURIComponent(String(localId))}`);
    } catch (error) {
      // Fall back to GUID lookup below.
    }
  }

  if (!fragments) {
    throw new Error("Fragments runtime is not ready.");
  }

  const guids = await fragments.modelIdMapToGuids(modelIdMap);
  const guid = guids.find((value) => typeof value === "string" && value);
  if (guid) {
    return api(`/api/element/${encodeURIComponent(guid)}`);
  }

  throw new Error("No ExpressId or GlobalId was available for the selected element.");
}

function summarizeSelection(modelIdMap) {
  const items = [];
  for (const [modelId, localIds] of Object.entries(modelIdMap || {})) {
    items.push(`${modelId}: ${Array.from(localIds || []).join(", ")}`);
  }
  return items.join(" | ");
}

async function renderSelectionDetails(modelIdMap) {
  try {
    const payload = await fetchElementPayload(modelIdMap);
    showElementDetails(payload.element || payload);
    setToolbarStatus("Selection ready | inspect details in the side panel");
  } catch (error) {
    setSelectionEmptyState(
      false,
      "Selection could not be resolved through the backend payload",
      [
        "Selected model items:",
        summarizeSelection(modelIdMap),
        "",
        `Lookup error: ${String(error)}`,
      ].join("\n")
    );
    setToolbarStatus("Selection captured, but property lookup fell back to raw IDs");
  }
}

async function loadIfcModel() {
  bootstrapPayload = await api("/api/model/bootstrap");
  await runSearch({ resetPage: true });

  const datasetName = bootstrapPayload.selected_dataset || "No active dataset";
  const cacheKey = bootstrapPayload.model_fragments_cache_key;
  const modelRuntimeId = buildModelRuntimeId(bootstrapPayload);

  if (!bootstrapPayload.model_ifc_available || !bootstrapPayload.model_ifc_url) {
    setEmptyState(true);
    setPageStatus(
      "No raw IFC file is currently available. Upload one or start the viewer with --ifc."
    );
    setToolbarStatus("3D model unavailable until an IFC source is attached");
    return;
  }

  await ensureViewerReady();
  setEmptyState(false);
  if (cacheKey) {
    try {
      const cachedBuffer = await readCachedFragments(cacheKey);
      if (cachedBuffer) {
        setPageStatus(
          `Loading cached 3D model for ${bootstrapPayload.model_ifc_name || "active IFC"}...`
        );
        setToolbarStatus("Loading cached fragments...");
        currentModel = await fragments.core.load(cachedBuffer, {
          modelId: modelRuntimeId,
          camera: world.camera.three,
        });
        await finalizeLoadedModel(datasetName, "cached fragments");
        return;
      }
    } catch (error) {
      await deleteCachedFragments(cacheKey).catch(() => {});
      setToolbarStatus("Cached fragments were unavailable. Rebuilding from IFC...");
    }
  }

  setPageStatus(
    `Loading ${bootstrapPayload.model_ifc_name || "active IFC"} into the embedded model viewer...`
  );
  setToolbarStatus("Fetching IFC bytes...");

  const ifcResponse = await fetch(bootstrapPayload.model_ifc_url, {
    cache: "no-store",
  });
  if (!ifcResponse.ok) {
    throw new Error(`Failed to fetch IFC source (${ifcResponse.status}).`);
  }
  const ifcBuffer = new Uint8Array(await ifcResponse.arrayBuffer());

  ifcLoader.settings.webIfc.COORDINATE_TO_ORIGIN = true;
  await ifcLoader.setup({
    autoSetWasm: false,
    wasm: {
      path: LOCAL_WASM_PATH,
      absolute: true,
    },
  });

  setToolbarStatus("Converting IFC to fragments...");
  currentModel = await ifcLoader.load(
    ifcBuffer,
    true,
    modelRuntimeId
  );
  await finalizeLoadedModel(datasetName, "fresh IFC conversion");

  if (cacheKey) {
    setToolbarStatus(
      `${bootstrapPayload.model_ifc_name || "IFC model"} ready | caching fragments for faster next load...`
    );
    const cached = await cacheCurrentModelFragments(cacheKey);
    if (cached) {
      setToolbarStatus(
        `${bootstrapPayload.model_ifc_name || "IFC model"} ready | cached for faster reopening | ${bootstrapPayload.graph.node_count} graph nodes available in the paired runtime`
      );
      return;
    }
    await deleteCachedFragments(cacheKey).catch(() => {});
    setToolbarStatus(
      `${bootstrapPayload.model_ifc_name || "IFC model"} ready | cache unavailable in this browser session`
    );
  }
}

function startIfcImportJob(file) {
  return new Promise((resolve, reject) => {
    const formData = new FormData();
    formData.append("ifc_file", file);

    const request = new XMLHttpRequest();
    request.open("POST", "/api/import-ifc");
    request.responseType = "json";

    request.upload.addEventListener("progress", (event) => {
      if (!event.lengthComputable || event.total <= 0) {
        setUploadBusy(true, `Uploading ${file.name}...`, 5);
        return;
      }
      const uploadProgress = Math.max(
        1,
        Math.min(10, Math.round((event.loaded / event.total) * 10))
      );
      setUploadBusy(
        true,
        `Uploading ${file.name}... ${Math.round((event.loaded / event.total) * 100)}%`,
        uploadProgress
      );
    });

    request.addEventListener("load", () => {
      const contentType = request.getResponseHeader("content-type") || "";
      const payload =
        request.response && typeof request.response === "object"
          ? request.response
          : contentType.includes("application/json")
            ? JSON.parse(request.responseText || "{}")
            : null;

      if (request.status < 200 || request.status >= 300) {
        const detail = payload?.detail ?? payload?.error ?? request.responseText;
        reject(new Error(typeof detail === "string" ? detail : pretty(detail)));
        return;
      }

      resolve(payload);
    });

    request.addEventListener("error", () => {
      reject(new Error("IFC upload failed."));
    });

    request.send(formData);
  });
}

async function pollImportJob(jobId) {
  for (;;) {
    const payload = await api(`/api/import-ifc/${encodeURIComponent(jobId)}`);
    setUploadBusy(true, payload.message, payload.progress);
    if (payload.status === "completed") {
      return payload;
    }
    if (payload.status === "failed") {
      throw new Error(payload.error || payload.message || "IFC import failed.");
    }
    await wait(600);
  }
}

async function uploadIfcFile(file) {
  if (!file || ifcUploadBusy) {
    return;
  }
  setUploadBusy(true, `Uploading ${file.name}...`, 1);
  try {
    const importResponse = await startIfcImportJob(file);
    if (hasImportJobId(importResponse)) {
      setUploadBusy(
        true,
        importResponse.message || `Building ${file.name}...`,
        importResponse.progress
      );
      const payload = await pollImportJob(importResponse.job_id);
      broadcastViewerRefresh({
        dataset: payload.dataset || file.name,
        source: "model",
      });
      setUploadBusy(
        true,
        `Loaded ${payload.dataset || file.name}. Refreshing viewer...`,
        100
      );
      await wait(400);
      window.location.reload();
      return;
    }

    if (isLegacyCompletedImport(importResponse)) {
      broadcastViewerRefresh({
        dataset: importResponse.dataset || file.name,
        source: "model",
      });
      setUploadBusy(
        true,
        `Loaded ${importResponse.dataset || file.name}. Refreshing viewer...`,
        100
      );
      await wait(400);
      window.location.reload();
      return;
    }

    throw new Error(
      "IFC import returned an unexpected response shape. Restart the viewer server and try again."
    );
  } catch (error) {
    setUploadBusy(false, `IFC import failed: ${String(error)}`, null);
    if (ifcUploadInput) {
      ifcUploadInput.value = "";
    }
  }
}

ifcUploadTrigger?.addEventListener("click", () => {
  ifcUploadInput?.click();
});

ifcUploadInput?.addEventListener("change", async () => {
  const [file] = Array.from(ifcUploadInput.files || []);
  if (!file) {
    return;
  }
  await uploadIfcFile(file);
});

modelToolResetButton?.addEventListener("click", async () => {
  await fitModelToView();
  setToolbarStatus("Camera fit refreshed for the active IFC model");
});

modelToolFullscreenButton?.addEventListener("click", async () => {
  if (!document.fullscreenEnabled || !modelStage) {
    return;
  }
  try {
    if (document.fullscreenElement === modelStage) {
      await document.exitFullscreen();
    } else {
      await modelStage.requestFullscreen();
    }
  } catch (error) {
    setToolbarStatus("Fullscreen unavailable for the current browser session");
  }
});

toggleSearchPanelButton?.addEventListener("click", () => {
  setSearchPanelVisible(!searchPanelVisible);
});

toggleInspectorPanelButton?.addEventListener("click", () => {
  setInspectorPanelVisible(!inspectorPanelVisible);
});

setSearchPanelVisible(true);
setInspectorPanelVisible(true);
setUploadBusy(false, DEFAULT_UPLOAD_STATUS, null);
setSelectionEmptyState(
  true,
  "Select an IFC element in the model or from search to inspect its properties.",
  "Select an IFC element in the model or from search to inspect its properties."
);
installViewerRefreshListener();
updateFullscreenButton();

if (!document.fullscreenEnabled && modelToolFullscreenButton) {
  modelToolFullscreenButton.hidden = true;
}

window.addEventListener("resize", () => {
  resizeModelViewport();
});

document.addEventListener("fullscreenchange", () => {
  updateFullscreenButton();
  resizeModelViewport();
});

searchForm?.addEventListener("submit", async (event) => {
  event.preventDefault();
  await runSearch({ resetPage: true });
});

searchClassFilter?.addEventListener("change", async () => {
  await runSearch({ resetPage: true });
});

searchPageSize?.addEventListener("change", async () => {
  await runSearch({ resetPage: true });
});

searchResetButton?.addEventListener("click", async () => {
  if (searchInput) {
    searchInput.value = "";
  }
  if (searchClassFilter) {
    searchClassFilter.value = "";
  }
  if (searchPageSize) {
    searchPageSize.value = "25";
  }
  await runSearch({ resetPage: true });
});

searchPrevPageButton?.addEventListener("click", async () => {
  if (searchState.page <= 1) {
    return;
  }
  await runSearch({ page: searchState.page - 1 });
});

searchNextPageButton?.addEventListener("click", async () => {
  if (searchState.page >= searchState.totalPages) {
    return;
  }
  await runSearch({ page: searchState.page + 1 });
});

loadIfcModel().catch((error) => {
  setEmptyState(true);
  setPageStatus(`Model viewer unavailable: ${String(error)}`);
  setToolbarStatus("3D model loading failed");
});
