import { WebGLGraphView } from "/static/graph3d.js";

const VIEWER_REFRESH_SIGNAL_KEY = "ragtag-viewer-refresh-signal";

const healthPill = document.getElementById("health-pill");
const graphStage = document.getElementById("graph-stage");
const graphCanvas = document.getElementById("graph3d-canvas");
const graphOverlay = document.getElementById("graph3d-overlay");
const graphToolbarStatus = document.getElementById("graph-toolbar-status");
const graphEmptyState = document.getElementById("graph-empty-state");
const graphEmptyTitle = document.getElementById("graph-empty-title");
const graphEmptyMessage = document.getElementById("graph-empty-message");
const graphEmptyHint = document.getElementById("graph-empty-hint");
const ifcUploadInput = document.getElementById("ifc-upload-input");
const ifcUploadTrigger = document.getElementById("ifc-upload-trigger");
const ifcUploadStatus = document.getElementById("ifc-upload-status");
const ifcUploadProgressRow = document.getElementById("ifc-upload-progress-row");
const ifcUploadProgress = document.getElementById("ifc-upload-progress");
const ifcUploadProgressValue = document.getElementById("ifc-upload-progress-value");
const graphModeButtons = Array.from(
  document.querySelectorAll("[data-graph-mode]")
);
const toggleEdgeLabelsButton = document.getElementById("toggle-edge-labels");
const toggleBBoxesButton = document.getElementById("toggle-bboxes");
const toggleMeshesButton = document.getElementById("toggle-meshes");
const toggleFullscreenButton = document.getElementById("toggle-fullscreen");
const toggleLegendButton = document.getElementById("toggle-legend");
const toggleSearchPanelButton = document.getElementById("toggle-search-panel");
const toggleInspectorPanelButton = document.getElementById(
  "toggle-inspector-panel"
);
const legendSidebar = document.getElementById("legend-sidebar");
const rightSidebar = document.getElementById("right-sidebar");
const searchPanel = document.getElementById("search-panel");
const legendTotalEdges = document.getElementById("legend-total-edges");
const graphLegendEdges = document.getElementById("graph-legend-edges");
const graphLegendNodes = document.getElementById("graph-legend-nodes");
const legendRelationsPage = document.getElementById("legend-relations-page");
const legendNodesPage = document.getElementById("legend-nodes-page");
const legendPageButtons = Array.from(
  document.querySelectorAll("[data-legend-page]")
);
const graphToolResetButton = document.getElementById("graph-tool-reset");
const graphToolFullscreenButton = document.getElementById("graph-tool-fullscreen");
const graphToolLegacyLink = document.getElementById("graph-tool-legacy");
const openModelViewerLink = document.getElementById("open-model-viewer");
const queryForm = document.getElementById("query-form");
const queryInput = document.getElementById("query-input");
const queryResult = document.getElementById("query-result");
const querySubmitButton = queryForm.querySelector('button[type="submit"]');
const queryStatus = document.getElementById("query-status");
const queryDock = document.getElementById("query-dock");
const queryToolFullscreenButton = document.getElementById(
  "query-tool-fullscreen"
);
const queryToggleDetailsButton = document.getElementById("query-toggle-details");
const queryCopyTranscriptButton = document.getElementById("query-copy-transcript");
const searchForm = document.getElementById("search-form");
const searchInput = document.getElementById("search-input");
const searchClassFilter = document.getElementById("search-class-filter");
const searchPageSize = document.getElementById("search-page-size");
const searchResetButton = document.getElementById("search-reset");
const searchResultsMeta = document.getElementById("search-results-meta");
const searchPrevPageButton = document.getElementById("search-prev-page");
const searchNextPageButton = document.getElementById("search-next-page");
const searchPageStatus = document.getElementById("search-page-status");
const searchResults = document.getElementById("search-results");
const inspectorPanel = document.getElementById("inspector-panel");
const inspectorCaption = document.getElementById("inspector-caption");
const inspector = document.getElementById("inspector");

let selectedNodeId = null;
let bootstrapPayload = null;
let graphView = null;
let edgeLabelsEnabled = false;
let bboxesEnabled = false;
let meshesEnabled = false;
let legendVisible = false;
let searchPanelVisible = false;
let inspectorPanelVisible = false;
let ifcUploadBusy = false;
let legendNodeGroups = {};
let activeLegendPage = "relations";
let queryDetailsVisible = false;
let lastQueryRoute = "";
let lastQueryDurationMs = 0;
let queryTranscriptEntries = [];
const DEFAULT_UPLOAD_STATUS = "Build the viewer directly from an IFC file.";
const DEFAULT_QUERY_SUBMIT_LABEL = querySubmitButton.textContent || "Run Query";
let searchState = {
  query: "",
  className: "",
  page: 1,
  pageSize: Number(searchPageSize.value) || 25,
  total: 0,
  totalPages: 0,
};

function truncateText(text, maxLen) {
  const value = String(text || "");
  if (value.length <= maxLen) {
    return value;
  }
  if (maxLen <= 3) {
    return value.slice(0, maxLen);
  }
  return `${value.slice(0, maxLen - 3)}...`;
}

function buildQueryStatusText() {
  const dataset = bootstrapPayload?.selected_dataset || "No dataset";
  const parts = [`Dataset: ${dataset}`];
  if (lastQueryRoute) {
    parts.push(`Route: ${lastQueryRoute}`);
  }
  if (lastQueryDurationMs > 0) {
    parts.push(`${Math.round(lastQueryDurationMs)}ms`);
  }
  parts.push(`details:${queryDetailsVisible ? "on" : "off"}`);
  return parts.join(" | ");
}

function updateQueryStatus(text = null) {
  if (!queryStatus) {
    return;
  }
  queryStatus.textContent = text || buildQueryStatusText();
}

function syncQueryDetailsToggle() {
  if (!queryToggleDetailsButton) {
    return;
  }
  queryToggleDetailsButton.textContent = `Details: ${queryDetailsVisible ? "On" : "Off"}`;
  queryToggleDetailsButton.setAttribute(
    "aria-pressed",
    queryDetailsVisible ? "true" : "false"
  );
}

function applyQueryDetailsVisibility() {
  for (const block of queryResult.querySelectorAll(".query-details")) {
    block.classList.toggle("is-hidden", !queryDetailsVisible);
  }
  syncQueryDetailsToggle();
  updateQueryStatus();
}

function createTranscriptEntry(question) {
  const entry = {
    question,
    route: "",
    decision: "",
    answer: "",
    warning: "",
    error: "",
    sqlItems: [],
    sqlMoreCount: 0,
    graphSample: [],
    detailsJson: "",
    detailsTruncated: false,
    status: "pending",
  };
  queryTranscriptEntries.push(entry);
  return entry;
}

function buildTranscriptMarkdown() {
  const blocks = [];
  for (const entry of queryTranscriptEntries) {
    blocks.push(`**Q:** ${entry.question}`);

    if (entry.status === "pending") {
      blocks.push("   working...");
      continue;
    }

    if (entry.route || entry.decision) {
      blocks.push(`   [${entry.route || "?"}] ${entry.decision || ""}`.trimEnd());
    }

    if (entry.error) {
      blocks.push(`Error: ${entry.error}`);
      blocks.push("---");
      continue;
    }

    blocks.push("**A:**");
    blocks.push(entry.answer || "No answer produced.");

    if (entry.warning) {
      blocks.push(`Warning: ${entry.warning}`);
    }

    if (entry.sqlItems.length) {
      blocks.push(...entry.sqlItems.map((item) => `   ${item}`));
      if (entry.sqlMoreCount > 0) {
        blocks.push(`   (${entry.sqlMoreCount} more items not shown)`);
      }
    }

    if (entry.graphSample.length) {
      blocks.push("   Sample:");
      blocks.push(...entry.graphSample.map((item) => `   - ${item}`));
    }

    if (queryDetailsVisible && entry.detailsJson) {
      blocks.push(`\`\`\`json\n${entry.detailsJson}\n\`\`\``);
    }

    blocks.push("---");
  }
  return blocks.join("\n\n").trim();
}

async function copyTextToClipboard(text) {
  if (navigator.clipboard && window.isSecureContext) {
    await navigator.clipboard.writeText(text);
    return;
  }

  const helper = document.createElement("textarea");
  helper.value = text;
  helper.setAttribute("readonly", "");
  helper.style.position = "fixed";
  helper.style.opacity = "0";
  helper.style.pointerEvents = "none";
  document.body.appendChild(helper);
  helper.select();
  document.execCommand("copy");
  helper.remove();
}

function flashQueryStatus(message) {
  updateQueryStatus(message);
  window.setTimeout(() => updateQueryStatus(), 1600);
}

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

async function api(path, options = {}) {
  const headers = new Headers(options.headers || {});
  headers.set("Accept", "application/json");
  if (options.body && !headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }
  const response = await fetch(path, {
    ...options,
    cache: "no-store",
    headers,
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
    const text = await response.text();
    throw new Error(text || `Request failed with ${response.status}`);
  }
  return response.json();
}

function pretty(value) {
  return JSON.stringify(value, null, 2);
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function renderInlineMarkdown(text) {
  return escapeHtml(text)
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
}

function markdownToHtml(markdown) {
  const source = String(markdown || "").trim();
  if (!source) {
    return "<p>No answer produced.</p>";
  }

  const blocks = source.split(/\n\s*\n/);
  return blocks
    .map((block) => {
      const trimmed = block.trim();
      if (!trimmed) {
        return "";
      }

      if (trimmed.startsWith("```") && trimmed.endsWith("```")) {
        const code = trimmed
          .replace(/^```[^\n]*\n?/, "")
          .replace(/\n?```$/, "");
        return `<pre><code>${escapeHtml(code)}</code></pre>`;
      }

      const lines = block.split("\n");
      if (lines.every((line) => line.trim().startsWith("- "))) {
        const items = lines
          .map((line) => line.trim().slice(2))
          .map((line) => `<li>${renderInlineMarkdown(line)}</li>`)
          .join("");
        return `<ul>${items}</ul>`;
      }

      return `<p>${lines
        .map((line) => renderInlineMarkdown(line))
        .join("<br>")}</p>`;
    })
    .join("");
}

function setHealth(online) {
  if (!healthPill) {
    return;
  }
  healthPill.textContent = online ? "Connected" : "Offline";
  healthPill.style.background = online
    ? "rgba(166, 227, 161, 0.12)"
    : "rgba(243, 139, 168, 0.16)";
  healthPill.style.borderColor = online
    ? "rgba(166, 227, 161, 0.42)"
    : "rgba(243, 139, 168, 0.4)";
  healthPill.style.color = online ? "#a6e3a1" : "#f38ba8";
}

function syncModelViewerLink() {
  if (!openModelViewerLink || !bootstrapPayload) {
    return;
  }
  openModelViewerLink.href = bootstrapPayload.model_viewer_url || "/model";
  openModelViewerLink.classList.toggle(
    "disabled",
    !bootstrapPayload.model_ifc_available
  );
  openModelViewerLink.setAttribute(
    "title",
    bootstrapPayload.model_ifc_available
      ? "Open the embedded IFC model viewer"
      : "Open the 3D model page. Upload an IFC or launch with --ifc to load geometry."
  );
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

function hasActiveGraphData(payload = bootstrapPayload) {
  if (!payload) {
    return false;
  }
  const graph = payload.graph || {};
  return Boolean(
    payload.webgl_graph_available ||
      payload.debug_graph_available ||
      Number(graph.node_count) > 0 ||
      Number(graph.edge_count) > 0
  );
}

function getLegendNodeEntries(nodeGroups = legendNodeGroups) {
  return Object.entries(nodeGroups || {});
}

function syncLegendPage(
  page = activeLegendPage,
  visibleEdgeCount = graphView?.getVisibleEdgeCount?.() || 0,
  nodeGroupCount = getLegendNodeEntries(legendNodeGroups).length
) {
  activeLegendPage = page;
  if (legendRelationsPage) {
    legendRelationsPage.hidden = page !== "relations";
  }
  if (legendNodesPage) {
    legendNodesPage.hidden = page !== "nodes";
  }
  for (const button of legendPageButtons) {
    const isActive = button.dataset.legendPage === page;
    button.classList.toggle("active", isActive);
    button.setAttribute("aria-pressed", isActive ? "true" : "false");
  }
  if (legendTotalEdges) {
    legendTotalEdges.textContent =
      page === "nodes"
        ? `Node groups: ${nodeGroupCount}`
        : `Visible edges: ${visibleEdgeCount}`;
  }
}

function renderLegend(nodeGroups = legendNodeGroups) {
  const relationEntries = graphView?.getRelationLegendEntries?.() || [];
  const visibleEdgeCount = graphView?.getVisibleEdgeCount?.() || 0;
  graphLegendEdges.innerHTML = "";
  for (const entry of relationEntries) {
    const item = document.createElement("button");
    item.type = "button";
    item.className = "legend-toggle";
    item.classList.toggle("active", entry.visible);
    item.classList.toggle("inactive", !entry.visible);
    item.classList.toggle("unavailable", !entry.available);
    item.disabled = !entry.available;
    item.innerHTML = `
      <span class="swatch line" style="--swatch:${entry.swatch}"></span>
      <span class="legend-item-text">
        <div class="legend-item-main">
          ${entry.label}
          <span class="legend-toggle-state">${entry.visible ? "Shown" : "Hidden"}</span>
        </div>
        <div class="legend-item-sub">${entry.subtitle}</div>
      </span>
      <span class="legend-count">${entry.count}</span>
    `;
    item.addEventListener("click", () => {
      if (!graphView) {
        return;
      }
      graphView.setRelationVisible(entry.relation, !entry.visible);
      renderLegend(nodeGroups);
      updateStageSummary();
    });
    graphLegendEdges.appendChild(item);
  }

  const nodeEntries = getLegendNodeEntries(nodeGroups);
  graphLegendNodes.innerHTML = "";
  for (const [groupName, group] of nodeEntries) {
    const item = document.createElement("button");
    item.type = "button";
    item.className = "legend-toggle legend-node-item";
    item.innerHTML = `
      <span class="swatch dot" style="--swatch:${group.color}"></span>
      <span class="legend-item-text">
        <div class="legend-item-main">
          ${group.label || groupName}
          <span class="legend-toggle-state">Focus</span>
        </div>
        <div class="legend-item-sub">${group.count} nodes</div>
      </span>
      <span class="legend-count">${group.count}</span>
    `;
    item.addEventListener("click", () => {
      if (!graphView) {
        return;
      }
      const focused = graphView.focusFirstNodeInClass(groupName);
      if (focused) {
        updateStageSummary();
      }
    });
    graphLegendNodes.appendChild(item);
  }

  syncLegendPage(activeLegendPage, visibleEdgeCount, nodeEntries.length);
}

function setLegendVisible(visible) {
  legendVisible = visible;
  if (legendSidebar) {
    legendSidebar.hidden = !visible;
  }
  syncPanelToggleButton(toggleLegendButton, visible);
  setChipToggleLabel(toggleLegendButton, "Legend", visible);
}

function setSearchPanelVisible(visible) {
  searchPanelVisible = visible;
  if (searchPanel) {
    searchPanel.hidden = !visible;
  }
  syncPanelToggleButton(toggleSearchPanelButton, visible);
  setChipToggleLabel(toggleSearchPanelButton, "Explorer", visible);
  if (rightSidebar) {
    rightSidebar.hidden = !(searchPanelVisible || inspectorPanelVisible);
  }
}

function setInspectorPanelVisible(visible) {
  inspectorPanelVisible = visible;
  if (inspectorPanel) {
    inspectorPanel.hidden = !visible;
  }
  syncPanelToggleButton(toggleInspectorPanelButton, visible);
  setChipToggleLabel(toggleInspectorPanelButton, "Inspector", visible);
  if (rightSidebar) {
    rightSidebar.hidden = !(searchPanelVisible || inspectorPanelVisible);
  }
}

function updateFullscreenButton() {
  const isFullscreen = document.fullscreenElement === graphStage;
  if (toggleFullscreenButton) {
    toggleFullscreenButton.classList.toggle("active", isFullscreen);
    toggleFullscreenButton.textContent = isFullscreen
      ? "Exit Fullscreen"
      : "Fullscreen";
  }
  graphToolFullscreenButton?.classList.toggle("active", isFullscreen);
}

function updateQueryFullscreenButton() {
  const isFullscreen = document.fullscreenElement === queryDock;
  queryToolFullscreenButton?.classList.toggle("active", isFullscreen);
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

function updateSearchPagination(payload = {}) {
  const page = payload.page || searchState.page || 1;
  const totalPages = payload.total_pages || searchState.totalPages || 0;
  const hasPrevious = Boolean(payload.has_previous);
  const hasNext = Boolean(payload.has_next);
  searchPrevPageButton.disabled = !hasPrevious;
  searchNextPageButton.disabled = !hasNext;
  searchPageStatus.textContent = totalPages
    ? `Page ${page} of ${totalPages}`
    : "No pages";
}

function renderSearchMeta(payload) {
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
  const currentValue = selectedValue || "";
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

  searchClassFilter.value = currentValue;
}

function setActiveGraphMode(mode) {
  graphModeButtons.forEach((button) => {
    button.classList.toggle("active", button.dataset.graphMode === mode);
  });
}

function updateStageSummary(statusText = null) {
  if (!graphToolbarStatus) {
    return;
  }
  if (!bootstrapPayload || !hasActiveGraphData()) {
    graphToolbarStatus.textContent = statusText || "No IFC uploaded";
    return;
  }
  const visibleEdgeCount = graphView?.getVisibleEdgeCount?.();
  const parts = [
    `${bootstrapPayload.graph.node_count} nodes`,
    Number.isFinite(visibleEdgeCount)
      ? `${visibleEdgeCount} edges shown`
      : `${bootstrapPayload.graph.edge_count} edges`,
  ];
  if (statusText) {
    parts.push(statusText);
  }
  graphToolbarStatus.textContent = parts.join(" | ");
}

function setGraphEmptyState(visible) {
  graphEmptyState.hidden = !visible;
}

function updateGraphEmptyStateContent(payload = bootstrapPayload) {
  if (!graphEmptyTitle || !graphEmptyMessage || !graphEmptyHint) {
    return;
  }
  if (!hasActiveGraphData(payload)) {
    graphEmptyTitle.textContent = "No IFC uploaded";
    graphEmptyMessage.textContent =
      "Upload an IFC to build the viewer graph, database, and overlays.";
    graphEmptyHint.textContent = "";
    graphEmptyHint.hidden = true;
    return;
  }
  graphEmptyTitle.textContent = "WebGL graph assets not found";
  graphEmptyMessage.innerHTML =
    "Regenerate them with " +
    "<code>uv run rag-tag-jsonl-to-graph --jsonl-dir output --out-dir output</code>.";
  graphEmptyHint.textContent =
    "The legacy Plotly debugger is still available from the toolbar link above.";
  graphEmptyHint.hidden = false;
}

function setInspectorEmptyState(empty, caption = null) {
  inspectorPanel.classList.toggle("is-empty", empty);
  inspector.classList.toggle("inspector-empty", empty);
  if (caption !== null) {
    inspectorCaption.textContent = caption;
  }
}

function syncGraphToggleButtons() {
  toggleEdgeLabelsButton.classList.toggle("active", edgeLabelsEnabled);
  toggleEdgeLabelsButton.textContent = edgeLabelsEnabled
    ? "Edge Labels: On"
    : "Edge Labels: Off";
  toggleBBoxesButton.classList.toggle("active", bboxesEnabled);
  toggleBBoxesButton.textContent = bboxesEnabled ? "BBoxes: On" : "BBoxes: Off";
  toggleMeshesButton.classList.toggle("active", meshesEnabled);
  toggleMeshesButton.textContent = meshesEnabled ? "Meshes: On" : "Meshes: Off";
  syncPanelToggleButton(toggleLegendButton, legendVisible);
  setChipToggleLabel(toggleLegendButton, "Legend", legendVisible);
  syncPanelToggleButton(toggleSearchPanelButton, searchPanelVisible);
  setChipToggleLabel(toggleSearchPanelButton, "Explorer", searchPanelVisible);
  syncPanelToggleButton(toggleInspectorPanelButton, inspectorPanelVisible);
  setChipToggleLabel(toggleInspectorPanelButton, "Inspector", inspectorPanelVisible);
}

function renderSearchUnavailableState(
  message = "Upload an IFC to search the graph."
) {
  populateSearchClassOptions([], "");
  searchResults.innerHTML = "";
  searchResults.textContent = message;
  searchResultsMeta.textContent = "No IFC uploaded.";
  updateSearchPagination({ page: 1, total_pages: 0 });
}

function clearQueryEmptyState() {
  if (!queryResult.classList.contains("query-empty")) {
    return;
  }
  queryResult.classList.remove("query-empty");
  queryResult.textContent = "";
}

function appendQueryLine(parent, className, text) {
  const line = document.createElement("div");
  line.className = className;
  line.textContent = text;
  parent.appendChild(line);
  return line;
}

function appendQueryList(parent, items) {
  const list = document.createElement("ul");
  list.className = "query-list";
  for (const item of items) {
    const row = document.createElement("li");
    row.className = "query-item-row";
    row.textContent = `   ${item}`;
    list.appendChild(row);
  }
  parent.appendChild(list);
}

function appendAnswerBody(parent, answer) {
  const body = document.createElement("div");
  body.className = "query-answer-body";
  body.innerHTML = markdownToHtml(answer);
  parent.appendChild(body);
}

function appendQueryDivider(parent) {
  appendQueryLine(parent, "query-line-divider", "-".repeat(60));
}

function appendQueryDetails(parent, detailsJson, truncated) {
  if (!detailsJson) {
    return;
  }

  const block = document.createElement("section");
  block.className = "query-details";
  if (!queryDetailsVisible) {
    block.classList.add("is-hidden");
  }

  const label = document.createElement("div");
  label.className = "query-details-label";
  label.textContent = truncated ? "Details (truncated)" : "Details";
  block.appendChild(label);

  const pre = document.createElement("pre");
  pre.textContent = detailsJson;
  block.appendChild(pre);

  parent.appendChild(block);
}

function createPendingQueryEntry(question) {
  clearQueryEmptyState();

  const entry = document.createElement("section");
  entry.className = "query-entry pending";
  appendQueryLine(entry, "query-line-question", `Q: ${question}`);
  const routeLine = appendQueryLine(entry, "query-line-route", "   working...");
  queryResult.appendChild(entry);
  queryResult.scrollTop = queryResult.scrollHeight;

  return { entry, routeLine, transcriptEntry: createTranscriptEntry(question) };
}

function renderQueryEntry(entryState, payload, durationMs) {
  const { entry, routeLine, transcriptEntry } = entryState;
  const presentation = payload.presentation || {};

  entry.classList.remove("pending");
  routeLine.textContent = `   [${payload.route || "?"}] ${payload.decision || ""}`;
  transcriptEntry.status = "done";
  transcriptEntry.route = payload.route || "?";
  transcriptEntry.decision = payload.decision || "";
  transcriptEntry.warning = presentation.warning || "";
  transcriptEntry.sqlItems = Array.isArray(presentation.sql_items)
    ? [...presentation.sql_items]
    : [];
  transcriptEntry.sqlMoreCount = Number(presentation.sql_more_count) || 0;
  transcriptEntry.graphSample = Array.isArray(presentation.graph_sample)
    ? [...presentation.graph_sample]
    : [];
  transcriptEntry.detailsJson = presentation.details_json || "";
  transcriptEntry.detailsTruncated = Boolean(presentation.details_truncated);

  lastQueryRoute = payload.route || "";
  lastQueryDurationMs = durationMs;
  updateQueryStatus();

  if (presentation.error) {
    appendQueryLine(entry, "query-warning", `Error: ${presentation.error}`);
    transcriptEntry.error = presentation.error;
  } else {
    appendQueryLine(entry, "query-line-answer", "A:");
    const answer = presentation.answer || payload.answer || "No answer produced.";
    appendAnswerBody(entry, answer);
    transcriptEntry.answer = answer;

    if (presentation.warning) {
      appendQueryLine(entry, "query-warning", `Warning: ${presentation.warning}`);
    }

    if (presentation.sql_items?.length) {
      appendQueryLine(entry, "query-item-row", "");
      appendQueryList(entry, presentation.sql_items);
      if (presentation.sql_more_count > 0) {
        appendQueryLine(
          entry,
          "query-item-row",
          `   (${presentation.sql_more_count} more items not shown)`
        );
      }
    }

    if (presentation.graph_sample?.length) {
      appendQueryLine(entry, "query-sample-title", "");
      appendQueryLine(entry, "query-sample-title", "   Sample:");
      appendQueryList(
        entry,
        presentation.graph_sample.map((item) => `- ${item}`)
      );
    }
  }

  appendQueryDetails(
    entry,
    presentation.details_json,
    presentation.details_truncated
  );

  appendQueryDivider(entry);
  queryResult.scrollTop = queryResult.scrollHeight;
}

function renderQueryFailure(entryState, error, durationMs) {
  const { entry, routeLine, transcriptEntry } = entryState;
  entry.classList.remove("pending");
  routeLine.textContent = "   [error] query failed";
  appendQueryLine(entry, "query-warning", `Error: ${String(error)}`);
  transcriptEntry.status = "error";
  transcriptEntry.route = "error";
  transcriptEntry.decision = "query failed";
  transcriptEntry.error = String(error);
  lastQueryRoute = "error";
  lastQueryDurationMs = durationMs;
  updateQueryStatus();
  queryResult.scrollTop = queryResult.scrollHeight;
}

async function loadBootstrap() {
  setGraphEmptyState(false);
  const [health, bootstrap] = await Promise.all([
    api("/api/health"),
    api("/api/bootstrap"),
  ]);
  setHealth(health.status === "ok");
  bootstrapPayload = bootstrap;
  updateQueryStatus();
  updateGraphEmptyStateContent(bootstrap);
  if (graphToolLegacyLink) {
    graphToolLegacyLink.href = bootstrap.debug_graph_url || "/debug/graph";
  }
  syncModelViewerLink();
  updateFullscreenButton();
  if (!bootstrap.webgl_graph_available) {
    setGraphEmptyState(true);
    updateStageSummary(
      hasActiveGraphData(bootstrap) ? "Viewer assets missing" : "No IFC uploaded"
    );
    return;
  }

  graphView = new WebGLGraphView({
    canvas: graphCanvas,
    overlayCanvas: graphOverlay,
  });
  graphView.onSelectionChange((node) => {
    selectedNodeId = node.id;
    loadElement(node.id, node);
  });
  await graphView.loadFromManifestUrl(bootstrap.webgl_graph_manifest_url);
  setGraphEmptyState(false);
  const manifest = graphView.manifest;
  edgeLabelsEnabled = graphView.showEdgeLabels;
  bboxesEnabled = graphView.showBBoxes;
  meshesEnabled = graphView.showMeshes;
  legendNodeGroups = manifest.node_groups || {};
  syncGraphToggleButtons();
  renderLegend(legendNodeGroups);
  setActiveGraphMode(graphView.mode);
  setLegendVisible(legendVisible);
  updateQueryStatus();
  updateStageSummary();
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
        `Uploading ${file.name}...`,
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
        source: "graph",
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
        source: "graph",
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

async function runQuery(question) {
  const entryState = createPendingQueryEntry(question);
  const startedAt = performance.now();
  queryInput.disabled = true;
  querySubmitButton.disabled = true;
  querySubmitButton.textContent = "Running...";
  updateQueryStatus(`Processing: ${truncateText(question, 48)}`);

  try {
    const payload = await api("/api/query", {
      method: "POST",
      body: JSON.stringify({ question }),
    });
    renderQueryEntry(entryState, payload, performance.now() - startedAt);
  } catch (error) {
    renderQueryFailure(entryState, error, performance.now() - startedAt);
  } finally {
    queryInput.disabled = false;
    querySubmitButton.disabled = false;
    querySubmitButton.textContent = DEFAULT_QUERY_SUBMIT_LABEL;
    queryInput.focus();
  }
}

function focusSearchResult(result) {
  selectedNodeId = result.id;
  if (graphView) {
    const focused = graphView.focusNode(result.id);
    if (!focused) {
      loadElement(result.id, result);
    }
    return;
  }
  loadElement(result.id, result);
}

function renderSearchResults(payload) {
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
    button.addEventListener("click", () => focusSearchResult(result));

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

async function loadElement(nodeId, nodeSummary = null) {
  setInspectorEmptyState(
    false,
    nodeSummary?.label
      ? `${nodeSummary.label} • loading details`
      : `${nodeId} • loading details`
  );
  inspector.textContent = "Loading element...";
  try {
    const payload = await api(`/api/element/${encodeURIComponent(nodeId)}`);
    const className = payload.element.class_ || nodeSummary?.class_name || "Unknown";
    const title = payload.element.label || nodeSummary?.label || nodeId;
    setInspectorEmptyState(false, `${title} • ${className} • ${nodeId}`);
    inspector.textContent = pretty(payload.element);
  } catch (error) {
    setInspectorEmptyState(false, `${nodeId} • failed to load`);
    inspector.textContent = `Failed to load element ${nodeId}\n${String(error)}`;
  }
}

async function runSearch({ resetPage = false, page = null } = {}) {
  if (bootstrapPayload && !hasActiveGraphData()) {
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

  searchResults.textContent = "Searching...";
  searchResultsMeta.textContent = "Searching node browser...";
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
    searchResults.textContent = String(error);
    searchResultsMeta.textContent = "Search failed.";
    updateSearchPagination({ page: 1, total_pages: 0 });
  }
}

queryForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const question = queryInput.value.trim();
  if (!question) {
    return;
  }
  queryInput.value = "";
  await runQuery(question);
});

queryInput.addEventListener("keydown", (event) => {
  if (event.key !== "Enter" || event.shiftKey) {
    return;
  }
  event.preventDefault();
  queryForm.requestSubmit();
});

queryToggleDetailsButton?.addEventListener("click", () => {
  queryDetailsVisible = !queryDetailsVisible;
  applyQueryDetailsVisibility();
});

queryCopyTranscriptButton?.addEventListener("click", async () => {
  const transcript = buildTranscriptMarkdown();
  if (!transcript) {
    flashQueryStatus("No transcript content to copy.");
    return;
  }

  try {
    await copyTextToClipboard(transcript);
    flashQueryStatus("Transcript copied.");
  } catch (error) {
    flashQueryStatus(`Copy failed: ${String(error)}`);
  }
});

searchForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  await runSearch({ resetPage: true });
});

searchClassFilter.addEventListener("change", async () => {
  await runSearch({ resetPage: true });
});

searchPageSize.addEventListener("change", async () => {
  await runSearch({ resetPage: true });
});

searchResetButton.addEventListener("click", async () => {
  searchInput.value = "";
  searchClassFilter.value = "";
  searchPageSize.value = "25";
  await runSearch({ resetPage: true });
});

searchPrevPageButton.addEventListener("click", async () => {
  if (searchState.page <= 1) {
    return;
  }
  await runSearch({ page: searchState.page - 1 });
});

searchNextPageButton.addEventListener("click", async () => {
  if (searchState.page >= searchState.totalPages) {
    return;
  }
  await runSearch({ page: searchState.page + 1 });
});

legendPageButtons.forEach((button) => {
  button.addEventListener("click", () => {
    const page = button.dataset.legendPage;
    if (!page || page === activeLegendPage) {
      return;
    }
    syncLegendPage(page);
  });
});

graphModeButtons.forEach((button) => {
  button.addEventListener("click", () => {
    if (!graphView) {
      return;
    }
    const mode = button.dataset.graphMode;
    graphView.setMode(mode);
    setActiveGraphMode(mode);
    renderLegend(legendNodeGroups);
    updateStageSummary();
  });
});

toggleEdgeLabelsButton.addEventListener("click", () => {
  if (!graphView) {
    return;
  }
  edgeLabelsEnabled = !edgeLabelsEnabled;
  graphView.setEdgeLabelsEnabled(edgeLabelsEnabled);
  syncGraphToggleButtons();
});

toggleBBoxesButton.addEventListener("click", async () => {
  if (!graphView) {
    return;
  }
  bboxesEnabled = !bboxesEnabled;
  await graphView.setOverlayEnabled("bbox", bboxesEnabled);
  syncGraphToggleButtons();
});

toggleMeshesButton.addEventListener("click", async () => {
  if (!graphView) {
    return;
  }
  meshesEnabled = !meshesEnabled;
  await graphView.setOverlayEnabled("mesh", meshesEnabled);
  syncGraphToggleButtons();
});

toggleFullscreenButton?.addEventListener("click", async () => {
  if (!document.fullscreenEnabled) {
    return;
  }
  try {
    if (document.fullscreenElement === graphStage) {
      await document.exitFullscreen();
    } else {
      await graphStage.requestFullscreen();
    }
  } catch (error) {
    updateStageSummary("Fullscreen unavailable");
  }
});

toggleLegendButton.addEventListener("click", () => {
  setLegendVisible(!legendVisible);
});

graphToolFullscreenButton?.addEventListener("click", async () => {
  if (toggleFullscreenButton) {
    toggleFullscreenButton.click();
    return;
  }
  if (!document.fullscreenEnabled) {
    return;
  }
  try {
    if (document.fullscreenElement === graphStage) {
      await document.exitFullscreen();
    } else {
      await graphStage.requestFullscreen();
    }
  } catch (error) {
    updateStageSummary("Fullscreen unavailable");
  }
});

graphToolResetButton?.addEventListener("click", () => {
  graphView?.resetView();
});

queryToolFullscreenButton?.addEventListener("click", async () => {
  if (!document.fullscreenEnabled || !queryDock) {
    return;
  }
  try {
    if (document.fullscreenElement === queryDock) {
      await document.exitFullscreen();
    } else {
      await queryDock.requestFullscreen();
    }
  } catch (error) {
    flashQueryStatus("Query fullscreen unavailable");
  }
});

toggleSearchPanelButton?.addEventListener("click", () => {
  setSearchPanelVisible(!searchPanelVisible);
});

toggleInspectorPanelButton?.addEventListener("click", () => {
  setInspectorPanelVisible(!inspectorPanelVisible);
});

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

document.addEventListener("fullscreenchange", () => {
  updateFullscreenButton();
  updateQueryFullscreenButton();
  graphView?.resize();
});

setLegendVisible(false);
setSearchPanelVisible(false);
setInspectorPanelVisible(false);
setUploadBusy(false, DEFAULT_UPLOAD_STATUS, null);
setInspectorEmptyState(
  true,
  "Select a node in the graph or from search to inspect its properties."
);
updateFullscreenButton();
updateQueryFullscreenButton();
installViewerRefreshListener();
syncQueryDetailsToggle();
updateQueryStatus();

if (!document.fullscreenEnabled) {
  if (toggleFullscreenButton) {
    toggleFullscreenButton.hidden = true;
  }
  if (graphToolFullscreenButton) {
    graphToolFullscreenButton.hidden = true;
  }
  if (queryToolFullscreenButton) {
    queryToolFullscreenButton.hidden = true;
  }
}

loadBootstrap()
  .then(() => runSearch({ resetPage: true }))
  .catch((error) => {
    if (graphToolbarStatus) {
      graphToolbarStatus.textContent = `Graph unavailable | ${String(error)}`;
    }
    setHealth(false);
    setGraphEmptyState(true);
  });
