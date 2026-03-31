const legendStatus = document.getElementById("legend-status");
const manifestLink = document.getElementById("manifest-link");
const legendModeCards = document.getElementById("legend-mode-cards");
const legendFamilyCards = document.getElementById("legend-family-cards");
const legendRelationList = document.getElementById("legend-relation-list");

const MODE_DESCRIPTIONS = {
  all: {
    title: "Full Graph",
    summary: "Shows every exported edge together for a complete graph read.",
    why: "Best when you want broad context before narrowing the graph view.",
  },
  hierarchy: {
    title: "Hierarchy",
    summary: "Highlights decomposition and containment structure from IFC.",
    why: "Useful for tracing where an element lives in the building breakdown.",
  },
  spatial: {
    title: "Spatial",
    summary: "Focuses on location and proximity-driven relations.",
    why: "Useful for neighborhood, distance, and placement-oriented questions.",
  },
  topology: {
    title: "Topology",
    summary: "Highlights overlap, boundary, and connectivity-style relations.",
    why: "Useful for adjacency, clash, and contact reasoning in the viewer.",
  },
  explicit: {
    title: "Explicit IFC",
    summary: "Shows authored IFC relationship edges exported directly from the model.",
    why: "Useful when you want model semantics rather than geometric heuristics.",
  },
  nodes: {
    title: "Nodes",
    summary: "Shows the element cloud without edge overlays.",
    why: "Useful when you want to inspect placement and density before relations.",
  },
  edges: {
    title: "Edges",
    summary: "Shows the relationship network without the node cloud.",
    why: "Useful when the structure of the relation network matters most.",
  },
};

const FAMILY_DESCRIPTIONS = {
  hierarchy: {
    title: "Hierarchy",
    summary: "Parent-child and container-child structure exported from IFC.",
    why: "This family anchors the graph in the authored building hierarchy.",
  },
  spatial: {
    title: "Spatial",
    summary: "Relations built from position, distance, or same-scope placement.",
    why: "This family helps the viewer answer location and neighborhood questions.",
  },
  topology: {
    title: "Topology",
    summary: "Relations based on overlap, contact, ordering, or boundaries.",
    why: "This family makes connectivity and physical interaction easier to inspect.",
  },
  explicit: {
    title: "Explicit IFC",
    summary: "Direct IFC relationships preserved as graph edges.",
    why: "This family keeps authored semantics visible instead of only inferred ones.",
  },
};

const CATEGORY_META = {
  hierarchy: { label: "Hierarchy", tone: "hierarchy" },
  spatial: { label: "Spatial", tone: "spatial" },
  topology: { label: "Topology", tone: "topology" },
  explicit: { label: "Explicit IFC", tone: "ifc" },
};

const CATEGORY_ORDER = {
  hierarchy: 0,
  spatial: 1,
  topology: 2,
  explicit: 3,
};

async function api(path) {
  const response = await fetch(path);
  if (!response.ok) {
    throw new Error(`${path}: ${response.status}`);
  }
  return response.json();
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function titleCase(value) {
  return String(value || "")
    .split("_")
    .filter(Boolean)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

function categoryKey(category) {
  const normalized = String(category || "").toLowerCase();
  return normalized in CATEGORY_META ? normalized : "topology";
}

function categoryLabel(category) {
  const key = categoryKey(category);
  return CATEGORY_META[key]?.label || titleCase(key);
}

function categoryTone(category) {
  const key = categoryKey(category);
  return CATEGORY_META[key]?.tone || "neutral";
}

function modeLabel(mode) {
  return MODE_DESCRIPTIONS[mode]?.title || titleCase(mode);
}

function modeTone(mode) {
  if (["hierarchy", "spatial", "topology", "explicit"].includes(mode)) {
    return categoryTone(mode);
  }
  return "neutral";
}

function badgeMarkup(
  values,
  {
    toneResolver = () => "neutral",
    labelResolver = (value) => titleCase(value),
    fallbackLabel = "None",
  } = {}
) {
  const entries = Array.isArray(values) && values.length ? values : [fallbackLabel];
  return entries
    .map((value) => {
      const tone = toneResolver(value);
      const label = labelResolver(value);
      return `
        <span class="guide-badge guide-badge--${escapeHtml(tone)}">
          ${escapeHtml(label)}
        </span>
      `;
    })
    .join("");
}

function relationModes(manifest, relation) {
  return Object.entries(manifest.viewer_modes || {})
    .filter(([mode, relations]) => {
      if (["all", "nodes", "edges"].includes(mode)) {
        return false;
      }
      return Array.isArray(relations) && relations.includes(relation);
    })
    .map(([mode]) => mode);
}

function renderUnavailable(message) {
  legendStatus.textContent = message;
  const card = `<article class="guide-empty-card">${escapeHtml(message)}</article>`;
  legendModeCards.innerHTML = card;
  legendFamilyCards.innerHTML = card;
  legendRelationList.innerHTML = card;
}

function renderModeCards(manifest) {
  const entries = Object.entries(manifest.viewer_modes || {});
  if (!entries.length) {
    legendModeCards.innerHTML =
      '<article class="guide-empty-card">No viewer modes were exported.</article>';
    return;
  }

  legendModeCards.innerHTML = entries
    .map(([mode, relations]) => {
      const description = MODE_DESCRIPTIONS[mode] || {
        title: titleCase(mode),
        summary: "Graph mode exported from the bundle manifest.",
        why: "Useful when you want to isolate one relation slice of the graph.",
      };
      return `
        <article class="guide-card">
          <div class="guide-card-head">
            <div>
              <h3>${escapeHtml(description.title)}</h3>
              <p class="guide-card-summary">${escapeHtml(description.summary)}</p>
            </div>
            <span class="guide-stat">
              ${Array.isArray(relations) ? relations.length : 0} relations
            </span>
          </div>
          <div class="guide-card-meta">
            <div class="guide-label">Best for</div>
            <p class="guide-card-why">${escapeHtml(description.why)}</p>
          </div>
          <div class="guide-badge-row">
            ${badgeMarkup(relations, {
              toneResolver: (relation) => categoryTone(
                manifest.relation_storage_categories?.[relation] || "topology"
              ),
              labelResolver: (relation) => relation,
              fallbackLabel: "No relations",
            })}
          </div>
        </article>
      `;
    })
    .join("");
}

function renderFamilyCards(manifest) {
  const storageCategories = manifest.storage_categories || {};
  const cards = Object.entries(storageCategories)
    .map(([category, relations]) => {
      const presentRelations = (relations || []).filter((relation) =>
        (manifest.relation_names || []).includes(relation)
      );
      if (!presentRelations.length) {
        return "";
      }
      const description = FAMILY_DESCRIPTIONS[category] || {
        title: titleCase(category),
        summary: "Relation family exported from the graph manifest.",
        why: "Useful for grouping similar edges into one viewer story.",
      };
      const tone = categoryTone(category);
      return `
        <article class="guide-card guide-card--family" data-tone="${escapeHtml(tone)}">
          <div class="guide-card-head">
            <div>
              <h3>${escapeHtml(description.title)}</h3>
              <p class="guide-card-summary">${escapeHtml(description.summary)}</p>
            </div>
            <span class="guide-stat">${presentRelations.length} types</span>
          </div>
          <div class="guide-card-meta">
            <div class="guide-label">Why it matters</div>
            <p class="guide-card-why">${escapeHtml(description.why)}</p>
          </div>
          <div class="guide-badge-row">
            ${badgeMarkup([category], {
              toneResolver: () => tone,
              labelResolver: () => categoryLabel(category),
            })}
            ${badgeMarkup(presentRelations, {
              toneResolver: () => "neutral",
              labelResolver: (relation) => relation,
            })}
          </div>
        </article>
      `;
    })
    .join("");

  legendFamilyCards.innerHTML =
    cards ||
    '<article class="guide-empty-card">No edge families were exported.</article>';
}

function renderRelationList(manifest) {
  const relationNames = [...(manifest.relation_names || [])].sort((left, right) => {
    const leftCategory = categoryKey(
      manifest.relation_storage_categories?.[left] || "topology"
    );
    const rightCategory = categoryKey(
      manifest.relation_storage_categories?.[right] || "topology"
    );
    const leftOrder = CATEGORY_ORDER[leftCategory] ?? 99;
    const rightOrder = CATEGORY_ORDER[rightCategory] ?? 99;
    if (leftOrder !== rightOrder) {
      return leftOrder - rightOrder;
    }
    return left.localeCompare(right);
  });

  if (!relationNames.length) {
    legendRelationList.innerHTML =
      '<article class="guide-empty-card">No relations were exported.</article>';
    return;
  }

  legendRelationList.innerHTML = relationNames
    .map((relation) => {
      const category = manifest.relation_storage_categories?.[relation] || "topology";
      const tone = categoryTone(category);
      const color = manifest.relation_colors?.[relation] || "#6b7280";
      const summary =
        manifest.relation_explanations?.[relation] ||
        "Graph relation exported from the bundle manifest.";
      const family = FAMILY_DESCRIPTIONS[categoryKey(category)] || {
        why: "Used to preserve meaningful graph context inside the viewer.",
      };
      const modes = relationModes(manifest, relation);
      return `
        <details class="relation-row" data-category="${escapeHtml(tone)}">
          <summary class="relation-row-summary">
            <div class="relation-col relation-col-name">
              <span
                class="relation-line-swatch"
                style="--relation-color:${escapeHtml(color)}"
                aria-hidden="true"
              ></span>
              <div class="relation-name-group">
                <div class="relation-name">${escapeHtml(relation)}</div>
                <div class="relation-caption">${escapeHtml(categoryLabel(category))} relation</div>
              </div>
            </div>
            <div class="relation-col relation-col-tags">
              ${badgeMarkup([category], {
                toneResolver: () => tone,
                labelResolver: () => categoryLabel(category),
              })}
              <span class="guide-badge guide-badge--neutral">
                ${escapeHtml(`${modes.length + 1} modes`)}
              </span>
            </div>
            <div class="relation-col relation-col-summary">
              ${escapeHtml(summary)}
            </div>
            <div class="relation-row-toggle" aria-hidden="true"></div>
          </summary>
          <div class="relation-row-body">
            <div class="relation-detail-grid">
              <section class="relation-detail-card">
                <div class="guide-label">Calculated from</div>
                <p>${escapeHtml(summary)}</p>
              </section>
              <section class="relation-detail-card">
                <div class="guide-label">Why it matters</div>
                <p>${escapeHtml(family.why)}</p>
              </section>
              <section class="relation-detail-card">
                <div class="guide-label">Shown in modes</div>
                <div class="guide-badge-row">
                  ${badgeMarkup(["all", ...modes], {
                    toneResolver: modeTone,
                    labelResolver: modeLabel,
                  })}
                </div>
              </section>
            </div>
          </div>
        </details>
      `;
    })
    .join("");
}

function renderLegendGuide(bootstrap, manifest) {
  const dataset = manifest.dataset || bootstrap.selected_dataset || "all datasets";
  legendStatus.textContent =
    `${dataset} | ${manifest.node_count || 0} nodes | ` +
    `${manifest.edge_count || 0} edges | ` +
    `${manifest.relation_names?.length || 0} relation types`;
  manifestLink.href = bootstrap.webgl_graph_manifest_url || manifestLink.href;
  renderModeCards(manifest);
  renderFamilyCards(manifest);
  renderRelationList(manifest);
}

async function init() {
  try {
    const bootstrap = await api("/api/bootstrap");
    if (!bootstrap.webgl_graph_available || !bootstrap.webgl_graph_manifest_url) {
      renderUnavailable(
        "WebGL graph bundle not found. Regenerate graph assets to populate this guide."
      );
      return;
    }
    const manifest = await api(bootstrap.webgl_graph_manifest_url);
    renderLegendGuide(bootstrap, manifest);
  } catch (error) {
    renderUnavailable(`Failed to load legend guide: ${String(error)}`);
  }
}

init();
