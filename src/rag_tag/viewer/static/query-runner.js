const queryForm = document.getElementById("query-form")
const queryInput = document.getElementById("query-input")
const queryResult = document.getElementById("query-result")
const queryStatus = document.getElementById("query-status")
const queryDock = document.getElementById("query-dock")
const queryToolFullscreenButton = document.getElementById(
  "query-tool-fullscreen"
)
const queryToggleDetailsButton = document.getElementById("query-toggle-details")
const queryCopyTranscriptButton = document.getElementById("query-copy-transcript")
const querySubmitButton = queryForm?.querySelector('button[type="submit"]')

const bootstrapEndpoint = document.body.classList.contains("model-page")
  ? "/api/model/bootstrap"
  : "/api/bootstrap"
const DEFAULT_QUERY_SUBMIT_LABEL = querySubmitButton?.textContent || "Run Query"

let bootstrapPayload = null
let queryDetailsVisible = false
let lastQueryRoute = ""
let lastQueryDurationMs = 0
let queryTranscriptEntries = []

function pretty(value) {
  return JSON.stringify(value, null, 2)
}

async function api(path, options = {}) {
  const headers = new Headers(options.headers || {})
  headers.set("Accept", "application/json")
  if (options.body && !headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json")
  }
  const response = await fetch(path, {
    ...options,
    cache: "no-store",
    headers,
  })
  if (!response.ok) {
    const contentType = response.headers.get("content-type") || ""
    if (contentType.includes("application/json")) {
      const payload = await response.json().catch(() => null)
      const detail = payload?.detail ?? payload?.error ?? payload
      throw new Error(
        typeof detail === "string"
          ? detail
          : pretty(detail || `Request failed with ${response.status}`)
      )
    }
    const text = await response.text()
    throw new Error(text || `Request failed with ${response.status}`)
  }
  return response.json()
}

function truncateText(text, maxLen) {
  const value = String(text || "")
  if (value.length <= maxLen) {
    return value
  }
  if (maxLen <= 3) {
    return value.slice(0, maxLen)
  }
  return `${value.slice(0, maxLen - 3)}...`
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;")
}

function renderInlineMarkdown(text) {
  return escapeHtml(text)
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
}

function markdownToHtml(markdown) {
  const source = String(markdown || "").trim()
  if (!source) {
    return "<p>No answer produced.</p>"
  }

  const blocks = source.split(/\n\s*\n/)
  return blocks
    .map((block) => {
      const trimmed = block.trim()
      if (!trimmed) {
        return ""
      }

      if (trimmed.startsWith("```") && trimmed.endsWith("```")) {
        const code = trimmed
          .replace(/^```[^\n]*\n?/, "")
          .replace(/\n?```$/, "")
        return `<pre><code>${escapeHtml(code)}</code></pre>`
      }

      const lines = block.split("\n")
      if (lines.every((line) => line.trim().startsWith("- "))) {
        const items = lines
          .map((line) => line.trim().slice(2))
          .map((line) => `<li>${renderInlineMarkdown(line)}</li>`)
          .join("")
        return `<ul>${items}</ul>`
      }

      return `<p>${lines
        .map((line) => renderInlineMarkdown(line))
        .join("<br>")}</p>`
    })
    .join("")
}

function buildQueryStatusText() {
  const dataset = bootstrapPayload?.selected_dataset || "No dataset"
  const parts = [`Dataset: ${dataset}`]
  if (lastQueryRoute) {
    parts.push(`Route: ${lastQueryRoute}`)
  }
  if (lastQueryDurationMs > 0) {
    parts.push(`${Math.round(lastQueryDurationMs)}ms`)
  }
  parts.push(`details:${queryDetailsVisible ? "on" : "off"}`)
  return parts.join(" | ")
}

function updateQueryStatus(text = null) {
  if (!queryStatus) {
    return
  }
  queryStatus.textContent = text || buildQueryStatusText()
}

function syncQueryDetailsToggle() {
  if (!queryToggleDetailsButton) {
    return
  }
  queryToggleDetailsButton.textContent = `Details: ${
    queryDetailsVisible ? "On" : "Off"
  }`
  queryToggleDetailsButton.setAttribute(
    "aria-pressed",
    queryDetailsVisible ? "true" : "false"
  )
}

function applyQueryDetailsVisibility() {
  if (!queryResult) {
    return
  }
  for (const block of queryResult.querySelectorAll(".query-details")) {
    block.classList.toggle("is-hidden", !queryDetailsVisible)
  }
  syncQueryDetailsToggle()
  updateQueryStatus()
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
  }
  queryTranscriptEntries.push(entry)
  return entry
}

function buildTranscriptMarkdown() {
  const blocks = []
  for (const entry of queryTranscriptEntries) {
    blocks.push(`**Q:** ${entry.question}`)

    if (entry.status === "pending") {
      blocks.push("   working...")
      continue
    }

    if (entry.route || entry.decision) {
      blocks.push(`   [${entry.route || "?"}] ${entry.decision || ""}`.trimEnd())
    }

    if (entry.error) {
      blocks.push(`Error: ${entry.error}`)
      blocks.push("---")
      continue
    }

    blocks.push("**A:**")
    blocks.push(entry.answer || "No answer produced.")

    if (entry.warning) {
      blocks.push(`Warning: ${entry.warning}`)
    }

    if (entry.sqlItems.length) {
      blocks.push(...entry.sqlItems.map((item) => `   ${item}`))
      if (entry.sqlMoreCount > 0) {
        blocks.push(`   (${entry.sqlMoreCount} more items not shown)`)
      }
    }

    if (entry.graphSample.length) {
      blocks.push("   Sample:")
      blocks.push(...entry.graphSample.map((item) => `   - ${item}`))
    }

    if (queryDetailsVisible && entry.detailsJson) {
      blocks.push(`\`\`\`json\n${entry.detailsJson}\n\`\`\``)
    }

    blocks.push("---")
  }
  return blocks.join("\n\n").trim()
}

async function copyTextToClipboard(text) {
  if (navigator.clipboard && window.isSecureContext) {
    await navigator.clipboard.writeText(text)
    return
  }

  const helper = document.createElement("textarea")
  helper.value = text
  helper.setAttribute("readonly", "")
  helper.style.position = "fixed"
  helper.style.opacity = "0"
  helper.style.pointerEvents = "none"
  document.body.appendChild(helper)
  helper.select()
  document.execCommand("copy")
  helper.remove()
}

function flashQueryStatus(message) {
  updateQueryStatus(message)
  window.setTimeout(() => updateQueryStatus(), 1600)
}

function updateQueryFullscreenButton() {
  if (!queryToolFullscreenButton || !queryDock) {
    return
  }
  const isFullscreen = document.fullscreenElement === queryDock
  queryToolFullscreenButton.classList.toggle("active", isFullscreen)
}

function createPendingQueryEntry(question) {
  const entry = createTranscriptEntry(question)
  if (!queryResult) {
    return entry
  }
  queryResult.classList.remove("query-empty")
  queryResult.textContent = ""
  const article = document.createElement("article")
  article.className = "query-entry pending"
  article.innerHTML = `
    <div class="query-line-question">Q: ${escapeHtml(question)}</div>
    <div class="query-line-route">Running query...</div>
  `
  queryResult.prepend(article)
  entry.element = article
  return entry
}

function renderQueryEntry(entry, payload, durationMs) {
  if (!entry?.element) {
    return
  }
  const presentation = payload?.presentation || {}
  entry.status = "completed"
  entry.route = payload?.route || ""
  entry.decision = payload?.decision || ""
  entry.answer = presentation.answer || payload?.answer || ""
  entry.warning = presentation.warning || ""
  entry.error = presentation.error || ""
  entry.sqlItems = Array.isArray(presentation.sql_items)
    ? presentation.sql_items
    : []
  entry.sqlMoreCount = Number(presentation.sql_more_count) || 0
  entry.graphSample = Array.isArray(presentation.graph_sample)
    ? presentation.graph_sample
    : []
  entry.detailsJson = presentation.details_json || ""
  entry.detailsTruncated = Boolean(presentation.details_truncated)

  lastQueryRoute = entry.route
  lastQueryDurationMs = durationMs

  const sqlList = entry.sqlItems.length
    ? `<ul class="query-list">${entry.sqlItems
        .map((item) => `<li class="query-item-row">${escapeHtml(item)}</li>`)
        .join("")}</ul>`
    : ""
  const sqlMore = entry.sqlMoreCount
    ? `<div class="query-item-row">(${entry.sqlMoreCount} more items not shown)</div>`
    : ""
  const graphSample = entry.graphSample.length
    ? `
      <div class="query-sample-title">Sample:</div>
      <ul class="query-list">${entry.graphSample
        .map((item) => `<li class="query-item-row">${escapeHtml(item)}</li>`)
        .join("")}</ul>
    `
    : ""
  const details = entry.detailsJson
    ? `
      <div class="query-details${queryDetailsVisible ? "" : " is-hidden"}">
        <div class="query-details-label">${
          entry.detailsTruncated ? "Details (truncated)" : "Details"
        }</div>
        <pre>${escapeHtml(entry.detailsJson)}</pre>
      </div>
    `
    : ""

  entry.element.classList.remove("pending")
  entry.element.innerHTML = `
    <div class="query-line-question">Q: ${escapeHtml(entry.question)}</div>
    <div class="query-line-route">[${escapeHtml(entry.route || "?")}] ${escapeHtml(
      entry.decision || ""
    )}</div>
    <div class="query-line-answer">A:</div>
    <div class="query-answer-body">${markdownToHtml(entry.answer)}</div>
    ${entry.warning ? `<div class="query-warning">${escapeHtml(entry.warning)}</div>` : ""}
    ${sqlList}
    ${sqlMore}
    ${graphSample}
    ${details}
  `
  updateQueryStatus()
}

function renderQueryFailure(entry, error, durationMs) {
  if (!entry?.element) {
    return
  }
  entry.status = "failed"
  entry.error = String(error)
  lastQueryDurationMs = durationMs
  entry.element.classList.remove("pending")
  entry.element.innerHTML = `
    <div class="query-line-question">Q: ${escapeHtml(entry.question)}</div>
    <div class="query-line-answer">Error:</div>
    <div class="query-warning">${escapeHtml(String(error))}</div>
  `
  updateQueryStatus(`Query failed | ${Math.round(durationMs)}ms`)
}

async function runQuery(question) {
  const entryState = createPendingQueryEntry(question)
  const startedAt = performance.now()
  queryInput.disabled = true
  querySubmitButton.disabled = true
  querySubmitButton.textContent = "Running..."
  updateQueryStatus(`Processing: ${truncateText(question, 48)}`)

  try {
    const payload = await api("/api/query", {
      method: "POST",
      body: JSON.stringify({ question }),
    })
    renderQueryEntry(entryState, payload, performance.now() - startedAt)
  } catch (error) {
    renderQueryFailure(entryState, error, performance.now() - startedAt)
  } finally {
    queryInput.disabled = false
    querySubmitButton.disabled = false
    querySubmitButton.textContent = DEFAULT_QUERY_SUBMIT_LABEL
    queryInput.focus()
  }
}

async function initializeQueryRunner() {
  if (!queryForm || !queryInput || !queryResult || !querySubmitButton) {
    return
  }

  try {
    bootstrapPayload = await api(bootstrapEndpoint)
  } catch {
    bootstrapPayload = null
  }

  updateQueryStatus()
  syncQueryDetailsToggle()
  updateQueryFullscreenButton()

  queryForm.addEventListener("submit", async (event) => {
    event.preventDefault()
    const question = queryInput.value.trim()
    if (!question) {
      return
    }
    queryInput.value = ""
    await runQuery(question)
  })

  queryInput.addEventListener("keydown", (event) => {
    if (event.key !== "Enter" || event.shiftKey) {
      return
    }
    event.preventDefault()
    queryForm.requestSubmit()
  })

  queryToggleDetailsButton?.addEventListener("click", () => {
    queryDetailsVisible = !queryDetailsVisible
    applyQueryDetailsVisibility()
  })

  queryCopyTranscriptButton?.addEventListener("click", async () => {
    const transcript = buildTranscriptMarkdown()
    if (!transcript) {
      flashQueryStatus("No transcript content to copy.")
      return
    }

    try {
      await copyTextToClipboard(transcript)
      flashQueryStatus("Transcript copied.")
    } catch (error) {
      flashQueryStatus(`Copy failed: ${String(error)}`)
    }
  })

  queryToolFullscreenButton?.addEventListener("click", async () => {
    if (!document.fullscreenEnabled || !queryDock) {
      return
    }
    try {
      if (document.fullscreenElement === queryDock) {
        await document.exitFullscreen()
      } else {
        await queryDock.requestFullscreen()
      }
    } catch (error) {
      flashQueryStatus(`Fullscreen unavailable: ${String(error)}`)
    }
  })

  document.addEventListener("fullscreenchange", () => {
    updateQueryFullscreenButton()
  })
}

initializeQueryRunner()
