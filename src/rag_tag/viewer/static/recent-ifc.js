const VIEWER_REFRESH_SIGNAL_KEY = "ragtag-viewer-refresh-signal"
const bootstrapEndpoint = document.body.classList.contains("model-page")
  ? "/api/model/bootstrap"
  : "/api/bootstrap"

const recentIfcPicker = document.getElementById("recent-ifc-picker")
const recentIfcSelect = document.getElementById("recent-ifc-select")
const recentIfcOpenButton = document.getElementById("recent-ifc-open")
const ifcUploadStatus = document.getElementById("ifc-upload-status")
const ifcUploadProgressRow = document.getElementById("ifc-upload-progress-row")
const ifcUploadProgress = document.getElementById("ifc-upload-progress")
const ifcUploadProgressValue = document.getElementById("ifc-upload-progress-value")

let recentIfcEntries = []
let activeRecentIfcId = null
let recentIfcBusy = false

async function api(path, options = {}) {
  const headers = new Headers(options.headers || {})
  headers.set("Accept", "application/json")
  if (options.body && !headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json")
  }
  const response = await fetch(path, {
    ...options,
    headers,
    cache: "no-store",
  })
  if (!response.ok) {
    const contentType = response.headers.get("content-type") || ""
    if (contentType.includes("application/json")) {
      const payload = await response.json().catch(() => null)
      const detail = payload?.detail ?? payload?.error ?? payload
      throw new Error(typeof detail === "string" ? detail : `Request failed with ${response.status}`)
    }
    throw new Error((await response.text()) || `Request failed with ${response.status}`)
  }
  return response.json()
}

function setProgress(progressValue = null) {
  const hasProgress = Number.isFinite(progressValue)
  if (ifcUploadProgressRow) {
    ifcUploadProgressRow.hidden = !hasProgress
  }
  if (!hasProgress) {
    if (ifcUploadProgress) {
      ifcUploadProgress.value = 0
    }
    if (ifcUploadProgressValue) {
      ifcUploadProgressValue.textContent = ""
    }
    return
  }
  const clamped = Math.max(0, Math.min(100, Math.round(progressValue)))
  if (ifcUploadProgress) {
    ifcUploadProgress.value = clamped
  }
  if (ifcUploadProgressValue) {
    ifcUploadProgressValue.textContent = `${clamped}%`
  }
}

function emitViewerRefreshSignal(detail = {}) {
  try {
    localStorage.setItem(
      VIEWER_REFRESH_SIGNAL_KEY,
      JSON.stringify({
        ...detail,
        pathname: window.location.pathname,
        timestamp: Date.now(),
      })
    )
  } catch {}
}

function setRecentIfcBusy(busy, statusText = null, progressValue = null) {
  recentIfcBusy = busy
  if (recentIfcSelect) {
    recentIfcSelect.disabled = busy || recentIfcEntries.length === 0
  }
  if (recentIfcOpenButton) {
    recentIfcOpenButton.disabled = busy || !recentIfcSelect?.value
    recentIfcOpenButton.textContent = busy ? "Opening..." : "Open Recent"
  }
  if (statusText !== null && ifcUploadStatus) {
    ifcUploadStatus.textContent = statusText
  }
  setProgress(progressValue)
}

function optionLabel(entry) {
  const flags = []
  if (entry.active) {
    flags.push("Active")
  } else if (entry.build_ready) {
    flags.push("Ready")
  } else {
    flags.push("Needs rebuild")
  }
  return `${entry.source_name} (${flags.join(", ")})`
}

function renderRecentIfcPicker() {
  if (!recentIfcPicker || !recentIfcSelect || !recentIfcOpenButton) {
    return
  }
  recentIfcSelect.innerHTML = ""
  if (!recentIfcEntries.length) {
    recentIfcPicker.hidden = true
    return
  }

  const placeholder = document.createElement("option")
  placeholder.value = ""
  placeholder.textContent = "Recent IFCs"
  recentIfcSelect.appendChild(placeholder)

  for (const entry of recentIfcEntries) {
    const option = document.createElement("option")
    option.value = entry.id
    option.textContent = optionLabel(entry)
    recentIfcSelect.appendChild(option)
  }

  recentIfcPicker.hidden = false
  recentIfcSelect.value = activeRecentIfcId || recentIfcEntries[0].id
  setRecentIfcBusy(false)
}

async function pollImportJob(jobId) {
  for (;;) {
    const payload = await api(`/api/import-ifc/${encodeURIComponent(jobId)}`)
    setRecentIfcBusy(true, payload.message, payload.progress)
    if (payload.status === "completed") {
      return payload
    }
    if (payload.status === "failed") {
      throw new Error(payload.error || payload.message || "IFC import failed.")
    }
    await new Promise((resolve) => window.setTimeout(resolve, 600))
  }
}

async function activateRecentIfc(recentIfcId) {
  setRecentIfcBusy(true, "Opening recent IFC...", 8)
  try {
    const payload = await api(
      `/api/recent-ifc/${encodeURIComponent(recentIfcId)}/activate`,
      { method: "POST" }
    )
    const result = payload.job_id ? await pollImportJob(payload.job_id) : payload
    setRecentIfcBusy(true, result.message, 100)
    emitViewerRefreshSignal({
      dataset: result.dataset || recentIfcId,
      source: "recent-ifc",
    })
    window.setTimeout(() => {
      window.location.reload()
    }, 400)
  } catch (error) {
    setRecentIfcBusy(false, `Recent IFC open failed: ${String(error)}`, null)
  }
}

async function loadRecentIfcs() {
  if (!recentIfcPicker || !recentIfcSelect || !recentIfcOpenButton) {
    return
  }
  try {
    const payload = await api(bootstrapEndpoint)
    recentIfcEntries = Array.isArray(payload.recent_ifcs) ? payload.recent_ifcs : []
    activeRecentIfcId = payload.active_recent_ifc_id || null
    renderRecentIfcPicker()
  } catch {
    recentIfcPicker.hidden = true
  }
}

recentIfcSelect?.addEventListener("change", () => {
  if (!recentIfcBusy) {
    setRecentIfcBusy(false)
  }
})

recentIfcOpenButton?.addEventListener("click", async () => {
  const recentIfcId = recentIfcSelect?.value || ""
  if (!recentIfcId || recentIfcBusy) {
    return
  }
  await activateRecentIfc(recentIfcId)
})

loadRecentIfcs()
