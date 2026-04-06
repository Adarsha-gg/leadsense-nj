const MAP_METRICS = {
  risk_score: {
    label: "Risk Score",
    format: (v) => `${(Number(v) * 100).toFixed(1)}%`,
  },
  risk_uncertainty: {
    label: "Uncertainty",
    format: (v) => Number(v).toFixed(3),
  },
  minority_share: {
    label: "Minority Share",
    format: (v) => `${(Number(v) * 100).toFixed(1)}%`,
  },
  pct_housing_pre_1950: {
    label: "Pre-1950 Housing Share",
    format: (v) => `${(Number(v) * 100).toFixed(1)}%`,
  },
};

const MAP_COLORS = ["#2b83ba", "#7fcdbb", "#ffffbf", "#fdae61", "#d7191c"];
const NO_DATA_COLOR = "#9aa5b1";
const DEFAULT_MAP_METRIC = "risk_score";
const MAP_ROW_LIMIT = 1200;

const state = {
  dashboard: null,
  benchmark: null,
  aiStatus: null,
  selectedGeoid: null,
  sortedRows: [],
  map: null,
  markersLayer: null,
};

function fmtMoney(n) {
  return `$${Number(n || 0).toLocaleString(undefined, { maximumFractionDigits: 0 })}`;
}

function qs() {
  const budget = document.getElementById("budget").value;
  const fairness = document.getElementById("fairness").value;
  const optimizer = document.getElementById("optimizer").value;
  const county = document.getElementById("county-filter")?.value || "all";
  const params = new URLSearchParams({
    budget: String(budget),
    fairness_tolerance: String(fairness),
    optimizer_method: String(optimizer),
    row_limit: String(MAP_ROW_LIMIT),
  });
  if (county && county !== "all") params.set("county", county);
  return params.toString();
}

function normalize(value, min, max) {
  if (!Number.isFinite(value)) return 0;
  if (!Number.isFinite(min) || !Number.isFinite(max)) return 0;
  if (max <= min) return 0.5;
  return Math.max(0, Math.min(1, (value - min) / (max - min)));
}

function quantile(sortedValues, q) {
  if (!sortedValues.length) return NaN;
  if (sortedValues.length === 1) return sortedValues[0];
  const pos = (sortedValues.length - 1) * q;
  const base = Math.floor(pos);
  const rest = pos - base;
  const lower = sortedValues[base];
  const upper = sortedValues[Math.min(base + 1, sortedValues.length - 1)];
  return lower + rest * (upper - lower);
}

function metricStats(rows, metricKey) {
  const values = rows
    .map((row) => Number(row?.[metricKey]))
    .filter((v) => Number.isFinite(v))
    .sort((a, b) => a - b);
  if (!values.length) return null;
  return {
    min: values[0],
    max: values[values.length - 1],
    q20: quantile(values, 0.2),
    q40: quantile(values, 0.4),
    q60: quantile(values, 0.6),
    q80: quantile(values, 0.8),
  };
}

function metricBin(value, stats) {
  if (!stats || !Number.isFinite(value)) return -1;
  if (stats.max <= stats.min) return 2;
  if (value <= stats.q20) return 0;
  if (value <= stats.q40) return 1;
  if (value <= stats.q60) return 2;
  if (value <= stats.q80) return 3;
  return 4;
}

function mapColor(value, stats) {
  const bin = metricBin(value, stats);
  if (bin < 0) return NO_DATA_COLOR;
  return MAP_COLORS[bin];
}

function stableJitter(seed, salt, spread = 0.006) {
  const s = `${seed}|${salt}`;
  let hash = 0;
  for (let i = 0; i < s.length; i += 1) {
    hash = (hash << 5) - hash + s.charCodeAt(i);
    hash |= 0;
  }
  const scaled = ((hash >>> 0) % 2001) / 1000 - 1.0;
  return scaled * spread;
}

function sortedRows(rows) {
  return [...rows].sort((a, b) => {
    const countyA = String(a.county || "");
    const countyB = String(b.county || "");
    if (countyA !== countyB) return countyA.localeCompare(countyB);
    const muniA = String(a.municipality || "");
    const muniB = String(b.municipality || "");
    if (muniA !== muniB) return muniA.localeCompare(muniB);
    return String(a.geoid).localeCompare(String(b.geoid));
  });
}

function populateCountyFilter() {
  const select = document.getElementById("county-filter");
  if (!select) return;
  const current = select.value || "all";
  const counties = state.dashboard?.available_counties || [];
  select.replaceChildren();

  const allOption = document.createElement("option");
  allOption.value = "all";
  allOption.textContent = "All Counties (sampled)";
  select.appendChild(allOption);

  for (const county of counties) {
    const option = document.createElement("option");
    option.value = String(county);
    option.textContent = String(county);
    select.appendChild(option);
  }
  const validCurrent = counties.includes(current) || current === "all";
  select.value = validCurrent ? current : "all";
}

async function fetchDashboard() {
  const res = await fetch(`/api/dashboard?${qs()}`);
  if (!res.ok) throw new Error(`dashboard fetch failed: ${res.status}`);
  state.dashboard = await res.json();
}

async function fetchBenchmark() {
  const res = await fetch("/api/benchmark");
  if (!res.ok) throw new Error(`benchmark fetch failed: ${res.status}`);
  state.benchmark = await res.json();
}

async function fetchAIStatus() {
  const res = await fetch("/api/ai/status");
  if (!res.ok) throw new Error(`ai status fetch failed: ${res.status}`);
  state.aiStatus = await res.json();
}

function ensureMap() {
  if (state.map) return;
  state.map = L.map("risk-map", { preferCanvas: true }).setView([40.04, -74.45], 8);
  L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>',
  }).addTo(state.map);
  state.markersLayer = L.layerGroup().addTo(state.map);
}

function renderMapLegend(metricKey, stats) {
  const metric = MAP_METRICS[metricKey] || MAP_METRICS[DEFAULT_MAP_METRIC];
  const legend = document.getElementById("map-legend");
  if (!stats) {
    legend.innerHTML = `<div class="legend-title">${metric.label}</div><div class="muted">No numeric values available for this layer.</div>`;
    return;
  }
  if (stats.max <= stats.min) {
    legend.innerHTML = `
      <div class="legend-title">${metric.label}</div>
      <div class="legend-item"><span class="legend-swatch" style="background:${MAP_COLORS[2]};"></span>All values ${metric.format(stats.min)}</div>
    `;
    return;
  }

  const bounds = [stats.min, stats.q20, stats.q40, stats.q60, stats.q80, stats.max];
  const items = [];
  for (let i = 0; i < MAP_COLORS.length; i += 1) {
    const low = bounds[i];
    const high = bounds[i + 1];
    items.push(`
      <div class="legend-item">
        <span class="legend-swatch" style="background:${MAP_COLORS[i]};"></span>
        ${metric.format(low)} - ${metric.format(high)}
      </div>
    `);
  }
  items.push(`
    <div class="legend-item">
      <span class="legend-swatch" style="background:${NO_DATA_COLOR};"></span>
      No data
    </div>
  `);

  legend.innerHTML = `
    <div class="legend-title">${metric.label} (quantile bins)</div>
    <div class="legend-grid">${items.join("")}</div>
  `;
}

function renderMap() {
  ensureMap();
  state.markersLayer.clearLayers();
  const rows = state.dashboard?.rows || [];
  const metricKey = document.getElementById("map-metric")?.value || DEFAULT_MAP_METRIC;
  const metric = MAP_METRICS[metricKey] || MAP_METRICS[DEFAULT_MAP_METRIC];
  const stats = metricStats(rows, metricKey);
  const riskStats = metricStats(rows, "risk_score");
  const useJitter = Boolean(document.getElementById("map-jitter")?.checked);

  for (const row of rows) {
    const lat0 = Number(row.lat);
    const lon0 = Number(row.lon);
    if (!Number.isFinite(lat0) || !Number.isFinite(lon0)) continue;
    const geoid = String(row.geoid);
    const lat = lat0 + (useJitter ? stableJitter(geoid, "lat") : 0);
    const lon = lon0 + (useJitter ? stableJitter(geoid, "lon") : 0);
    const metricValue = Number(row?.[metricKey]);
    const riskNorm = normalize(Number(row.risk_score), riskStats?.min, riskStats?.max);
    const baseRadius = 4 + riskNorm * 5;
    const selected = geoid === String(state.selectedGeoid);
    const marker = L.circleMarker([lat, lon], {
      radius: selected ? baseRadius + 2 : baseRadius,
      fillColor: mapColor(metricValue, stats),
      color: selected ? "#111827" : "#0b132b",
      weight: selected ? 2.2 : 0.9,
      fillOpacity: selected ? 0.96 : 0.78,
    });
    marker.bindTooltip(
      `
        ${row.municipality || geoid}<br/>
        ${row.county || "Unknown"} County<br/>
        ${metric.label}: ${Number.isFinite(metricValue) ? metric.format(metricValue) : "N/A"}<br/>
        Risk: ${(Number(row.risk_score) * 100).toFixed(1)}%
      `,
      { direction: "top" },
    );
    marker.on("click", () => {
      state.selectedGeoid = geoid;
      renderDetail();
      renderMap();
      activateTab("tab-detail");
    });
    marker.addTo(state.markersLayer);
  }

  renderMapLegend(metricKey, stats);
  const scope = document.getElementById("map-scope");
  if (scope) {
    const county = state.dashboard?.row_scope_county || "All counties";
    const rowsReturned = Number(state.dashboard?.rows_returned || rows.length);
    const rowsTotal = Number(state.dashboard?.rows_total_available || rows.length);
    const limited = rowsReturned < rowsTotal;
    const limitHint = limited ? ` (showing top ${rowsReturned} by risk)` : "";
    scope.textContent = `Scope: ${county} • ${rowsReturned} loaded / ${rowsTotal} available${limitHint}`;
  }
}

function trendSvg(values) {
  const v = (values || []).map((x) => Number(x)).filter((x) => Number.isFinite(x));
  if (!v.length) return `<div class="muted">No quarterly trend available.</div>`;

  const w = 600;
  const h = 140;
  const pad = 16;
  const min = Math.min(...v);
  const max = Math.max(...v);
  const span = Math.max(max - min, 1e-6);
  const points = v
    .map((y, i) => {
      const x = pad + (i * (w - 2 * pad)) / Math.max(v.length - 1, 1);
      const yy = h - pad - ((y - min) / span) * (h - 2 * pad);
      return `${x},${yy}`;
    })
    .join(" ");

  return `
    <svg class="trend" viewBox="0 0 ${w} ${h}" preserveAspectRatio="none">
      <polyline points="${points}" fill="none" stroke="#006d77" stroke-width="2" />
    </svg>
  `;
}

function populateDetailSelector() {
  const select = document.getElementById("detail-geoid");
  const rows = state.sortedRows;
  select.replaceChildren();
  for (const row of rows) {
    const option = document.createElement("option");
    option.value = String(row.geoid);
    option.textContent = `${row.county || "Unknown"} • ${row.municipality || "Area"} (${row.geoid})`;
    select.appendChild(option);
  }
  if (rows.length) {
    const selected = rows.some((row) => String(row.geoid) === String(state.selectedGeoid))
      ? String(state.selectedGeoid)
      : String(rows[0].geoid);
    state.selectedGeoid = selected;
    select.value = selected;
  }
}

function selectRelativeArea(step) {
  const rows = state.sortedRows;
  if (rows.length < 2) return;
  const currentIdx = Math.max(
    0,
    rows.findIndex((row) => String(row.geoid) === String(state.selectedGeoid)),
  );
  const nextIdx = (currentIdx + step + rows.length) % rows.length;
  state.selectedGeoid = String(rows[nextIdx].geoid);
  renderDetail();
  renderMap();
}

function renderDetail() {
  const container = document.getElementById("detail-content");
  const rows = state.sortedRows;
  if (!rows.length) {
    container.innerHTML = `<div class="panel">No data loaded.</div>`;
    return;
  }
  const row = rows.find((r) => String(r.geoid) === String(state.selectedGeoid)) || rows[0];
  state.selectedGeoid = String(row.geoid);
  document.getElementById("detail-geoid").value = state.selectedGeoid;

  const prevBtn = document.getElementById("detail-prev");
  const nextBtn = document.getElementById("detail-next");
  prevBtn.disabled = rows.length < 2;
  nextBtn.disabled = rows.length < 2;

  const drivers = row.top_drivers || [];
  const brief = state.dashboard.policy_briefs?.[String(row.geoid)] || "No policy brief for this block under current constraints.";

  container.innerHTML = `
    <div class="panel">
      <h2>${row.municipality || row.geoid} (${row.geoid})</h2>
      <p class="muted">${row.county || "Unknown county"} County • Risk band ${row.risk_band}</p>
      <div class="cards">
        <div class="card"><div class="label">Risk Score</div><div class="value">${(Number(row.risk_score) * 100).toFixed(1)}%</div></div>
        <div class="card"><div class="label">Uncertainty</div><div class="value">${Number(row.risk_uncertainty || 0).toFixed(3)}</div></div>
        <div class="card"><div class="label">Replacement Cost</div><div class="value">${fmtMoney(row.replacement_cost)}</div></div>
        <div class="card"><div class="label">Minority Share</div><div class="value">${(Number(row.minority_share || 0) * 100).toFixed(1)}%</div></div>
      </div>
      <h3>Top Drivers</h3>
      <ul>
        ${drivers.map((d) => `<li>${d.feature}: ${Number(d.score).toFixed(3)}</li>`).join("") || "<li>No drivers available.</li>"}
      </ul>
      <h3>Historical Lead Trend (Quarterly)</h3>
      ${trendSvg(row.lead_trend)}
      <h3>Policy Brief</h3>
      <pre>${brief}</pre>
    </div>
  `;
  renderAIStatus();
}

function renderAIStatus() {
  const statusEl = document.getElementById("ai-status");
  const selectedEl = document.getElementById("ai-selected-geoid");
  if (!statusEl || !selectedEl) return;
  const geoid = state.selectedGeoid || "None";
  selectedEl.textContent = geoid;

  const enabled = Boolean(state.aiStatus?.enabled);
  const model = state.aiStatus?.model || "not configured";
  if (enabled) {
    statusEl.textContent = `AI enabled. Model: ${model}.`;
  } else {
    statusEl.textContent =
      "AI key not configured; copilot uses deterministic fallback summaries. Set OPENAI_API_KEY to enable LLM responses.";
  }
}

function currentCopilotRequest(question) {
  return {
    geoid: String(state.selectedGeoid || ""),
    question: String(question || "").trim(),
    budget: Number(document.getElementById("budget").value),
    fairness_tolerance: Number(document.getElementById("fairness").value),
    min_county_coverage: 0,
    optimizer_method: String(document.getElementById("optimizer").value),
  };
}

async function askAICopilot(question) {
  const answerEl = document.getElementById("ai-answer");
  const payload = currentCopilotRequest(question);
  if (!payload.geoid) {
    answerEl.textContent = "Select an area first.";
    return;
  }
  if (!payload.question) {
    answerEl.textContent = "Enter a question first.";
    return;
  }

  answerEl.textContent = "Generating answer...";
  const res = await fetch("/api/ai/copilot", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    const errText = await res.text();
    throw new Error(`ai copilot failed: ${res.status} ${errText}`);
  }
  const body = await res.json();
  const mode = body.ai_used ? "LLM" : "Fallback";
  const model = body.model ? ` (${body.model})` : "";
  const reason = body.fallback_reason ? `\n\nReason: ${body.fallback_reason}` : "";
  answerEl.textContent = `[${mode}${model}]\n\n${body.answer}${reason}`;
}

function renderFairness() {
  const fair = state.dashboard?.fairness_comparison;
  if (!fair) return;
  const withFair = fair.with_fairness.summary;
  const noFair = fair.without_fairness.summary;
  const cards = document.getElementById("fairness-summary");
  cards.innerHTML = `
    <div class="card"><div class="label">With Fairness • Selected</div><div class="value">${withFair.selected_count}</div></div>
    <div class="card"><div class="label">With Fairness • Cost</div><div class="value">${fmtMoney(withFair.total_cost)}</div></div>
    <div class="card"><div class="label">With Fairness • Minority Share</div><div class="value">${(Number(withFair.achieved_minority_share) * 100).toFixed(1)}%</div></div>
    <div class="card"><div class="label">Without Fairness • Selected</div><div class="value">${noFair.selected_count}</div></div>
    <div class="card"><div class="label">Without Fairness • Cost</div><div class="value">${fmtMoney(noFair.total_cost)}</div></div>
    <div class="card"><div class="label">Without Fairness • Minority Share</div><div class="value">${(Number(noFair.achieved_minority_share) * 100).toFixed(1)}%</div></div>
  `;

  const samePlan =
    Number(withFair.selected_count) === Number(noFair.selected_count) &&
    Math.abs(Number(withFair.total_risk_reduced) - Number(noFair.total_risk_reduced)) < 1e-6 &&
    Math.abs(Number(withFair.achieved_minority_share) - Number(noFair.achieved_minority_share)) < 1e-6;
  const noteEl = document.getElementById("fairness-note");
  const mode = fair.fairness_mode || "requested_tolerance";
  if (mode === "stress_override") {
    noteEl.textContent =
      `Requested fairness setting was non-binding, so dashboard switched to a stricter fairness stress target (${Number(fair.stress_target || 0).toFixed(2)}) to show tradeoffs.`;
  } else if (samePlan) {
    noteEl.textContent =
      "Fairness constraint is currently non-binding at these settings (with/without fairness pick the same plan). Try lower fairness tolerance or different budget to stress-test tradeoffs.";
  } else {
    noteEl.textContent = "Fairness constraint changed the selected replacement plan.";
  }

  const rows = fair.county_spend_comparison || [];
  document.getElementById("county-table").innerHTML = `
    <table>
      <thead>
        <tr>
          <th>County</th>
          <th>With Fairness</th>
          <th>Without Fairness</th>
          <th>Delta</th>
        </tr>
      </thead>
      <tbody>
        ${
          rows
            .map(
              (r) => `
                <tr>
                  <td>${r.county}</td>
                  <td>${fmtMoney(r.with_fairness_spend)}</td>
                  <td>${fmtMoney(r.without_fairness_spend)}</td>
                  <td>${fmtMoney(r.spend_delta)}</td>
                </tr>
              `,
            )
            .join("") || '<tr><td colspan="4">No county spending rows.</td></tr>'
        }
      </tbody>
    </table>
  `;
}

function renderPerformance() {
  const metrics = state.dashboard?.comparison_metrics;
  const cv = state.dashboard?.cv_metrics;
  if (!metrics || !cv) return;
  const cards = document.getElementById("perf-summary");
  cards.innerHTML = `
    <div class="card"><div class="label">CV Historical Acc</div><div class="value">${(Number(cv.historical_accuracy_mean) * 100).toFixed(1)}%</div></div>
    <div class="card"><div class="label">CV Fusion Acc</div><div class="value">${(Number(cv.fusion_accuracy_mean) * 100).toFixed(1)}%</div></div>
    <div class="card"><div class="label">CV Graph Acc</div><div class="value">${(Number(cv.graph_accuracy_mean) * 100).toFixed(1)}%</div></div>
    <div class="card"><div class="label">CV Fusion AUROC</div><div class="value">${Number(cv.fusion_auroc_mean).toFixed(3)}</div></div>
    <div class="card"><div class="label">CV Fusion AUPRC</div><div class="value">${Number(cv.fusion_auprc_mean).toFixed(3)}</div></div>
    <div class="card"><div class="label">In-Sample (Snapshot) Acc</div><div class="value">${(metrics.model.accuracy * 100).toFixed(1)}%</div></div>
  `;

  const rows = state.benchmark?.ablation_accuracy_table || [];
  document.getElementById("ablation-table").innerHTML = `
    <table>
      <thead>
        <tr>
          <th>Model</th>
          <th>Accuracy Mean</th>
          <th>Accuracy Std</th>
          <th>AUROC</th>
          <th>AUPRC</th>
        </tr>
      </thead>
      <tbody>
        ${
          rows
            .map(
              (r) => `
                <tr>
                  <td>${r.model}</td>
                  <td>${Number(r.accuracy_mean).toFixed(3)}</td>
                  <td>${Number(r.accuracy_std).toFixed(3)}</td>
                  <td>${Number(r.auroc_mean).toFixed(3)}</td>
                  <td>${Number(r.auprc_mean).toFixed(3)}</td>
                </tr>
              `,
            )
            .join("") || '<tr><td colspan="5">No ablation data.</td></tr>'
        }
      </tbody>
    </table>
  `;
}

function activateTab(id) {
  document.querySelectorAll(".tab").forEach((el) => el.classList.remove("active"));
  document.querySelectorAll(".tab-btn").forEach((el) => el.classList.remove("active"));
  document.getElementById(id).classList.add("active");
  document.querySelector(`.tab-btn[data-tab="${id}"]`)?.classList.add("active");
}

async function refreshAll() {
  document.getElementById("budget-label").textContent = fmtMoney(document.getElementById("budget").value);
  document.getElementById("fairness-label").textContent = Number(document.getElementById("fairness").value).toFixed(2);
  await fetchDashboard();
  if (!state.aiStatus) await fetchAIStatus();
  populateCountyFilter();
  if (!state.benchmark) await fetchBenchmark();

  const rows = state.dashboard?.rows || [];
  state.sortedRows = sortedRows(rows);
  if (!rows.length) {
    renderAIStatus();
    return;
  }

  const exists = rows.some((r) => String(r.geoid) === String(state.selectedGeoid));
  if (!exists) state.selectedGeoid = String(rows[0].geoid);
  populateDetailSelector();

  renderMap();
  renderDetail();
  renderFairness();
  renderPerformance();
}

function wireEvents() {
  document.querySelectorAll(".tab-btn").forEach((btn) => {
    btn.addEventListener("click", () => activateTab(btn.dataset.tab));
  });

  document.getElementById("refresh-btn").addEventListener("click", () => refreshAll().catch(console.error));
  document.getElementById("budget").addEventListener("input", () => {
    document.getElementById("budget-label").textContent = fmtMoney(document.getElementById("budget").value);
  });
  document.getElementById("fairness").addEventListener("input", () => {
    document.getElementById("fairness-label").textContent = Number(document.getElementById("fairness").value).toFixed(2);
  });
  document.getElementById("county-filter").addEventListener("change", () => refreshAll().catch(console.error));
  document.getElementById("map-metric").addEventListener("change", () => renderMap());
  document.getElementById("map-jitter").addEventListener("change", () => renderMap());
  document.getElementById("detail-geoid").addEventListener("change", (event) => {
    state.selectedGeoid = String(event.target.value || "");
    renderDetail();
    renderMap();
  });
  document.getElementById("detail-prev").addEventListener("click", () => selectRelativeArea(-1));
  document.getElementById("detail-next").addEventListener("click", () => selectRelativeArea(1));
  document.getElementById("ai-ask-btn").addEventListener("click", () => {
    const question = document.getElementById("ai-question").value;
    askAICopilot(question).catch((err) => {
      console.error(err);
      document.getElementById("ai-answer").textContent = `Failed to get AI response: ${err.message}`;
    });
  });
  document.getElementById("ai-brief-btn").addEventListener("click", () => {
    const question =
      "Create a concise policy brief for this selected area with immediate actions (0-3 months), medium-term actions (3-12 months), and key uncertainty/fairness caveats.";
    document.getElementById("ai-question").value = question;
    askAICopilot(question).catch((err) => {
      console.error(err);
      document.getElementById("ai-answer").textContent = `Failed to get AI response: ${err.message}`;
    });
  });
}

wireEvents();
refreshAll().catch((err) => {
  console.error(err);
  alert(`Failed to load dashboard: ${err.message}`);
});
