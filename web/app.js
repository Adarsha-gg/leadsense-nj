const state = {
  dashboard: null,
  benchmark: null,
  selectedGeoid: null,
  map: null,
  markersLayer: null,
};

function colorForRisk(score) {
  if (score >= 0.75) return "#a4133c";
  if (score >= 0.45) return "#cc4b00";
  return "#2a9d8f";
}

function fmtMoney(n) {
  return `$${Number(n || 0).toLocaleString(undefined, { maximumFractionDigits: 0 })}`;
}

function qs() {
  const budget = document.getElementById("budget").value;
  const fairness = document.getElementById("fairness").value;
  const optimizer = document.getElementById("optimizer").value;
  return `budget=${budget}&fairness_tolerance=${fairness}&optimizer_method=${optimizer}`;
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

function ensureMap() {
  if (state.map) return;
  state.map = L.map("risk-map").setView([39.85, -74.7], 8);
  L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>',
  }).addTo(state.map);
  state.markersLayer = L.layerGroup().addTo(state.map);
}

function renderMap() {
  ensureMap();
  state.markersLayer.clearLayers();
  const rows = state.dashboard?.rows || [];
  for (const row of rows) {
    const lat = Number(row.lat);
    const lon = Number(row.lon);
    if (!Number.isFinite(lat) || !Number.isFinite(lon)) continue;

    const marker = L.circleMarker([lat, lon], {
      radius: 8,
      fillColor: colorForRisk(Number(row.risk_score)),
      color: "#0b132b",
      weight: 1,
      fillOpacity: 0.85,
    });
    marker.bindTooltip(
      `${row.municipality || row.geoid}<br/>Risk ${(Number(row.risk_score) * 100).toFixed(1)}%`,
      { direction: "top" },
    );
    marker.on("click", () => {
      state.selectedGeoid = String(row.geoid);
      renderDetail();
      activateTab("tab-detail");
    });
    marker.addTo(state.markersLayer);
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

function renderDetail() {
  const container = document.getElementById("detail-content");
  const rows = state.dashboard?.rows || [];
  if (!rows.length) {
    container.innerHTML = `<div class="panel">No data loaded.</div>`;
    return;
  }
  const row = rows.find((r) => String(r.geoid) === String(state.selectedGeoid)) || rows[0];
  state.selectedGeoid = String(row.geoid);

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
  if (!metrics) return;
  const cards = document.getElementById("perf-summary");
  cards.innerHTML = `
    <div class="card"><div class="label">Historical Accuracy</div><div class="value">${(metrics.historical.accuracy * 100).toFixed(1)}%</div></div>
    <div class="card"><div class="label">Model Accuracy</div><div class="value">${(metrics.model.accuracy * 100).toFixed(1)}%</div></div>
    <div class="card"><div class="label">Model AUROC</div><div class="value">${Number(metrics.model_auroc).toFixed(3)}</div></div>
    <div class="card"><div class="label">Model AUPRC</div><div class="value">${Number(metrics.model_auprc).toFixed(3)}</div></div>
    <div class="card"><div class="label">Model ECE</div><div class="value">${Number(metrics.model_ece).toFixed(3)}</div></div>
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
  if (!state.benchmark) await fetchBenchmark();

  const rows = state.dashboard?.rows || [];
  if (!rows.length) return;
  if (!state.selectedGeoid) state.selectedGeoid = String(rows[0].geoid);

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
}

wireEvents();
refreshAll().catch((err) => {
  console.error(err);
  alert(`Failed to load dashboard: ${err.message}`);
});
