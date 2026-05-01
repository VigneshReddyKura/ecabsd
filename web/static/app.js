/* ────────────────────────────────────────────
   ECABSD Web UI — Frontend JavaScript
   ──────────────────────────────────────────── */

const API_BASE = window.location.origin;

// ── State ──────────────────────────────────────
let currentResults = null;
let probChart = null;
let showAllResidues = false;

// ── DOM refs ───────────────────────────────────
const dropzone        = document.getElementById('dropzone');
const fileInput       = document.getElementById('file-input');
const fileNameDisplay = document.getElementById('file-name-display');
const predictBtn      = document.getElementById('predict-btn');
const chainA          = document.getElementById('chain-a');
const chainB          = document.getElementById('chain-b');
const threshold       = document.getElementById('threshold');
const thresholdVal    = document.getElementById('threshold-val');
const loadingOverlay  = document.getElementById('loading-overlay');
const loadingStep     = document.getElementById('loading-step');
const resultsSection  = document.getElementById('results-section');
const resultsMeta     = document.getElementById('results-meta');
const summaryGrid     = document.getElementById('summary-grid');
const resultsTbody    = document.getElementById('results-tbody');
const errorToast      = document.getElementById('error-toast');
const toastMsg        = document.getElementById('toast-msg');
const toastClose      = document.getElementById('toast-close');
const exportCsvBtn    = document.getElementById('export-csv-btn');
const exportJsonBtn   = document.getElementById('export-json-btn');
const exportPymolBtn  = document.getElementById('export-pymol-btn');
const filterBinding   = document.getElementById('filter-binding');
const filterAll       = document.getElementById('filter-all');

let selectedFile = null;

// ── Threshold slider ───────────────────────────
threshold.addEventListener('input', () => {
  thresholdVal.textContent = parseFloat(threshold.value).toFixed(2);
});

// ── File selection ─────────────────────────────
function handleFile(file) {
  if (!file) return;
  if (!file.name.endsWith('.pdb') && !file.name.endsWith('.PDB')) {
    showError('Please upload a .pdb file.');
    return;
  }
  selectedFile = file;
  fileNameDisplay.textContent = file.name;
  dropzone.classList.add('has-file');
  predictBtn.disabled = false;
}

fileInput.addEventListener('change', (e) => handleFile(e.target.files[0]));
dropzone.addEventListener('click', (e) => {
  if (!e.target.closest('.btn')) fileInput.click();
});

// Drag & Drop
dropzone.addEventListener('dragover', (e) => { e.preventDefault(); dropzone.classList.add('drag-over'); });
dropzone.addEventListener('dragleave', () => dropzone.classList.remove('drag-over'));
dropzone.addEventListener('drop', (e) => {
  e.preventDefault();
  dropzone.classList.remove('drag-over');
  handleFile(e.dataTransfer.files[0]);
});

// ── Predict ────────────────────────────────────
predictBtn.addEventListener('click', runPrediction);

async function runPrediction() {
  if (!selectedFile) return;

  const steps = [
    'Building residue graph…',
    'Running GCN encoder…',
    'Applying SE(3) refinement…',
    'Computing cross-attention…',
    'Classifying binding residues…',
  ];
  let stepIdx = 0;

  showLoading(true, steps[0]);
  const stepInterval = setInterval(() => {
    stepIdx = Math.min(stepIdx + 1, steps.length - 1);
    loadingStep.textContent = steps[stepIdx];
  }, 1200);

  try {
    const formData = new FormData();
    formData.append('pdb_file', selectedFile);
    formData.append('chain_a', chainA.value.trim() || 'A');
    formData.append('chain_b', chainB.value.trim());
    formData.append('threshold', threshold.value);

    const response = await fetch(`${API_BASE}/predict`, {
      method: 'POST',
      body: formData,
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.detail || 'Prediction failed');
    }

    currentResults = data;
    renderResults(data);

  } catch (err) {
    showError(err.message || 'An unexpected error occurred.');
  } finally {
    clearInterval(stepInterval);
    showLoading(false);
  }
}

// ── Render Results ─────────────────────────────
function renderResults(data) {
  // Show section
  resultsSection.hidden = false;
  resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });

  // Meta
  resultsMeta.textContent =
    `${data.pdb_file} · Chain ${data.chain_a}${data.chain_b ? ' × ' + data.chain_b : ''} · threshold=${data.threshold}`;

  // Summary cards
  const bindingPct = data.total_residues > 0
    ? ((data.binding_residues_count / data.total_residues) * 100).toFixed(1)
    : '0.0';
  const avgProb = data.residues.length > 0
    ? (data.residues.reduce((s, r) => s + r.probability, 0) / data.residues.length).toFixed(3)
    : '0';
  const maxProb = data.residues.length > 0
    ? Math.max(...data.residues.map(r => r.probability)).toFixed(3)
    : '0';

  summaryGrid.innerHTML = `
    <div class="summary-card fade-in">
      <div class="summary-label">Total Residues</div>
      <div class="summary-value v-primary">${data.total_residues}</div>
    </div>
    <div class="summary-card fade-in">
      <div class="summary-label">Binding Residues</div>
      <div class="summary-value v-green">${data.binding_residues_count}</div>
    </div>
    <div class="summary-card fade-in">
      <div class="summary-label">Binding %</div>
      <div class="summary-value v-cyan">${bindingPct}%</div>
    </div>
    <div class="summary-card fade-in">
      <div class="summary-label">Avg Probability</div>
      <div class="summary-value v-yellow">${avgProb}</div>
    </div>
    <div class="summary-card fade-in">
      <div class="summary-label">Max Probability</div>
      <div class="summary-value v-primary">${maxProb}</div>
    </div>
  `;

  // Chart
  renderChart(data.residues, data.threshold);

  // Table
  renderTable(data.residues, showAllResidues);
}

// ── Chart ──────────────────────────────────────
function renderChart(residues, threshold) {
  const labels = residues.map(r => `${r.resname}${r.resid}`);
  const probs  = residues.map(r => r.probability);
  const colors = residues.map(r =>
    r.is_binding
      ? 'rgba(16, 185, 129, 0.85)'
      : 'rgba(99, 102, 241, 0.4)'
  );
  const borderColors = residues.map(r =>
    r.is_binding ? 'rgba(16, 185, 129, 1)' : 'rgba(99,102,241,0.6)'
  );

  const ctx = document.getElementById('prob-chart').getContext('2d');

  if (probChart) probChart.destroy();

  probChart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        label: 'Binding Probability',
        data: probs,
        backgroundColor: colors,
        borderColor: borderColors,
        borderWidth: 1,
        borderRadius: 2,
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 600, easing: 'easeOutQuart' },
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            title: (items) => items[0].label,
            label: (item) => {
              const r = residues[item.dataIndex];
              return [
                `Probability: ${r.probability.toFixed(4)}`,
                `Status: ${r.is_binding ? '✓ Binding' : '– Non-binding'}`,
              ];
            },
          },
          backgroundColor: '#0f1420',
          borderColor: 'rgba(255,255,255,0.1)',
          borderWidth: 1,
          titleColor: '#e2e8f0',
          bodyColor: '#94a3b8',
          padding: 12,
        },
        annotation: {}
      },
      scales: {
        x: {
          ticks: {
            color: '#475569',
            font: { size: 9, family: 'JetBrains Mono' },
            maxRotation: 90,
            maxTicksLimit: Math.min(residues.length, 40),
          },
          grid: { color: 'rgba(255,255,255,0.04)' },
        },
        y: {
          min: 0, max: 1,
          ticks: { color: '#64748b', font: { size: 10 } },
          grid: { color: 'rgba(255,255,255,0.06)' },
        },
      },
    },
  });

  // Draw threshold line manually after render
  const thresholdPlugin = {
    id: 'thresholdLine',
    afterDraw(chart) {
      const { ctx, chartArea, scales } = chart;
      const y = scales.y.getPixelForValue(threshold);
      ctx.save();
      ctx.setLineDash([6, 4]);
      ctx.strokeStyle = 'rgba(244, 63, 94, 0.7)';
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(chartArea.left, y);
      ctx.lineTo(chartArea.right, y);
      ctx.stroke();
      ctx.restore();
    }
  };
  probChart.options.plugins.thresholdLine = {};
  Chart.register(thresholdPlugin);
  probChart.update();
}

// ── Table ──────────────────────────────────────
function renderTable(residues, showAll) {
  const filtered = showAll
    ? residues
    : residues.filter(r => r.is_binding);

  if (filtered.length === 0) {
    resultsTbody.innerHTML = `<tr><td colspan="6" style="text-align:center;padding:28px;color:var(--text-muted)">No ${showAll ? '' : 'binding '}residues found.</td></tr>`;
    return;
  }

  resultsTbody.innerHTML = filtered.map(r => {
    const prob = r.probability;
    const pct  = (prob * 100).toFixed(1);
    const color = prob >= 0.75
      ? '#10b981'
      : prob >= 0.5
        ? '#06b6d4'
        : '#6366f1';

    const badge = r.is_binding
      ? `<span class="badge-binding">✓ Binding</span>`
      : `<span class="badge-nonbinding">Non-binding</span>`;

    return `
      <tr>
        <td>${r.index}</td>
        <td style="color:var(--text);font-weight:600">${r.resname}</td>
        <td>${r.resid}</td>
        <td>${r.chain}</td>
        <td>
          <div class="prob-bar-wrap">
            <div class="prob-bar">
              <div class="prob-bar-fill" style="width:${pct}%;background:${color}"></div>
            </div>
            <span style="color:${color};min-width:52px">${prob.toFixed(4)}</span>
          </div>
        </td>
        <td>${badge}</td>
      </tr>`;
  }).join('');
}

// Filter buttons
filterBinding.addEventListener('click', () => {
  showAllResidues = false;
  filterBinding.classList.add('active');
  filterAll.classList.remove('active');
  if (currentResults) renderTable(currentResults.residues, false);
});
filterAll.addEventListener('click', () => {
  showAllResidues = true;
  filterAll.classList.add('active');
  filterBinding.classList.remove('active');
  if (currentResults) renderTable(currentResults.residues, true);
});

// ── Export ─────────────────────────────────────
function downloadJSON(content, filename) {
  const blob = new Blob([JSON.stringify(content, null, 2)], { type: 'application/json' });
  triggerDownload(blob, filename);
}
function downloadText(content, filename) {
  const blob = new Blob([content], { type: 'text/plain' });
  triggerDownload(blob, filename);
}
function triggerDownload(blob, filename) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = filename; a.click();
  URL.revokeObjectURL(url);
}

exportJsonBtn.addEventListener('click', () => {
  if (!currentResults) return;
  downloadJSON(currentResults, `ecabsd_${currentResults.pdb_file.replace('.pdb','')}.json`);
});

exportCsvBtn.addEventListener('click', () => {
  if (!currentResults) return;
  const header = 'index,resname,resid,chain,probability,is_binding\n';
  const rows = currentResults.residues.map(r =>
    `${r.index},${r.resname},${r.resid},${r.chain},${r.probability.toFixed(6)},${r.is_binding ? 1 : 0}`
  ).join('\n');
  downloadText(header + rows, `ecabsd_${currentResults.pdb_file.replace('.pdb','')}.csv`);
});

exportPymolBtn.addEventListener('click', () => {
  if (!currentResults) return;
  const d = currentResults;
  const bindingIds = d.residues.filter(r => r.is_binding).map(r => r.resid).join('+');
  let pml = `# ECABSD Binding Site — ${d.pdb_file} Chain ${d.chain_a}\n`;
  pml += `load ${d.pdb_file}, protein\nhide everything\nshow cartoon, protein\nbg_color white\n\n`;
  pml += `color grey80, chain ${d.chain_a}\n\n`;
  d.residues.forEach(r => {
    const p = r.probability;
    const red   = p < 0.5 ? Math.round(p * 2 * 255) : 255;
    const green = p < 0.5 ? 255 : Math.round((1 - (p - 0.5) * 2) * 255);
    pml += `color 0x${red.toString(16).padStart(2,'0')}${green.toString(16).padStart(2,'0')}00, chain ${d.chain_a} and resi ${r.resid}\n`;
  });
  if (bindingIds) {
    pml += `\nselect binding_site, chain ${d.chain_a} and resi ${bindingIds}\n`;
    pml += `show sticks, binding_site\nzoom binding_site\n`;
  }
  downloadText(pml, `ecabsd_${d.pdb_file.replace('.pdb','')}.pml`);
});

// ── Helpers ────────────────────────────────────
function showLoading(show, msg = '') {
  loadingOverlay.hidden = !show;
  if (msg) loadingStep.textContent = msg;
}

function showError(msg) {
  toastMsg.textContent = msg;
  errorToast.hidden = false;
  setTimeout(() => { errorToast.hidden = true; }, 6000);
}

toastClose.addEventListener('click', () => { errorToast.hidden = true; });
