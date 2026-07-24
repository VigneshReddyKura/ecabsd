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
const pdbId           = document.getElementById('pdb-id');
const thresholdAuto   = document.getElementById('threshold-auto');
const generateGradcamBtn = document.getElementById('generate-gradcam-btn');
const explainPlaceholderArea = document.getElementById('explain-placeholder-area');
const gradcamImgWrapper = document.getElementById('gradcam-img-wrapper');

let selectedFile = null;

// ── Threshold slider ───────────────────────────
threshold.addEventListener('input', () => {
  if (!thresholdAuto.checked) {
    thresholdVal.textContent = parseFloat(threshold.value).toFixed(2);
  }
});

// Auto Checkbox listener
thresholdAuto.addEventListener('change', () => {
  if (thresholdAuto.checked) {
    threshold.disabled = true;
    threshold.value = "0.58";
    thresholdVal.textContent = `Auto (0.58)`;
  } else {
    threshold.disabled = false;
    thresholdVal.textContent = parseFloat(threshold.value).toFixed(2);
  }
});

// ── File selection ─────────────────────────────
function handleFile(file) {
  if (!file) return;
  if (!file.name.endsWith('.pdb') && !file.name.endsWith('.PDB')) {
    showError('Please upload a .pdb file.');
    return;
  }
  selectedFile = file;
  pdbId.value = ''; // Clear PDB ID input
  fileNameDisplay.textContent = file.name;
  dropzone.classList.add('has-file');
  predictBtn.disabled = false;
}

const PDB_PRESETS = {
  '1AY7': { a: 'A', b: 'B' },
  '1BRS': { a: 'A', b: 'D' },
  '2PTC': { a: 'E', b: 'I' },
  '1CGI': { a: 'E', b: 'I' },
  '2SNI': { a: 'E', b: 'I' }
};

// PDB ID Input Listener — fires on type, paste, and change
function checkInputReady() {
  const val = pdbId.value.trim().toUpperCase();
  if (val.length === 4) {
    selectedFile = null;
    dropzone.classList.remove('has-file');
    fileNameDisplay.textContent = `PDB ID: ${val}`;
    predictBtn.disabled = false;
    
    // Auto-populate chains for known PDB presets
    if (PDB_PRESETS[val]) {
      chainA.value = PDB_PRESETS[val].a;
      chainB.value = PDB_PRESETS[val].b;
    }
  } else if (!selectedFile) {
    predictBtn.disabled = true;
    if (!val.length) fileNameDisplay.textContent = 'No file selected';
  }
}

pdbId.addEventListener('input',  checkInputReady);
pdbId.addEventListener('paste',  () => setTimeout(checkInputReady, 0));
pdbId.addEventListener('change', checkInputReady);


// Auto-capitalize chain inputs on typing
chainA.addEventListener('input', () => {
  chainA.value = chainA.value.toUpperCase();
});
chainB.addEventListener('input', () => {
  chainB.value = chainB.value.toUpperCase();
});

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
  if (!selectedFile && !pdbId.value.trim()) return;

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
    if (selectedFile) {
      formData.append('pdb_file', selectedFile);
    } else {
      formData.append('pdb_id', pdbId.value.trim().toUpperCase());
    }
    formData.append('chain_a', chainA.value.trim().toUpperCase() || 'A');
    formData.append('chain_b', chainB.value.trim().toUpperCase());
    formData.append('threshold', thresholdAuto.checked ? -1 : threshold.value);

    const response = await fetch(`${API_BASE}/predict`, {
      method: 'POST',
      body: formData,
    });

    let data = null;
    const text = await response.text();
    const contentType = response.headers.get("content-type");

    if (contentType && contentType.indexOf("application/json") !== -1) {
      try {
        data = text ? JSON.parse(text) : null;
      } catch (jsonErr) {
        console.error("JSON parsing error:", jsonErr, "Response text was:", text);
        throw new Error(`Failed to parse JSON response: ${text.substring(0, 120) || '(empty response)'}`);
      }
    } else {
      const cleanText = text ? text.replace(/<[^>]*>/g, '').trim() : '';
      const summaryText = cleanText ? cleanText.substring(0, 120) : response.statusText;
      throw new Error(`Server error (${response.status}): ${summaryText || 'Bad Gateway'}`);
    }

    if (!response.ok) {
      throw new Error((data && data.detail) ? data.detail : 'Prediction failed');
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
  // Sync auto threshold slider
  if (thresholdAuto.checked) {
    threshold.value = data.threshold;
    thresholdVal.textContent = `Auto (${data.threshold.toFixed(2)})`;
  }

  // Show section
  resultsSection.hidden = false;
  resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });

  // Meta
  resultsMeta.textContent =
    `${data.pdb_file} · Chain ${data.chain_a}${data.chain_b ? ' × ' + data.chain_b : ''} · threshold=${parseFloat(data.threshold).toFixed(4)}`;

  // Custom Alerts / Warnings
  const alertContainer = document.getElementById('results-alert-container');
  if (alertContainer) {
    alertContainer.style.display = 'none';
    alertContainer.innerHTML = '';
    
    if (data.is_1brs) {
      alertContainer.style.display = 'block';
      alertContainer.innerHTML = `
        <div style="background: rgba(239, 68, 68, 0.12); border-left: 4px solid var(--red); padding: 16px; border-radius: 6px; color: var(--text-dim); font-size: 0.9rem; line-height: 1.5;">
          <div style="font-weight: 700; color: var(--red); margin-bottom: 4px; display: flex; align-items: center; gap: 8px;">
            <span>⚠️</span> Prediction: Low-confidence underprediction
          </div>
          The PDB is valid, but the model assigned very low residue probabilities.<br/>
          This sample should be reviewed or tested with a stronger V3 model.
        </div>
      `;
    } else if (data.warning_msg || (data.max_prob && data.max_prob < 0.05)) {
      alertContainer.style.display = 'block';
      alertContainer.innerHTML = `
        <div style="background: rgba(245, 158, 11, 0.12); border-left: 4px solid var(--yellow); padding: 16px; border-radius: 6px; color: var(--text-dim); font-size: 0.9rem; line-height: 1.5;">
          <div style="font-weight: 700; color: var(--yellow); margin-bottom: 4px; display: flex; align-items: center; gap: 8px;">
            <span>⚠️</span> Low Model Confidence
          </div>
          Low model confidence. Prediction should be reviewed (max probability is ${parseFloat(data.max_prob || 0.0).toFixed(4)}).
        </div>
      `;
    }
  }

  // Summary cards
  const bindingPct = data.total_residues > 0
    ? ((data.binding_residues_count / data.total_residues) * 100).toFixed(1)
    : '0.0';
  const avgProb = data.residues.length > 0
    ? (data.residues.reduce((s, r) => s + r.probability, 0) / data.residues.length).toFixed(3)
    : '0';
  const maxProb = data.max_prob !== undefined
    ? parseFloat(data.max_prob).toFixed(3)
    : (data.residues.length > 0 ? Math.max(...data.residues.map(r => r.probability)).toFixed(3) : '0');

  // Confidence level class and styling
  const conf = data.confidence || 'High';
  let confColor = 'var(--green)';
  if (conf === 'Very Low') confColor = 'var(--red)';
  else if (conf === 'Low') confColor = 'var(--yellow)';
  else if (conf === 'Medium') confColor = 'var(--cyan)';

  const qualityCardHtml = `
    <div class="summary-card fade-in" style="display: flex; flex-direction: column; justify-content: center; min-height: 96px;">
      <div class="summary-label">Sample Classification</div>
      <div class="summary-value" style="font-size: 0.92rem; font-weight: 600; color: var(--text-dim); margin-top: 4px; line-height: 1.35;">${data.prediction_quality || 'Unknown'}</div>
    </div>
  `;

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
    <div class="summary-card fade-in">
      <div class="summary-label">Model Confidence</div>
      <div class="summary-value" style="color: ${confColor}; font-weight: 600;">${conf}</div>
    </div>
    ${qualityCardHtml}
  `;

  // Dynamic Recommendation
  const recPanel = document.getElementById('recommendation-panel');
  const recText = document.getElementById('recommendation-text');
  if (recPanel && recText && data.recommendation) {
    recPanel.style.display = 'block';
    recText.textContent = data.recommendation;
  } else if (recPanel) {
    recPanel.style.display = 'none';
  }

  // Explainability Cards (Heatmap & Grad-CAM)
  const explainCard = document.getElementById('explain-card');
  const heatmapImg = document.getElementById('heatmap-img');
  const heatmapContainer = document.getElementById('heatmap-container');
  const downloadHeatmapBtn = document.getElementById('download-heatmap-btn');
  const downloadGradcamBtn = document.getElementById('download-gradcam-btn');

  if (data.heatmap_url) {
    explainCard.style.display = 'block';
    if (heatmapContainer) heatmapContainer.style.display = 'block';
    
    const isBase64 = data.heatmap_url.startsWith('data:');
    heatmapImg.src = isBase64 ? data.heatmap_url : `${data.heatmap_url}?t=${new Date().getTime()}`;
    if (downloadHeatmapBtn) {
      downloadHeatmapBtn.href = data.heatmap_url;
      downloadHeatmapBtn.style.display = 'inline-block';
      downloadHeatmapBtn.download = `ecabsd_heatmap_${data.pdb_file.replace('.pdb','')}_chain_${data.chain_a}.png`;
    }

    // Reset Explainability UI state for the new prediction
    const explainPlaceholder = document.getElementById('explain-placeholder-area');
    const gradcamContainer = document.getElementById('gradcam-container');
    const attentionContainer = document.getElementById('attention-container');
    const overlapContainer = document.getElementById('overlap-container');
    const gradcamErrorMsg = document.getElementById('gradcam-error-msg');
    const gradcamImgWrapper = document.getElementById('gradcam-img-wrapper');
    const downloadGradcamBtn = document.getElementById('download-gradcam-btn');
    const generateGradcamBtn = document.getElementById('generate-gradcam-btn');

    if (explainPlaceholder) explainPlaceholder.style.display = 'block';
    if (gradcamContainer) gradcamContainer.style.display = 'none';
    if (attentionContainer) attentionContainer.style.display = 'none';
    if (overlapContainer) overlapContainer.style.display = 'none';
    if (gradcamErrorMsg) gradcamErrorMsg.style.display = 'none';
    if (gradcamImgWrapper) gradcamImgWrapper.style.display = 'none';
    if (downloadGradcamBtn) downloadGradcamBtn.style.display = 'none';

    if (generateGradcamBtn) {
      if (data.gradcam_allowed === false) {
        generateGradcamBtn.disabled = true;
        generateGradcamBtn.style.opacity = '0.5';
        generateGradcamBtn.title = 'GradCAM disabled: large protein (>200 residues) or low memory';
        generateGradcamBtn.textContent = '⚠ GradCAM unavailable';
      } else {
        generateGradcamBtn.disabled = false;
        generateGradcamBtn.style.opacity = '1.0';
        generateGradcamBtn.title = '';
        generateGradcamBtn.textContent = '⚡ Generate Explanations';
      }
    }
  } else {
    explainCard.style.display = 'none';
  }

  // Chart
  renderChart(data.residues, data.threshold);

  // Table
  renderTable(data.residues, showAllResidues);

  // 3D Molecular Viewer
  const viewerCard = document.getElementById('viewer-card');
  if (viewerCard) {
    viewerCard.style.display = 'block';
    
    if (data.pdb_content) {
      receptorPdbString = data.pdb_content;
      init3DViewer(receptorPdbString, data.chain_a, data.chain_b, data.residues);
    } else if (selectedFile) {
      let reader = new FileReader();
      reader.onload = function(e) {
        receptorPdbString = e.target.result;
        init3DViewer(receptorPdbString, data.chain_a, data.chain_b, data.residues);
      };
      reader.readAsText(selectedFile);
    } else {
      const pid = (pdbId ? pdbId.value.trim().toUpperCase() : '');
      if (pid) {
        fetch(`https://files.rcsb.org/download/${pid}.pdb`)
          .then(res => {
            if (!res.ok) throw new Error("Failed to fetch PDB from RCSB");
            return res.text();
          })
          .then(text => {
            receptorPdbString = text;
            init3DViewer(receptorPdbString, data.chain_a, data.chain_b, data.residues);
          })
          .catch(err => {
            console.error("3D Viewer PDB fetch failed:", err);
            showError("Could not fetch structure from RCSB for 3D Molecular Viewer.");
          });
      }
    }
  }
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

// ── Grad-CAM Explanation ───────────────────────
if (generateGradcamBtn) {
  generateGradcamBtn.addEventListener('click', async () => {
    if (!currentResults) return;

    generateGradcamBtn.disabled = true;
    generateGradcamBtn.textContent = 'Generating...';

    try {
      const formData = new FormData();
      if (selectedFile) {
        formData.append('pdb_file', selectedFile);
      } else {
        formData.append('pdb_id', pdbId.value.trim().toUpperCase());
      }
      formData.append('chain_a', currentResults.chain_a);
      if (currentResults.chain_b) {
        formData.append('chain_b', currentResults.chain_b);
      }
      formData.append('threshold', currentResults.threshold);

      const response = await fetch(`${API_BASE}/explain`, {
        method: 'POST',
        body: formData,
      });

      const text = await response.text();

      if (!response.ok) {
        const cleanText = text ? text.replace(/<[^>]*>/g, '').replace(/\s+/g, ' ').trim() : '';
        const summaryText = cleanText ? cleanText.substring(0, 200) : response.statusText;
        throw new Error(`Server error (${response.status}): ${summaryText || 'Error occurred'}`);
      }

      let data = null;
      try {
        data = text ? JSON.parse(text) : null;
      } catch (jsonErr) {
        console.error("JSON parsing error:", jsonErr, "Response text was:", text);
        throw new Error(`Failed to parse JSON response: ${text.substring(0, 120) || '(empty response)'}`);
      }

      if (data && data.error) {
        throw new Error(data.error);
      }

      if (data && data.status === 'success') {
        const gradcamImg = document.getElementById('gradcam-img');
        const downloadGradcamBtn = document.getElementById('download-gradcam-btn');
        const gradcamContainer = document.getElementById('gradcam-container');
        const gradcamErrorMsg = document.getElementById('gradcam-error-msg');
        const gradcamImgWrapper = document.getElementById('gradcam-img-wrapper');

        const attentionImg = document.getElementById('attention-img');
        const downloadAttentionBtn = document.getElementById('download-attention-btn');
        const attentionContainer = document.getElementById('attention-container');
        const attentionImgWrapper = document.getElementById('attention-img-wrapper');

        const overlapContainer = document.getElementById('overlap-container');
        const overlapText = document.getElementById('overlap-text');
        const explainPlaceholder = document.getElementById('explain-placeholder-area');

        // Hide placeholder banner
        if (explainPlaceholder) explainPlaceholder.style.display = 'none';

        // 1. Render Grad-CAM Saliency Map if available
        if (gradcamContainer) gradcamContainer.style.display = 'block';
        
        // Final Render-safe logic: check both gradcam_available and gradcam_image
        const isGradcamAvailable = (data.gradcam_available !== false) && !!data.gradcam_image;

        if (isGradcamAvailable) {
          if (gradcamErrorMsg) gradcamErrorMsg.style.display = 'none';
          if (gradcamImgWrapper) {
            gradcamImgWrapper.style.display = 'block';
            gradcamImg.src = data.gradcam_image;
          }
          if (downloadGradcamBtn) {
            downloadGradcamBtn.href = data.gradcam_image;
            downloadGradcamBtn.style.display = 'inline-block';
            downloadGradcamBtn.download = `ecabsd_gradcam_${currentResults.pdb_file.replace('.pdb','')}_chain_${currentResults.chain_a}.png`;
          }
          currentResults.gradcam_scores = data.gradcam_scores;
        } else {
          // Show Grad-CAM error fallback message
          if (gradcamImgWrapper) gradcamImgWrapper.style.display = 'none';
          if (downloadGradcamBtn) downloadGradcamBtn.style.display = 'none';
          if (gradcamErrorMsg) {
            gradcamErrorMsg.style.display = 'block';
            gradcamErrorMsg.textContent = data.gradcam_message || data.gradcam_error || "Grad-CAM skipped due to low memory. Try smaller sample or run locally.";
          }
        }

        // 2. Render Attention Saliency Map
        if (data.attention_image) {
          if (attentionContainer) attentionContainer.style.display = 'block';
          if (attentionImgWrapper) {
            attentionImgWrapper.style.display = 'block';
            attentionImg.src = data.attention_image;
          }
          if (downloadAttentionBtn) {
            downloadAttentionBtn.href = data.attention_image;
            downloadAttentionBtn.style.display = 'inline-block';
            downloadAttentionBtn.download = `ecabsd_attention_${currentResults.pdb_file.replace('.pdb','')}_chain_${currentResults.chain_a}.png`;
          }
          currentResults.attention_scores = data.attention_scores;
        }

        // 3. Render Overlap Analysis
        if (overlapContainer && data.overlap_percentage !== undefined) {
          overlapContainer.style.display = 'block';
          
          let overlapMsg = "";
          if (data.gradcam_image) {
            const numOverlap = Math.round((data.overlap_percentage / 100) * 10);
            const expectedPct = data.random_overlap_percentage !== undefined ? data.random_overlap_percentage : 0.0;
            overlapMsg = `Calculated overlap of <strong>${data.overlap_percentage}%</strong> (${numOverlap}/10 residues) between the top 10 Grad-CAM residues and the predicted binding residues.<br/>` +
                         `<span style="display: block; margin-top: 6px; font-size: 0.75rem; color: var(--text-muted);">Expected (random) baseline overlap: <strong>${expectedPct}%</strong> (based on random selection under hypergeometric baseline).</span>`;
          } else {
            overlapMsg = `Grad-CAM calculation was bypassed due to server constraint fallback. Overlap analysis requires gradient maps.`;
          }
          if (overlapText) overlapText.innerHTML = overlapMsg;
        }

      } else {
        throw new Error('Explain endpoint did not return success status.');
      }
    } catch (err) {
      showError(err.message || 'An unexpected error occurred during explanation generation.');
      generateGradcamBtn.disabled = false;
      generateGradcamBtn.textContent = '⚡ Generate Explanations';
    }
  });
}

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

// ── 3D Molecular Viewer & Docking Overlay ──────
let viewer3D = null;
let receptorPdbString = null;
let dockedLigandPdbString = null;

function init3DViewer(pdbString, chainAId, chainBId, residues) {
  const container = document.getElementById('3d-viewer');
  if (!container || !pdbString) return;
  if (typeof $3Dmol === 'undefined') {
    console.error("3Dmol library not loaded");
    return;
  }
  
  container.innerHTML = '';
  
  try {
    viewer3D = $3Dmol.createViewer(container, { backgroundColor: '#0b0f19' });
    viewer3D.addModel(pdbString, 'pdb');
    
    applyViewerStyle('cartoon', chainAId, chainBId, residues);
    
    // Ensure WebGL canvas adjusts to non-zero container width/height after unhide
    setTimeout(() => {
      if (viewer3D) {
        viewer3D.resize();
        viewer3D.zoomTo();
        viewer3D.render();
      }
    }, 100);
  } catch (err) {
    console.error("Failed to initialize 3Dmol viewer:", err);
  }
  
  // Reset Docking Controls
  const dockLigandInput = document.getElementById('dock-ligand-input');
  const runDockBtn = document.getElementById('run-dock-btn');
  const dockStatus = document.getElementById('dock-status');
  
  if (dockLigandInput) dockLigandInput.value = '';
  if (runDockBtn) {
    runDockBtn.disabled = true;
    runDockBtn.textContent = '⚡ Run Docking';
  }
  if (dockStatus) dockStatus.textContent = '';
  dockedLigandPdbString = null;
  isSpinning = false;
  const spinBtn = document.getElementById('spin-btn');
  if (spinBtn) {
    spinBtn.classList.remove('active');
    spinBtn.textContent = '🔄 Auto-Rotate';
  }
  setActiveStyleButton('style-cartoon-btn');
}

let currentOpacity = 0.5; // Default 50% opacity for backbone/non-binding residues
let show5AInterface = false; // Toggle state for 5Å contact interface comparison
let currentStyleType = 'cartoon';

function applyViewerStyle(styleType, chainAId, chainBId, residues) {
  if (!viewer3D) return;
  currentStyleType = styleType;
  
  // Clear any existing surfaces and labels
  viewer3D.removeAllSurfaces();
  viewer3D.removeAllLabels();

  const opacity = parseFloat(currentOpacity);

  // 1. Style Target Chain A (Default 50% semi-transparent)
  let styleA = {};
  if (styleType === 'cartoon') styleA = { cartoon: { color: '#6366f1', opacity: opacity } };
  else if (styleType === 'sphere') styleA = { sphere: { color: '#64748b', opacity: opacity, scale: 0.7 } };
  else if (styleType === 'stick') styleA = { stick: { color: '#64748b', opacity: opacity, radius: 0.25 } };

  viewer3D.setStyle({ model: 0, chain: chainAId }, styleA);

  // 2. Style Partner Chain B (Default 50% semi-transparent)
  if (chainBId) {
    let styleB = {};
    if (styleType === 'cartoon') styleB = { cartoon: { color: '#0d9488', opacity: opacity } };
    else if (styleType === 'sphere') styleB = { sphere: { color: '#0d9488', opacity: opacity, scale: 0.7 } };
    else if (styleType === 'stick') styleB = { stick: { color: '#0d9488', opacity: opacity, radius: 0.25 } };
    viewer3D.setStyle({ model: 0, chain: chainBId }, styleB);
  }

  // 3. Highlight Actual 5Å Contact Interface Residues if comparison mode is enabled
  if (show5AInterface && residues) {
    residues.forEach(res => {
      if (res.is_interface) {
        let interfaceStyle = {};
        if (styleType === 'cartoon') interfaceStyle = { cartoon: { color: '#3b82f6', opacity: 0.9 }, stick: { color: '#3b82f6', radius: 0.35, opacity: 0.9 } };
        else if (styleType === 'sphere') interfaceStyle = { sphere: { color: '#3b82f6', opacity: 0.9, scale: 0.85 } };
        else if (styleType === 'stick') interfaceStyle = { stick: { color: '#3b82f6', opacity: 0.9, radius: 0.4 } };
        viewer3D.setStyle({ model: 0, chain: res.chain, resi: res.resid }, interfaceStyle);
      }
    });
  }

  // 4. Highlight Predicted Binding Residues in FULL 100% SOLID BRIGHT RED (#ef4444)
  if (residues) {
    residues.forEach(res => {
      if (res.is_binding) {
        let activeStyle = {};
        if (styleType === 'cartoon') activeStyle = { cartoon: { color: '#ef4444', opacity: 1.0 }, stick: { color: '#ef4444', radius: 0.45, opacity: 1.0 } };
        else if (styleType === 'sphere') activeStyle = { sphere: { color: '#ef4444', opacity: 1.0, scale: 1.0 } };
        else if (styleType === 'stick') activeStyle = { stick: { color: '#ef4444', opacity: 1.0, radius: 0.5 } };
        
        viewer3D.setStyle({ model: 0, chain: res.chain, resi: res.resid }, activeStyle);
      }
    });
  }

  // 5. Molecular Surface Rendering Mode
  if (styleType === 'surface') {
    try {
      // Set underlying cartoon at low opacity for shape outline
      viewer3D.setStyle({ model: 0, chain: chainAId }, { cartoon: { color: '#6366f1', opacity: 0.3 } });
      if (chainBId) {
        viewer3D.setStyle({ model: 0, chain: chainBId }, { cartoon: { color: '#0d9488', opacity: 0.3 } });
      }

      const predictedResids = new Set(residues ? residues.filter(r => r.is_binding).map(r => r.resid) : []);
      const interfaceResids = new Set(residues ? residues.filter(r => r.is_interface).map(r => r.resid) : []);

      // Add VDW surface with per-residue color mapping: predicted=RED, interface=BLUE, rest=SLATE/TEAL
      viewer3D.addSurface($3Dmol.SurfaceType.VDW, {
        opacity: opacity,
        colorscheme: {
          prop: 'resi',
          map: (atom) => {
            if (atom.chain === chainAId) {
              if (predictedResids.has(atom.resi)) return '#ef4444'; // Red for predicted
              if (show5AInterface && interfaceResids.has(atom.resi)) return '#3b82f6'; // Blue for 5A contact
              return '#475569'; // Slate for rest of Chain A
            }
            if (chainBId && atom.chain === chainBId) return '#0d9488'; // Teal for Chain B
            return '#64748b';
          }
        }
      }, { chain: [chainAId, chainBId].filter(Boolean) });

      // Highlight predicted sticks inside surface
      if (residues) {
        residues.forEach(res => {
          if (res.is_binding) {
            viewer3D.setStyle({ model: 0, chain: res.chain, resi: res.resid }, { stick: { color: '#ef4444', radius: 0.45, opacity: 1.0 } });
          }
        });
      }
    } catch (surfErr) {
      console.warn("Surface rendering fallback:", surfErr);
    }
  }

  // 6. Style Model 1 (Docked Ligand) if present
  if (viewer3D.models && viewer3D.models[1]) {
    viewer3D.setStyle({ model: 1 }, { stick: { color: '#fbbf24', radius: 0.35, opacity: 1.0 } });
  }

  // 7. Update overlay legend text & 3D text labels
  updateViewerLegend(chainAId, chainBId, opacity);
  addChainLabels3D(chainAId, chainBId);

  viewer3D.render();
}

function updateViewerLegend(chainAId, chainBId, opacity) {
  const pct = Math.round(opacity * 100);
  const legendA = document.getElementById('legend-chain-a-text');
  const legendB = document.getElementById('legend-chain-b-text');
  const legendBRow = document.getElementById('legend-chain-b-row');
  const legendInterfaceRow = document.getElementById('legend-interface-row');

  if (legendA) legendA.textContent = `Target Chain (${chainAId}) — ${pct}% Opacity`;
  if (legendBRow) {
    if (chainBId) {
      legendBRow.style.display = 'flex';
      if (legendB) legendB.textContent = `Partner Chain (${chainBId}) — ${pct}% Opacity`;
    } else {
      legendBRow.style.display = 'none';
    }
  }
  if (legendInterfaceRow) {
    legendInterfaceRow.style.display = show5AInterface ? 'flex' : 'none';
  }
}

function addChainLabels3D(chainAId, chainBId) {
  if (!viewer3D) return;
  
  try {
    const atoms = viewer3D.selectedAtoms({ model: 0 });
    if (!atoms || !atoms.length) return;

    let atomA = atoms.find(a => a.chain === chainAId && a.atom === 'CA') || atoms.find(a => a.chain === chainAId);
    let atomB = chainBId ? (atoms.find(a => a.chain === chainBId && a.atom === 'CA') || atoms.find(a => a.chain === chainBId)) : null;

    if (atomA) {
      viewer3D.addLabel(`Chain ${chainAId} (Target)`, {
        position: { x: atomA.x, y: atomA.y, z: atomA.z },
        backgroundColor: '#1e1b4b',
        backgroundOpacity: 0.85,
        fontColor: '#a5b4fc',
        fontSize: 11,
        fontFamily: 'Inter, sans-serif',
        borderThickness: 1,
        borderColor: '#6366f1'
      });
    }

    if (atomB) {
      viewer3D.addLabel(`Chain ${chainBId} (Partner)`, {
        position: { x: atomB.x, y: atomB.y, z: atomB.z },
        backgroundColor: '#042f2e',
        backgroundOpacity: 0.85,
        fontColor: '#5eead4',
        fontSize: 11,
        fontFamily: 'Inter, sans-serif',
        borderThickness: 1,
        borderColor: '#0d9488'
      });
    }
  } catch (lblErr) {
    console.warn("3D label addition error:", lblErr);
  }
}

function setActiveStyleButton(styleId) {
  ['style-cartoon-btn', 'style-surface-btn', 'style-sphere-btn', 'style-stick-btn'].forEach(id => {
    const btn = document.getElementById(id);
    if (btn) {
      if (id === styleId) btn.classList.add('active');
      else btn.classList.remove('active');
    }
  });
}

// Event Listeners for 3D Viewer spin & camera controls
let isSpinning = false;
const spinBtn = document.getElementById('spin-btn');
if (spinBtn) {
  spinBtn.addEventListener('click', () => {
    if (!viewer3D) return;
    isSpinning = !isSpinning;
    viewer3D.spin(isSpinning, 1.0);
    spinBtn.classList.toggle('active', isSpinning);
    spinBtn.textContent = isSpinning ? '⏸ Pause Rotate' : '🔄 Auto-Rotate';
  });
}

const resetViewBtn = document.getElementById('reset-view-btn');
if (resetViewBtn) {
  resetViewBtn.addEventListener('click', () => {
    if (!viewer3D) return;
    viewer3D.zoomTo();
    viewer3D.render();
  });
}

// Opacity Dropdown Selector
const opacitySelect = document.getElementById('opacity-select');
if (opacitySelect) {
  opacitySelect.addEventListener('change', (e) => {
    currentOpacity = parseFloat(e.target.value);
    if (currentResults) {
      applyViewerStyle(currentStyleType, currentResults.chain_a, currentResults.chain_b, currentResults.residues);
    }
  });
}

// 5Å Interface Toggle Button
const toggleInterfaceBtn = document.getElementById('toggle-interface-btn');
if (toggleInterfaceBtn) {
  toggleInterfaceBtn.addEventListener('click', () => {
    if (!currentResults) return;
    show5AInterface = !show5AInterface;
    toggleInterfaceBtn.classList.toggle('active', show5AInterface);
    toggleInterfaceBtn.style.background = show5AInterface ? 'rgba(59, 130, 246, 0.25)' : '';
    toggleInterfaceBtn.style.borderColor = show5AInterface ? '#3b82f6' : '';
    toggleInterfaceBtn.style.color = show5AInterface ? '#60a5fa' : '';
    applyViewerStyle(currentStyleType, currentResults.chain_a, currentResults.chain_b, currentResults.residues);
  });
}

// Surface Style Button
const styleSurfaceBtn = document.getElementById('style-surface-btn');
if (styleSurfaceBtn) {
  styleSurfaceBtn.addEventListener('click', () => {
    if (!currentResults) return;
    setActiveStyleButton('style-surface-btn');
    applyViewerStyle('surface', currentResults.chain_a, currentResults.chain_b, currentResults.residues);
  });
}

// Event Listeners for 3D Viewer styles
document.getElementById('style-cartoon-btn').addEventListener('click', () => {
  if (!currentResults) return;
  setActiveStyleButton('style-cartoon-btn');
  applyViewerStyle('cartoon', currentResults.chain_a, currentResults.chain_b, currentResults.residues);
});

document.getElementById('style-sphere-btn').addEventListener('click', () => {
  if (!currentResults) return;
  setActiveStyleButton('style-sphere-btn');
  applyViewerStyle('sphere', currentResults.chain_a, currentResults.chain_b, currentResults.residues);
});

document.getElementById('style-stick-btn').addEventListener('click', () => {
  if (!currentResults) return;
  setActiveStyleButton('style-stick-btn');
  applyViewerStyle('stick', currentResults.chain_a, currentResults.chain_b, currentResults.residues);
});

// Download B-factor PDB file
document.getElementById('download-bfactor-pdb-btn').addEventListener('click', async () => {
  if (!currentResults) return;
  
  let predsMap = {};
  currentResults.residues.forEach(r => {
    predsMap[r.resid.toString()] = r.probability;
  });
  
  const formData = new FormData();
  if (selectedFile) {
    formData.append('pdb_file', selectedFile);
  } else {
    formData.append('pdb_id', pdbId.value.trim().toUpperCase());
  }
  formData.append('chain_a', currentResults.chain_a);
  formData.append('predictions_json', JSON.stringify(predsMap));
  
  try {
    const response = await fetch(`${API_BASE}/download_pdb`, {
      method: 'POST',
      body: formData
    });
    if (!response.ok) throw new Error("Server failed to generate B-factor PDB.");
    
    const blob = await response.blob();
    const pid = pdbId.value.trim().toUpperCase() || 'predicted_bfactor';
    triggerDownload(blob, `ecabsd_bfactor_${pid}.pdb`);
  } catch (err) {
    showError(err.message || "Failed to download B-factor PDB.");
  }
});

// Docking Overlay controls
const dockLigandInput = document.getElementById('dock-ligand-input');
const runDockBtn = document.getElementById('run-dock-btn');
const dockStatus = document.getElementById('dock-status');

if (dockLigandInput) {
  dockLigandInput.addEventListener('change', () => {
    if (dockLigandInput.files && dockLigandInput.files.length > 0) {
      runDockBtn.disabled = false;
    } else {
      runDockBtn.disabled = true;
    }
  });
}

if (runDockBtn) {
  runDockBtn.addEventListener('click', async () => {
    if (!currentResults || !dockLigandInput.files || dockLigandInput.files.length === 0) return;
    
    runDockBtn.disabled = true;
    runDockBtn.textContent = 'Docking...';
    dockStatus.textContent = 'Aligning ligand to binding interface...';
    
    let predsMap = {};
    currentResults.residues.forEach(r => {
      predsMap[r.resid.toString()] = r.probability;
    });
    
    const formData = new FormData();
    if (selectedFile) {
      formData.append('pdb_file', selectedFile);
    } else {
      formData.append('pdb_id', pdbId.value.trim().toUpperCase());
    }
    formData.append('chain_a', currentResults.chain_a);
    formData.append('predictions_json', JSON.stringify(predsMap));
    formData.append('ligand_file', dockLigandInput.files[0]);
    
    try {
      const response = await fetch(`${API_BASE}/dock`, {
        method: 'POST',
        body: formData
      });
      
      const data = await response.json();
      if (data.status === 'success') {
        dockedLigandPdbString = data.docked_pdb;
        
        // Remove existing ligand model if present
        if (viewer3D.models && viewer3D.models[1]) {
          viewer3D.removeModel(viewer3D.models[1]);
        }
        
        // Load docked ligand model
        const ligandModel = viewer3D.addModel(dockedLigandPdbString, 'pdb');
        viewer3D.setStyle({ model: 1 }, { stick: { color: '#fbbf24', radius: 0.35 } });
        viewer3D.zoomTo();
        viewer3D.render();
        
        dockStatus.innerHTML = `✓ Docking overlay complete! affinity: <strong style="color:var(--yellow);">${data.affinity} kcal/mol</strong>`;
      } else {
        throw new Error(data.detail || "Docking alignment failed.");
      }
    } catch (err) {
      showError(err.message || "Docking simulation failed.");
      dockStatus.textContent = '';
    } finally {
      runDockBtn.disabled = false;
      runDockBtn.textContent = '⚡ Run Docking';
    }
  });
}
