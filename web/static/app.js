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

// PDB ID Input Listener
pdbId.addEventListener('input', () => {
  const val = pdbId.value.trim();
  if (val.length === 4) {
    selectedFile = null;
    dropzone.classList.remove('has-file');
    fileNameDisplay.textContent = `PDB ID: ${val.toUpperCase()}`;
    predictBtn.disabled = false;
  } else if (!selectedFile) {
    predictBtn.disabled = true;
    fileNameDisplay.textContent = 'No file selected';
  }
});

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
