# ECABSD — Kaggle Execution Guide

This guide explains exactly how to run the ECABSD training and validation
pipeline on Kaggle to produce publication-quality metrics.

---

## Prerequisites

1. A [Kaggle account](https://www.kaggle.com) (free)
2. GPU enabled: **Notebook Settings → Accelerator → GPU T4** (recommended for PyTorch 2.1+ compatibility)
3. Internet enabled: **Notebook Settings → Internet → On**
4. Optionally: a GitHub Personal Access Token (to push results back)

---

## Quick Start (3 steps)

### Step 1 — Upload the notebook to Kaggle

Go to [kaggle.com/code](https://www.kaggle.com/code) → **New Notebook** →  
Upload `docs/kaggle_ecabsd_pipeline.ipynb` from this repo.

### Step 2 — Enable GPU & Internet

In Notebook Settings (right sidebar):
- **Accelerator**: GPU T4 (or P100 if PyTorch version is downgraded, but T4 is highly recommended) ← required
- **Internet**: On ← required for pip install and git clone

### Step 3 — Run All Cells

Click **Run All** and wait. Full pipeline takes ~6–8 hours.

---

## What Each Cell Does

| Cell | Task | Time |
|---|---|---|
| 1 | GPU check | ~5s |
| 2 | Install PyTorch Geometric + deps | ~5 min |
| 3 | Install MMseqs2 | ~2 min |
| 4 | Clone ECABSD from GitHub | ~30s |
| 5 | Download DB5 benchmark PDBs | ~10–30 min |
| 6 | Build residue graphs (PDB → .pt) | ~30–60 min |
| 7 | Generate homology-aware splits | ~5–15 min |
| 8 | Update config.yaml for Kaggle | ~5s |
| 9 | **Train V3 model** (main step) | ~3–5 hours |
| 10 | Evaluate best checkpoint | ~5 min |
| 11 | **5-fold cross-validation** | ~4–6 hours |
| 12 | Print final results summary | ~5s |
| 13 | Save outputs for download | ~30s |
| 14 | Push results to GitHub (optional) | ~30s |

> **Important**: Cells 9 and 11 both train the model. If you are short on time,
> run Cell 9 (single training run) and skip Cell 11 (k-fold). Single-run
> metrics on homology-filtered splits are still significantly stronger than
> the current random-split numbers.

---

## Optional: Push Results to GitHub

In Cell 4, paste your GitHub Personal Access Token:

```python
GITHUB_TOKEN = 'ghp_yourtoken...'
```

This enables Cell 14 to:
- Copy the results JSON into `results/`
- Append real metrics to `RESULTS.md`
- Commit and push automatically

To generate a token: [GitHub → Settings → Developer settings → PAT → Classic → Generate](https://github.com/settings/tokens)  
Required scopes: `repo` (read/write)

---

## Expected Output Metrics

After the full pipeline, Cell 12 prints a table like:

```
📊 Test Set Results (Homology-Aware Split):
  F1-Score:  0.67xx      ← may be lower than 0.7010 (stricter split)
  ROC-AUC:   0.91xx
  PR-AUC:    0.72xx
  MCC:       0.61xx

📊 5-Fold Cross-Validation Results:
  Metric          Mean     ±Std
  ---------------------------------
  f1            0.6xxx   0.0xxx
  auc_roc       0.9xxx   0.0xxx
```

> The homology-filtered metrics may be slightly lower than the current
> random-split results (0.7010 F1) — this is expected and scientifically
> correct. Lower but honest metrics are better for publication than
> inflated random-split metrics.

---

## Downloading Results

After the run completes:

1. Go to the **Output** tab in Kaggle
2. Download `/kaggle/working/output/`:
   - `best_model_v3.pt` — trained checkpoint
   - `test_metrics.json` — test set metrics
   - `kfold_results.json` — k-fold CV metrics

3. Copy `best_model_v3.pt` to your local `checkpoints/` folder
4. Update `RESULTS.md` with the new numbers
5. Push: `git add -A && git commit -m "results: add kaggle validation metrics" && git push`

---

## Troubleshooting

| Problem | Solution |
|---|---|
| OOM (out of memory) | Reduce `batch_size` in Cell 8 to 1, reduce `hidden_dim` to 128 |
| MMseqs2 install fails | Skip Cell 3 — Cell 7 falls back to random splits automatically |
| Download fails | Check Internet is enabled in Notebook Settings |
| 9-hour timeout | Run Cells 1–10 first (single training), skip k-fold Cell 11 |
| Push fails | Check GitHub token has `repo` scope |
