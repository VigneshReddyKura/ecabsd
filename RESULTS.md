# ECABSD — Results & Reproducibility

This document provides a transparent record of reported metrics, experimental
conditions, and planned validation steps for ECABSD.

---

## V3 Model — Current Benchmark Results

**Model version:** V3 (6-layer GATv2 + Global Context Pooling + Cross-Attention)  
**Evaluated:** May 2026  
**Checkpoint:** `checkpoints/best_model_v3.pt`  
**Dataset:** 3,816 protein–protein complexes (PDBbind + DIPS subset)  
**Split type:** Random train/val/test (70/15/15) by PDB complex ID  
**Optimal threshold:** Determined by PR-curve sweep on validation set  

| Metric | Score |
|---|---|
| **F1 Score** | `0.7010` |
| **ROC-AUC** | `0.9373` |
| **PR-AUC** | `0.7462` |
| **Recall** | `0.7756` |
| **Precision** | `0.6396` |
| **Accuracy** | `0.8989` |
| **MCC** | `0.6452` |

> **Limitation:** The above metrics are on a random split. Homology-filtered
> metrics (MMseqs2, ≤30% identity) are in progress — see below.

---

## Experimental Conditions

### Training Setup

| Parameter | Value |
|---|---|
| Loss | Focal (α=0.90, γ=2.0) + Soft-Dice (weight=0.40) |
| Optimizer | AdamW (lr=3e-4, wd=1e-4) |
| LR Schedule | Linear warmup (15 epochs) → CosineAnnealing |
| Early stopping | Patience=60 epochs on val F1 |
| Chain-swap augmentation | p=0.50 (doubles effective training data) |
| Batch size | 1 (graph-level, variable size) |
| Gradient clipping | 1.0 |

### Model Architecture

| Component | Details |
|---|---|
| Node features | 33-dim (ESM-2 `esm2_t6_8M_UR50D` + geometry) |
| Edge features | 5-dim (SE(3)-aware distance + direction) |
| GATv2 layers | 6 layers, 256 hidden dim, residual connections |
| Attention heads (cross) | 4 heads, 256 dim |
| Classifier | 3-layer MLP (LayerNorm → ReLU → Dropout → Sigmoid) |
| Graph cutoff | 10.0 Å (Cα–Cα intra-chain) |
| Labeling cutoff | 4.5 Å (interfacial atomic contact) |

---

## Reproducibility

To reproduce the reported metrics:

```bash
# 1. Set up environment
conda env create -f environment.yml
conda activate ecabsd

# 2. Verify checkpoint is present
ls checkpoints/best_model_v3.pt

# 3. Run evaluation
python main.py evaluate --checkpoint checkpoints/best_model_v3.pt

# 4. Check for data leakage (PDB-level)
python check_leakage.py

# 5. Verify splits file
python -c "import pandas as pd; df=pd.read_csv('data/splits.csv'); print(df['split'].value_counts())"
```

All random seeds are fixed via `set_seed(42)` in `train.py`.

---

## Planned Validation (In Progress)

The following validation steps are infrastructure-ready but require
GPU compute time to complete. Results will be added here upon completion.

### 1. Homology-Aware Splits (MMseqs2)

**Status:** Script ready at `scripts/generate_homology_splits.py`  
**Command:**
```bash
# Requires: conda install -c bioconda mmseqs2
python scripts/generate_homology_splits.py \
    --splits data/splits.csv \
    --pdb-dir data/raw/pdbs \
    --output data/splits_homology.csv \
    --identity 0.30
```

**Results (Kaggle GPU T4, May 2026):**

| Metric | Random Split | Homology-Filtered (≤30%) |
|---|---|---|
| F1 Score | 0.7010 | `0.5797` |
| ROC-AUC | 0.9373 | `0.8928` |
| PR-AUC | 0.7462 | `0.6077` |
| Recall | 0.7756 | `0.6389` |
| Precision | 0.6396 | `0.5305` |
| Accuracy | 0.8989 | `0.8828` |
| MCC | 0.6452 | `0.5152` |

### 2. 5-Fold Cross-Validation

**Status:** ✅ Completed (Kaggle GPU T4, May 2026)  
**Settings:** 20 epochs per fold, patience=7, seed=42  
**Command:**
```bash
python scripts/train_kfold.py \
    --config config.yaml \
    --splits data/splits_homology.csv \
    --folds 5 \
    --output results/kfold_results.json
```

**Results (5-Fold CV, homology-aware splits):**

| Metric | Mean | ±Std |
|---|---|---|
| **F1 Score** | `0.4673` | `0.0077` |
| **ROC-AUC** | `0.8338` | `0.0057` |
| **PR-AUC** | `0.4595` | `0.0162` |
| **Precision** | `0.4069` | `0.0153` |
| **Recall** | `0.5506` | `0.0251` |
| **Accuracy** | `0.8516` | `0.0092` |
| **MCC** | `0.3898` | `0.0065` |

> **Note:** K-fold models were trained with 20 epochs (vs 80 for single split)
> due to Kaggle GPU time constraints. Full 80-epoch K-fold is expected to
> yield scores closer to the single-split results (F1 ≈ 0.58).

### 3. Baseline Comparison

**Status:** Script ready at `scripts/benchmark_crossPPI.py`  
**Command:**
```bash
python scripts/benchmark_crossPPI.py \
    --checkpoint checkpoints/best_model_v3.pt
```

**Expected results table (to be filled after benchmark run):**

| Method | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|
| ECABSD V3 (ours) | 0.7010 | 0.9373 | 0.7462 |
| MaSIF-site | *pending* | *pending* | *pending* |
| CrossPPI | *pending* | *pending* | *pending* |
| ESMFold-IF | *pending* | *pending* | *pending* |

---

## Data Leakage Analysis

### PDB-level check (current)
No overlapping PDB IDs across train/val/test splits — verified by:

```
Train-Val overlap:  0 complexes
Train-Test overlap: 0 complexes
Val-Test overlap:   0 complexes
```

This check runs automatically at the start of every training run (`train.py:L367-370`).

### Sequence-level check (completed ✅)
MMseqs2 clustering at ≤30% sequence identity, ≥80% coverage applied.
Zero leakage confirmed across all train/val/test splits:
```
Train-Val overlap:  0 complexes
Train-Test overlap: 0 complexes
Val-Test overlap:   0 complexes
```

---

## Version History

| Version | Date | F1 | Notes |
|---|---|---|---|
| V2 (overboost) | 2026-04 | 0.6351 | 4-layer GCN, tuned threshold |
| V3 | 2026-05 | 0.7010 | 6-layer GATv2, focal+dice loss, chain-swap aug |
| V3-homology | 2026-05 | 0.5797 | V3 + MMseqs2-filtered splits (≤30% identity) |
| V3-kfold | 2026-05 | 0.4673±0.0077 | V3 + 5-fold CV (20 epochs), mean±std reported |
