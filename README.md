# ECABSD — Explainable Cross Attention Model for Binding Site Discovery

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1-orange?logo=pytorch)
![PyG](https://img.shields.io/badge/PyG-2.7-red)
![License](https://img.shields.io/badge/License-MIT-green)

**Deep learning model for per-residue protein–protein binding site discovery using graph neural networks and explainable cross-attention.**

</div>

---

## Table of Contents
- [Overview](#overview)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [CLI Usage](#cli-usage)
- [Web Interface](#web-interface)
- [Training](#training)
- [Evaluation](#evaluation)
- [Explainability](#explainability)
- [Docking Integration](#docking-integration)
- [Exports](#exports)
- [Project Structure](#project-structure)

---

## Overview

ECABSD predicts which residues in a protein chain form the binding interface with another protein. It uses the new V2+ Overboost architecture:

1. **Graph Construction** — each protein chain becomes a residue graph with distance cutoff edges
2. **GCN Encoder** — 4-layer GATv2 stack (33 → 192 hidden dimensions)
3. **SE(3) Refinement** — equivariant feature refinement block
4. **Cross-Attention** — Multi-head attention between two protein chains with Global Context Pooling
5. **Per-residue Classifier** — Deep MLP with sigmoid for binding probability

---

## Architecture

```
Protein A  ─→ [Graph Construction] ─→ [GCN × 4] ─→ [SE3 Refine] ─┐
                                                                     ├─→ CrossAttention (8 heads) ─→ Classifier ─→ P(binding) per residue
Protein B  ─→ [Graph Construction] ─→ [GCN × 4] ─→ [SE3 Refine] ─┘
```

**Node features (33-dim):** ESM-2 Language Model embeddings + geometric features
**Edge features (5-dim):** SE(3)-aware distance and direction vectors
**Graph cutoff:** Configurable (default 10.0 Å)

---

## Performance Benchmark

The V2+ Overboost architecture achieves state-of-the-art predictive performance on the test set:

| Metric | Score |
|---|---|
| **F1 Score** | `0.6351` |
| **ROC-AUC** | `0.8977` |
| **PR-AUC** | `0.6722` |
| **Recall** | `0.6878` |
| **Precision** | `0.5899` |
| **Accuracy** | `0.8682` |
| **MCC** | `0.5577` |

---

## Installation

```bash
# Clone repository
git clone https://github.com/amanigreeva/ECABSD.git
cd ecabsd

# Create environment
conda create -n ecabsd python=3.10 -y
conda activate ecabsd

# Install PyTorch (CPU)
pip install torch==2.1.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install PyTorch Geometric
pip install torch-geometric==2.7.0
pip install torch-scatter torch-sparse torch-cluster --find-links https://data.pyg.org/whl/torch-2.1.0+cpu.html

# Install remaining dependencies
pip install biopython pydssp fastapi uvicorn typer pyyaml scikit-learn tqdm matplotlib seaborn
```

---

## Quick Start

### 1. Predict binding sites on 1AY7.pdb

```bash
python predict.py --pdb 1AY7.pdb --chain-a A --chain-b B
```

### 2. Run tests

```bash
pytest tests/
```

### 3. Launch web interface

```bash
cd web && python app.py
# → Open http://localhost:8000
```

---

## CLI Usage

```
python main.py --help

Commands:
  train          Train the ECABSD model
  evaluate       Evaluate on test set
  predict        Predict binding sites for a single PDB
  batch-predict  Batch predict for a directory of PDBs
  export         Export results to CSV / JSON / PyMOL
  web            Launch the web interface
```

### Examples

```bash
# Train (needs processed data)
python main.py train --config config.yaml

# Single prediction
python main.py predict --pdb 1AY7.pdb --chain-a A --chain-b B --threshold 0.5

# Batch prediction
python main.py batch-predict --input-dir data/raw/pdbs --output-dir results/batch

# Export to PyMOL script
python main.py export --results results/predictions_1AY7_A.json --format pymol
```

---

## Web Interface

The deployed web application uses stateless in-memory prediction and visualization export. No prediction artifacts are permanently stored on the server; results are returned directly to the browser and downloaded client-side.

```bash
# From project root
python web/app.py
```

Opens at **http://localhost:8000**. Features:
- Drag-and-drop PDB upload
- Chain selection + probability threshold slider
- Interactive probability chart (Chart.js)
- Per-residue results table with filter
- One-click export: CSV, JSON, PyMOL script

---

## Training

### Step 1: Download PDB structures

```bash
python scripts/download_pdbbind.py --benchmark
```

### Step 2: Prepare dataset

```bash
python scripts/prepare_dataset.py \
    --pdb-dir data/raw/pdbs \
    --output-dir data/processed \
    --cutoff 4.5
```

### Step 3: Train

```bash
python train.py
# or
python main.py train
```

Training config is in `config.yaml`. Key parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `hidden_dim` | 128 | Model hidden dimension |
| `num_heads` | 8 | Cross-attention heads |
| `graph_cutoff` | 8.0 Å | Edge distance cutoff |
| `epochs` | 100 | Max training epochs |
| `learning_rate` | 0.001 | Adam LR |
| `pos_weight` | 5.0 | BCE class weight for binding sites |
| `early_stopping_patience` | 15 | Epochs to wait before stopping |

Checkpoints saved to `checkpoints/`, logs to `logs/training_history.json`.

### Future V3 Model Retraining Protocol

To deliver next-generation improvements in predictive precision and recall, the future V3 training protocol will follow a strict, scientifically rigorous roadmap to resolve low-confidence outliers:

1. **Remove Train/Test Leakage**: Eliminate homology-based and sequence-similarity overlap between train, validation, and test partitions (using MMseqs2 at $30\%$ sequence identity cutoff).
2. **Clean Complexes**: Leverage high-resolution, curated Docking Benchmark 5 (DB5) and Binding Benchmark 5 (BM5) complexes to ensure accurate interfacial physical contacts.
3. **Split by PDB ID**: Restructure cross-validation folds strictly by PDB ID / protein family clusters to prevent intra-cluster leakage.
4. **Balance Residue Classes**: Apply advanced oversampling, focal loss, or dynamically weighted loss functions to handle the heavy imbalance between positive (binding) and negative (non-binding) residues.
5. **Optimize Decision Threshold**: Rather than static global boundaries, save the absolute best mathematical threshold optimized per validation fold to maximize the validation F1 score.
6. **Strict Unseen Validation**: Evaluate and validate exclusively on unseen PDB structures during active epoch runs.

### Scientific Probability Interpretation & Low-Confidence Flags

ECABSD produces per-residue binding probabilities derived from structural and language-model sequence embeddings. In scientific paper publications, it is critical not to claim "perfect prediction" or force binary predictions on ambiguous structures. Instead:
* **Confidence Categorization**: Samples are classified using the maximum residue probability (`max_prob`) to reflect the model's confidence in its predictions.
* **Low-Confidence Flags**: Outlier samples (such as PDB 1BRS) are explicitly flagged as `"Low-confidence / Needs Review"` rather than forced into positive/negative predictions.
* **Review Protocol**: Users are advised that these samples are valid biological structures, but the model assigned very low probabilities and they should be validated experimentally or tested with the advanced V3 model.

---

## Evaluation

```bash
python main.py evaluate --checkpoint checkpoints/best_model.pt
```

Outputs:
- `results/metrics.json` — Accuracy, Precision, Recall, F1, MCC, AUC-ROC, AUC-PR
- `results/confusion_matrix.png` — Confusion matrix plot

### Benchmark vs. Baselines

```bash
python scripts/benchmark_crossPPI.py --checkpoint checkpoints/best_model.pt
```

---

## Explainability

```python
from models.ecabsd_model import ECABSDModel
from models.graph_construction import build_residue_graph
from explainability.attention_rollout import explain_prediction
from explainability.gradcam import explain_with_gradcam

model = ECABSDModel()
data_a = build_residue_graph("1AY7.pdb", "A")

# Attention rollout
scores, attn_matrix = explain_prediction(model, data_a, output_dir="results/")

# Grad-CAM
saliency = explain_with_gradcam(model, data_a, output_dir="results/")
```

---

## Docking Integration

Requires AutoDock Vina: `conda install -c conda-forge autodock-vina`

```python
from predict import run_prediction
from docking.docking_input import binding_residues_to_box, write_vina_config
from docking.vina_runner import VinaRunner

# Get predictions
results = run_prediction("1AY7.pdb", "A", "B")
binding_residues = [r for r in results["residues"] if r["is_binding"]]

# Compute docking box
center, box_size = binding_residues_to_box(binding_residues, "1AY7.pdb", "A")

# Run docking
runner = VinaRunner(exhaustiveness=8)
result = runner.dock("receptor.pdbqt", "ligand.pdbqt", center, box_size)
```

---

## Exports

```bash
# CSV
python main.py export --results results/predictions_1AY7_A.json --format csv

# JSON (with metadata + confidence bands)
python main.py export --results results/predictions_1AY7_A.json --format json

# PyMOL script (probability-gradient coloring)
python main.py export --results results/predictions_1AY7_A.json --format pymol
```

---

## Project Structure

```
ecabsd/
├── 1AY7.pdb                    # Sample PDB structure
├── config.yaml                 # Central configuration
├── main.py                     # Entry point
├── cli.py                      # Typer CLI
├── train.py                    # Training pipeline
├── evaluate.py                 # Evaluation pipeline
├── predict.py                  # Single-structure prediction
├── batch_predict.py            # Batch prediction
│
├── models/
│   ├── __init__.py
│   ├── ecabsd_model.py         # End-to-end model
│   ├── encoder.py              # GCN + SE3 chain encoder
│   ├── gcn_model.py            # 4-layer GCNConv encoder
│   ├── se3_model.py            # SE(3) refinement block
│   ├── cross_attention.py      # Multi-head cross-attention
│   ├── classifier.py           # Per-residue MLP classifier
│   └── graph_construction.py  # PDB → residue graph
│
├── data/
│   ├── __init__.py
│   ├── dataset.py              # PyG Dataset
│   ├── raw/                    # Raw PDB files
│   └── processed/              # Preprocessed .pt graphs
│
├── scripts/
│   ├── prepare_dataset.py      # PDB → labeled graphs
│   ├── download_pdbbind.py     # Download PDB structures
│   └── benchmark_crossPPI.py  # Benchmark comparison
│
├── explainability/
│   ├── __init__.py
│   ├── attention_rollout.py    # Attention-based explainability
│   └── gradcam.py              # Grad-CAM for GNNs
│
├── docking/
│   ├── __init__.py
│   ├── vina_runner.py          # AutoDock Vina wrapper
│   ├── docking_input.py        # Box definition + PDBQT prep
│   └── rmsd.py                 # Docking pose RMSD
│
├── exports/
│   ├── __init__.py
│   ├── csv_export.py           # CSV export
│   ├── json_export.py          # JSON export with metadata
│   └── pymol_export.py         # PyMOL .pml script
│
├── web/
│   ├── app.py                  # FastAPI backend
│   ├── templates/index.html    # Web UI
│   └── static/
│       ├── style.css           # Dark-mode CSS
│       └── app.js              # Frontend JavaScript
│
├── notebooks/
│   └── quickstart_1AY7.ipynb  # Quickstart Jupyter notebook
│
├── tests/
│   └── test_graph_construction.py
│
├── checkpoints/                # Saved model weights
├── logs/                       # Training logs
├── results/                    # Prediction outputs
└── requirements.txt
```

## Known Limitations

- **Render Free Tier**: Render free tier may disable Grad-CAM for large proteins due to RAM limits (512MB).
- **Leakage Check**: Exact-match leakage check is included by default. However, MMseqs2 clustering is highly recommended for stronger sequence-similarity leakage prevention.

---

## License

MIT License — see [LICENSE](LICENSE) for details.
