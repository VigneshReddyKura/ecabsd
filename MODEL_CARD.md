# Model Card: ECABSD V3 (Explainable Cross-Attention for Binding Site Discovery)

This model card provides detailed technical specifications, performance metrics, and usage guidelines for the **ECABSD V3** model.

## Model Details

* **Developed by:** Anumala Manigreeva, D. Nayaneesh, Kantam Pavan Sai Reddy, Kura Vignesh Reddy, Vitta Karthikeya.
* **Mentored by:** Mr. Challa Sundeep Babu, Assistant Professor, Department of CSE, KMIT.
* **Model Type:** Graph Neural Network (GNN) with bidirectional cross-attention and Evolutionary Scale Modeling (ESM-2) features.
* **Architecture:** 6 GATv2 layers, 4 attention heads, global context pooling, and a 3-layer MLP prediction head.
* **Language:** PyTorch Geometric & Python.
* **License:** Open source (academic use).
* **Repository:** [https://github.com/amanigreeva/ECABSD](https://github.com/amanigreeva/ECABSD)

## Intended Use

* **Primary Use Case:** Predicts protein-protein interaction (PPI) binding site residues on a target protein structure (Chain A) given the structural context of a partner protein (Chain B).
* **Downstream Tasks:** Virtual screening, pocket identification, drug discovery, and biological validation prioritization.
* **Out-of-Scope Uses:** Predicting binding affinities directly without node classifications, or modeling fast conformational changes in real time (the GNN operates on rigid or static coordinates).

## Training Data

* **Source:** PDBbind (v2020) + DIPS subset (3,816 complexes total).
* **Data Augmentation:** Chain-swap augmentation ($p = 0.5$) applied during training, doubling training diversity.
* **Splits:** Strict komplex-level splitting using MMseqs2 clustering at $\le$30% sequence identity and $\ge$80% coverage to prevent homology-based data leakage.

## Evaluation Benchmarks & Metrics

The model is evaluated on the hold-out test set of the Docking Benchmark 5.5 (DB5.5).

### Quantitative Metrics

| Benchmark Split | F1-Score | MCC | ROC-AUC | PR-AUC |
| :--- | :---: | :---: | :---: | :---: |
| **Standard Random Split** | `0.7010` | `0.6452` | `0.9373` | `0.7462` |
| **Homology-Filtered ($\le$30%)** | `0.5797` | `0.5152` | `0.8928` | `0.6077` |
| **5-Fold Cross-Validation** | `0.4673` | `0.3898` | `0.8338` | `0.4595` |

### Conformation Generalization (Bound vs. Unbound)
* **Bound F1:** `0.8390`
* **Unbound F1:** `0.6871`
* **Conformational Degradation Margin:** `0.1519` (reflects high robustness compared to isolated-protein baselines).

### Statistical Significance (Wilcoxon Signed-Rank Test)
* Wilcoxon p-value: **`0.000`** ($p < 0.05$), demonstrating statistically significant improvements over non-interaction-aware models.

## Explainability and Biological Hotspots

* **Interpretability Paradigms:** Grad-CAM saliency maps and Attention Rollout.
* **Hotspot Alignment (RNase Sa/Barstar 1AY7):**
  - Pearson correlation between Grad-CAM saliency and physical distance to partner: **`-0.955`** ($p = 1.37 \times 10^{-51}$).
  - Pearson correlation between Grad-CAM saliency and solvent contact burial: **`0.727`** ($p = 5.18 \times 10^{-17}$).

## Limitations & Failure Cases

* **Low-Memory Fallback:** When free system RAM is <2.0 GB (e.g. Render free tier), the inference pipeline automatically swaps the ESM2-650M model for the ESM2-8M model and zero-pads the output. While this prevents server crashes, it may cause a slight degradation in fine-grained prediction accuracy compared to the native 650M GNN.
* **Large Complexes:** Proteins exceeding 800 residues are excluded by default to avoid memory saturation.
* **Rigid Coordinates:** Does not predict dynamic conformational changes that happen during complex binding.

## Ethical Considerations

* This model is intended for scientific research. It has no direct clinical application without secondary laboratory validation.
* Pretrained weights are hosted transparently on public repositories.
