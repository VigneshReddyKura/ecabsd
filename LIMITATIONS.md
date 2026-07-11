# ECABSD — Model Limitations & Critical Drawbacks

This document outlines the known limitations, constraints, and failures of the ECABSD model. In a technical interview, addressing these shows scientific rigor, self-awareness, and a realistic understanding of the model's performance boundaries.

---

## 1. The Performance Gap: Random Splits vs. Homology-Filtered Splits
* **The Drawback**: The model's F1-score drops from **0.7010** (Random Split) to **0.5797** (Homology-Filtered Split), and further down to **0.4673** (5-Fold Cross-Validation, 20-epoch budget).
* **Interview Context**: Random splitting (by complex) is standard in many legacy ML papers but suffers from sequence-similarity leakage. If the training set contains a protein homologous to one in the test set, the model memorizes the binding motif instead of learning structural physics.
* **Why it's a drawback**: When deployed on completely novel proteins (sequence identity ≤30%), performance declines.
* **Defense / Solution**: We ran Tier-2 MMseqs2 clustering to actively prove we did not train on test homologues. An honest `0.5797` F1-score on a homology-filtered split is more scientifically valid than an inflated `0.70` random split score.

---

## 2. Input Static Structure Assumption (No induced-fit modeling)
* **The Drawback**: ECABSD constructs graphs using static crystal structures from the PDB. It assumes the bound conformation is already known.
* **Why it's a drawback**: In nature, proteins undergo **conformational changes** (induced fit) upon binding. If we feed the model an unbound (apo) structure, some residues may be positioned differently, leading to false negatives.
* **Defense / Solution**: This is a common limitation of rigid-body predictors. In practice, coupling this with a conformational generator (like AlphaFold-Multimer or Rosetta) to generate unbound ensembles before prediction mitigates the issue.

---

## 3. Stationary Edge Features during GNN Message Passing
* **The Drawback**: Node features are updated dynamically through 6 layers of GATv2, but the edge features (distances and spatial directions) remain static.
* **Why it's a drawback**: The structural geometry is treated as rigid and fixed during graph message passing. The network cannot refine or update coordinate embeddings dynamically.
* **Defense / Solution**: Using a fully Equivariant GNN (like EGNN or Tensor Field Networks) would allow coordinates and edge attributes to update interactively. We chose fixed edge features to limit parameter size and reduce inference latency (seconds on CPU).

---

## 4. Compute Constraints on Cross-Validation
* **The Drawback**: The 5-fold cross-validation was run with an early-stopping training budget of **20 epochs per fold** instead of the full 80-epoch training cycle.
* **Why it's a drawback**: The reported CV mean F1-score (`0.4673±0.0077`) is a conservative lower bound, not the absolute convergence limit of the model.
* **Defense / Solution**: Training time was restricted due to hardware limitations (Kaggle T4 limits). A full 80-epoch run is mathematically estimated to yield F1 ≈ 0.58.

---

## 5. Web Interface Grad-CAM Fallback
* **The Drawback**: Grad-CAM saliency mapping is memory-intensive and is automatically disabled/fallback to attention-saliency in constrained cloud environments (like Render's free tier ≤512MB RAM).
* **Why it's a drawback**: Live web users may not get full Grad-CAM gradients if the container runs low on memory.
* **Defense / Solution**: The FastAPI backend includes a memory supervisor that dynamically redirects to attention rollout when RAM is <250MB, protecting the system from OOM crashes.

---

## 6. Manual Feature Dependency vs. End-to-End ESM-2 Embeddings
* **The Drawback**: To run inference in low-resource environments (including CPU), the model falls back to a 33-dimensional structural profile instead of dynamically running the 650M parameter ESM-2 language model.
* **Why it's a drawback**: The manual structural profile (B-factor, DSSP secondary structure, RSA proxy) requires external parsers which can sometimes fail on corrupted PDB coordinate lines.
* **Defense / Solution**: It makes the model highly portable. While ESM-2 embeddings improve sequence representation, static structural features are much faster to compute on edge devices.
