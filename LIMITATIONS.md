# ECABSD — Model Limitations & Critical Drawbacks

This document outlines the known limitations, failure modes, and scientific boundaries of the ECABSD model. Addressing these demonstrates scientific rigor and self-awareness.

---

## 1. Precision Ceiling & False Positive Dynamics
* **The Drawback**: The model prioritizes recall over precision (Precision = 0.5305 vs Recall = 0.6389 on homology splits). Because true binding interface residues constitute <15% of total protein surface area, false positives occur on peripheral surface residues located immediately adjacent to the core binding pocket.
* **Why it's a drawback**: In downstream virtual screening or experimental mutation design, false positives increase laboratory validation costs if uncalibrated.
* **Defense / Solution**: ECABSD functions as a high-recall prioritization filter. Using dynamic thresholding or Top-K precision ranking isolates high-confidence core interface residues.

---

## 2. Small and Discontinuous Interface Failure Modes
* **The Drawback**: ECABSD relies on global context pooling and GATv2 neighborhood aggregation over a 10.0 Å Cα–Cα cutoff.
* **Why it's a drawback**: For small interfaces (<8 contact residues) or highly discontinuous binding patches, global context pooling can dilute localized interface signals, leading to false negatives on isolated contact loops.
* **Defense / Solution**: Incorporating localized multi-scale graph pooling (e.g. SAGPool or DiffPool) for small interfacial regions.

---

## 3. Input Static Structure Assumption (No Induced-Fit Dynamics)
* **The Drawback**: Graphs are constructed from rigid PDB crystal coordinates without modeling induced-fit conformational changes during message passing.
* **Why it's a drawback**: On unbound (apo) protein structures, structural movement of side-chains or flexible loops leads to an F1 degradation of 0.1519 (Bound F1 = 0.8390 vs Unbound F1 = 0.6871 on DB5.5).
* **Defense / Solution**: Coupling ECABSD with structural ensemble generators (AlphaFold-Multimer / Rosetta) or upgrading to equivariant graph architectures (EGNN).

---

## 4. Dataset Scope & Generalization Boundaries
* **The Drawback**: Training was conducted on 3,816 protein-protein complexes from PDBbind and DIPS.
* **Why it's a drawback**: While sufficient for standard benchmarks, this represents a fraction of the full structural interactome. Generalization to rare protein families, non-standard amino acids, or large multi-protein assemblies (>4 chains) has not been comprehensively benchmarked.
* **Defense / Solution**: Expanding training datasets to large-scale interactomes like PINDER.

---

## 5. Cross-Validation Training Budget Constraints
* **The Drawback**: Reported 5-fold cross-validation metrics (F1 = 0.4673 ± 0.0077) reflect an early-stopped 20-epoch training budget per fold rather than full convergence.
* **Why it's a drawback**: Serves as a conservative lower bound rather than the fully converged cross-validation capacity of the model.
* **Defense / Solution**: Running complete 80-epoch cross-validation runs on dedicated GPU compute clusters.

---

## 6. Web Interface RAM Supervision & Saliency Fallback
* **The Drawback**: Grad-CAM saliency mapping is memory-intensive. In constrained cloud environments (Render free tier ≤512MB RAM), full gradient calculation is dynamically redirected to attention rollout when free RAM drops below 250MB.
* **Defense / Solution**: Ensures 100% web uptime without OOM container crashes.

---

## 7. ESM-2 650M Model Feature Extraction Latency
* **The Drawback**: ECABSD utilizes the 650M-parameter ESM-2 language model (`esm2_t33_650M_UR50D`, 1280-dimensional per-residue embeddings) for deep evolutionary sequence representation.
* **Why it's a drawback**: Extracting 1280-dim sequence embeddings introduces feature extraction latency (~1.2s per protein on GPU) and requires higher GPU VRAM during dataset generation and online prediction.
* **Defense / Solution**: An automated fallback mechanism dynamically redirects to `esm2_t6_8M_UR50D` (320-dim zero-padded to 1280-dim) in low-memory compute environments (<2.0 GB RAM), maintaining high uptime without container memory crashes.

