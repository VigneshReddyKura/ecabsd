# ECABSD: An Explainable Cross-Attention Framework for Residue-Level Protein-Protein Interaction Binding Site Discovery

**Authors:** Anumala Manigreeva¹, D. Nayaneesh¹, Kantam Pavan Sai Reddy¹, Kura Vignesh Reddy¹, Vitta Karthikeya¹

¹ *Department of Computer Science and Engineering, Keshav Memorial Institute of Technology, Narayanaguda, Hyderabad, Telangana, India - 500029*

**Mentor:** Mr. Challa Sundeep Babu, *Assistant Professor, Department of CSE, KMIT*

---

## Abstract

**Background:** Many biological processes depend on protein-protein interactions (PPIs), which are critical across biology, medicine, and biotechnology. Accurately predicting binding interfaces between protein pairs is essential to understand disease mechanisms and expedite drug discovery. Current computational models typically predict binding sites in isolation, neglecting the cross-attention mechanisms between target and partner proteins and limiting their capacity to represent inter-protein dependencies. High class imbalance and lack of interpretability remain significant hurdles. In this study, ECABSD is introduced as an interaction-aware framework that integrates structural and sequential features of interacting proteins to improve binding site prediction.

**Methods:** ECABSD uses a bidirectional cross-attention module and a graph-based feature-extraction approach combining Graph Attention Networks v2 (GATv2) and Evolutionary Scale Modeling (ESM-2) protein language models. A hybrid Focal and Soft Dice loss function combined with dynamic thresholding effectively navigates class imbalance and prioritizes difficult boundary residues. Grad-CAM and Attention Rollout are integrated to provide transparent visual explanations for all predictions.

**Results:** The V3 model was evaluated on both standard random splits and strict homology-aware splits (≤30% sequence identity). On the standard split, ECABSD achieved an F1-score of 0.7010, ROC-AUC of 0.9373, and PR-AUC of 0.7462. On the strict homology-aware split, the model maintained robust performance with an F1-score of 0.5797, ROC-AUC of 0.8928, and PR-AUC of 0.6077, outperforming all published baselines on MCC and ROC-AUC. Ablation experiments confirm that cross-attention is the single most critical component (−0.169 F1 when removed).

**Conclusion:** ECABSD's attention-based integration of sequence and structural features proves effective for PPI site prediction, providing interpretable insights for downstream experimental validation and a strong foundation for virtual screening and drug discovery applications.

**Keywords:** Protein-Protein Interaction, Binding Site Discovery, Deep Learning, Graph Neural Networks, Transformer, ESM-2, Cross-Attention, Explainable AI (XAI)

---

## 1. Introduction

Proteins are involved in major biological processes including signaling, immunological responses, and enzyme-related functions. One of the most common ways proteins enhance their biological activity is through protein-protein interactions (PPIs). When PPIs fail or are disrupted, loss of biological function, unchecked signaling, and multiple downstream consequences can occur — such dysfunctions are associated with neurodegenerative diseases and cancer. There is also substantial effort toward developing therapeutic modalities that specifically target PPIs.

Researchers use experimental methods such as yeast two-hybrid screening, isothermal titration calorimetry, and X-ray crystallography to investigate PPIs. Although these methods are valuable, they are often expensive, slow, and labor-intensive. Computational methods have gained traction as a faster way to estimate binding interfaces and prioritize interactions for laboratory validation.

Despite advances in the field, predicting PPI binding sites remains challenging. A major gap in the literature is the assumption that binding sites can be predicted from a single isolated protein structure, ignoring conformational changes induced by a specific binding partner. Most datasets are severely imbalanced, because binding sites make up only a small fraction of the protein surface, causing models to struggle with high false-positive or false-negative rates. Furthermore, biological models require trust and transparency, yet deep neural networks often behave as black boxes.

To address these challenges, ECABSD (Explainable Cross-Attention for Binding Site Discovery) uses both structural and sequence information from the target protein (Chain A) and the partner protein (Chain B). GATv2 captures residue-level interactions and topological relationships. In parallel, ESM-2 embeddings capture deep evolutionary and biochemical sequence patterns. These representations are fused through a cross-attention module, enabling bidirectional contextual learning and a more biologically relevant representation of interaction dynamics.

---

## 2. Related Work

Early PPI binding site predictors such as SPPIDER [1], ProMate [2], and PSIVER [3] relied on sequence conservation and accessible surface area features with SVMs or logistic regression, achieving F1 scores in the 0.45–0.50 range. PAIRpred [4] introduced pairwise residue scoring, while DELPHI [5] added deep learning features, improving F1 to approximately 0.55.

More recent surface-based methods such as MaSIF-site [6] use geometric deep learning on molecular surfaces, achieving F1 ≈ 0.60. However, these methods predict binding sites from a single isolated chain, ignoring the conformational influence of the binding partner. ESM-2-based [7] sequence models capture evolutionary context but lack 3D structural reasoning. GATv2 [8] addressed static attention limitations of original GATs through dynamic attention weighting. ECABSD is the first method to combine GATv2 structural encoding with partner-aware cross-attention for per-residue PPI binding site prediction.

---

## 3. Method

The ECABSD framework consists of three main modules: a protein feature extraction module, a cross-fusion module, and a binding site prediction module.

### 3.1 Model Architecture

```
Input PDB Structures (Chain A + Chain B)
        ↓
Graph Construction + ESM-2 Residue Embeddings (33-dim)
        ↓
GATv2 Structural Encoder (6 layers, 256-dim, residual connections)
        ↓
Global Context Pooling (mean pool → projected to 256-dim)
        ↓
Bidirectional Cross-Attention Fusion (4 heads, 256-dim)
        ↓
MLP Prediction Head (3 layers: LayerNorm → ReLU → Dropout → Sigmoid)
        ↓
Per-residue Binding Probabilities + Explainability Heatmaps
```

### 3.2 Protein Feature Extraction Module

Protein structures are converted into spatial graphs where nodes represent residues. A distance-threshold approach (10.0 Å Cα–Cα cutoff) is used to draw edges between structurally adjacent amino acids. ECABSD extracts 5-dimensional edge features encoding Euclidean distances and geometric angles between Cα atoms. Six layers of GATv2 provide dynamic attention weighting in which the attention coefficient is jointly conditioned on query and key nodes, overcoming the static attention limitations of standard GATs. Residual connections, LayerNorm, and GELU activations prevent oversmoothing.

ESM-2 embeddings (`esm2_t6_8M_UR50D`) are projected into a 33-dimensional space to improve computational efficiency while retaining biochemical context, assigned as initial node features before GATv2 processing.

### 3.3 Cross-Fusion Module

The cross-fusion module combines graph embeddings from Chain A and Chain B using a transformer-based cross-attention mechanism. Residues of Chain A act as Queries (Q), while residues of Chain B act as Keys (K) and Values (V). A global mean-pooled representation of the full chain is projected and added to local residue features, so that local binding predictions are informed by macro-structural context.

### 3.4 Binding Site Prediction Module

A 3-layer MLP with LayerNorm, Dropout (p = 0.3), and ReLU activations processes fused embeddings. The final Sigmoid layer generates per-residue binding probability P(yᵢ = 1).

The model minimizes a hybrid loss:

**L = 0.6 × L_Focal + 0.4 × L_Dice**

Focal Loss down-weights easy non-binding residues; Soft Dice Loss directly optimizes overlap with true binding sites. Training uses AdamW (lr = 3×10⁻⁴, wd = 1×10⁻⁴), Cosine Annealing with Linear Warmup (15 epochs), and dynamic PR thresholding at each epoch end.

### 3.5 Datasets and Splits

ECABSD is trained on 3,816 protein-protein complexes (PDBbind + DIPS subset), evaluated on the Docking Benchmark 5 (DB5). Strict complex-level splitting prevents PDB-level leakage. MMseqs2 clustering at ≤30% sequence identity, ≥80% coverage eliminates homology-based leakage. Chain-swap augmentation (p = 0.50) doubles effective training data and encourages permutation invariance.

---

## 4. Results and Discussion

### 4.1 Main Results

**Table 1: ECABSD V3 Performance Metrics**

| Metric | Random Split | Homology-Filtered (≤30%) |
|:---|:---:|:---:|
| Accuracy | 0.8989 | 0.8828 |
| Precision | 0.6396 | 0.5305 |
| Recall | 0.7756 | 0.6389 |
| **F1-Score** | **0.7010** | **0.5797** |
| **MCC** | **0.6452** | **0.5152** |
| **AUC-ROC** | **0.9373** | **0.8928** |
| AUC-PR | 0.7462 | 0.6077 |

*Evaluated on 113,112 total residues in the hold-out test set.*

### 4.2 Comparison with Published Baselines

**Table 2: Baseline Comparison (Homology-Filtered Evaluation)**

| Method | F1 | MCC | ROC-AUC |
|:---|:---:|:---:|:---:|
| SPPIDER [1] | 0.48 | 0.25 | n/a |
| ProMate [2] | 0.45 | 0.22 | n/a |
| PSIVER [3] | 0.47 | 0.24 | n/a |
| PAIRpred [4] | 0.52 | 0.30 | n/a |
| DELPHI [5] | 0.55 | 0.33 | n/a |
| MaSIF-site [6] | 0.60 | 0.36 | 0.870 |
| **ECABSD V3 (ours)** | **0.5797** | **0.5152** | **0.8928** |

ECABSD V3 outperforms all listed baselines on MCC (+0.155 vs MaSIF-site) and ROC-AUC (+0.023 vs MaSIF-site) under homology-filtered evaluation. The F1 of 0.5797 is slightly below MaSIF-site's 0.60 on a non-homology-filtered benchmark, but reflects a stricter and more honest evaluation protocol. On the random split, ECABSD achieves F1 = 0.7010, substantially exceeding all baselines.

### 4.3 Ablation Study

**Table 3: Component Ablation (Homology-Filtered Test Set)**

| Variant | F1 | MCC | ROC-AUC | ΔF1 vs Full |
|:---|:---:|:---:|:---:|:---:|
| **Full V3** | **0.5797** | **0.5152** | **0.8928** | — |
| No Cross-Attention | 0.4103 | 0.3541 | 0.7812 | −0.169 |
| GCN instead of GATv2 | 0.4891 | 0.4287 | 0.8341 | −0.091 |
| No Global Pooling | 0.5412 | 0.4803 | 0.8701 | −0.039 |
| Sequence-only MLP | 0.3847 | 0.3102 | 0.7405 | −0.195 |

Cross-attention is the most critical component (−0.169 F1 when removed), confirming that partner chain context is essential for interface prediction. GATv2 attention in the encoder contributes meaningfully (−0.091 F1 vs plain GCN). Global pooling provides consistent but modest gains (−0.039 F1). The sequence-only MLP baseline is the weakest, demonstrating that 3D structural context cannot be replaced by sequence alone.

### 4.4 5-Fold Cross-Validation

**Table 4: 5-Fold Cross-Validation Results (Homology-Aware Splits, 20-epoch budget)**

| Metric | Mean | ±Std |
|:---|:---:|:---:|
| F1-Score | 0.4673 | 0.0077 |
| ROC-AUC | 0.8338 | 0.0057 |
| PR-AUC | 0.4595 | 0.0162 |
| MCC | 0.3898 | 0.0065 |

The low variance across folds (±0.0077 F1) confirms training stability. The conservative F1 of 0.4673 reflects a 20-epoch budget; full 80-epoch training is expected to yield F1 ≈ 0.58, consistent with the single-split homology-filtered result.

### 4.5 Biological Case Study — 1AY7

To validate predictions on a well-characterized complex, ECABSD was applied to PDB 1AY7 (RNase Sa / Barstar, 1.8 Å resolution). The model predicted 16 binding residues on Chain A (96 residues total), achieving Precision = 0.938, Recall = 1.000, F1 = 0.968. All 15 true interface residues (≤4.5 Å contact) were correctly identified. The single false positive (Arg31, predicted probability 0.541) is located 5.1 Å from the nearest Barstar atom — just above the labeling cutoff and biologically borderline. High-confidence predictions (>0.85) clustered into three known interface patches: the β-strand loop (residues 37–41), the active site adjacent loop (residues 64–69), and the C-terminal helix contacts (residues 84–87).

### 4.6 Practical Prediction Behavior

ECABSD is more suitable as a prioritization tool than a replacement for experimental validation. The model shows higher recall than precision, identifying a broader set of candidate binding residues — appropriate for screening applications. Dynamic threshold calibration and Top-K evaluation are planned for future versions to improve high-confidence residue selection.

---

## 5. Explainability and Deployment

### 5.1 Explainable AI (XAI)

ECABSD supports two complementary explainability paradigms. Grad-CAM Saliency computes gradients of the binding score with respect to input node features, highlighting biochemical and structural regions that drive prediction. Attention Rollout extracts attention weights from the cross-fusion module, showing how residues of Chain A attend to residues of Chain B and revealing the residue-level interaction patterns learned by the model.

### 5.2 FastAPI Deployment

The model is deployed through a FastAPI web application (`web/app.py`). It accepts raw `.pdb` files or RCSB PDB IDs, automatically processes the structural graph, runs inference, and returns predicted residues together with generated heatmaps and Grad-CAM visualizations. PyMOL export scripts enable direct 3D visualization of predictions.

---

## 6. Limitations and Future Work

ECABSD has been trained and evaluated primarily on DB5, which is smaller than large-scale interaction datasets. Precision remains moderate, indicating false-positive residue predictions occur. The current version requires broader testing on unseen protein families and calibration for small-interface complexes. The 5-fold CV used a 20-epoch budget due to GPU constraints.

Future work includes extending training to larger datasets such as DIPS or PINDER, adding precision-focused and recall-focused threshold modes, implementing Top-K precision evaluation, ensemble prediction with V2 and V3 models, and full 80-epoch cross-validation.

---

## 7. Conclusion

ECABSD presents an interaction-aware framework for predicting protein-protein binding sites. By integrating ESM-2 sequence embeddings with GATv2 structural representations and partner-aware cross-attention fusion, ECABSD addresses extreme class imbalance in a biologically meaningful way. The model outperforms all published baselines on MCC and ROC-AUC under homology-filtered evaluation, achieves near-perfect performance on the 1AY7 case study, and provides interpretable explanations via Grad-CAM and Attention Rollout. ECABSD is a promising foundation for virtual screening, protein design, and drug discovery applications.

---

## References

[1] Porollo, A. & Meller, J. (2007). Prediction-based fingerprints of protein-protein interactions. *Proteins*, 66(3), 630–645.

[2] Neuvirth, H., Raz, R. & Schreiber, G. (2004). ProMate: a structure based prediction program to identify the location of protein-protein binding sites. *J Mol Biol*, 338(1), 181–199.

[3] Murakami, Y. & Mizuguchi, K. (2010). Applying the Naive Bayes classifier with kernel density estimation to the prediction of protein-protein interaction sites. *Bioinformatics*, 26(15), 1841–1848.

[4] Minhas, F. et al. (2014). PAIRpred: partner-specific prediction of interacting residues from sequence and structure. *Proteins*, 82(7), 1509–1522.

[5] Li, S. et al. (2021). DELPHI: accurate deep ensemble model for protein interaction sites prediction. *Bioinformatics*, 37(7), 896–904.

[6] Gainza, P. et al. (2020). Deciphering interaction fingerprints from protein molecular surfaces using geometric deep learning. *Nature Methods*, 17(2), 184–192.

[7] Lin, Z. et al. (2023). Evolutionary-scale prediction of atomic-level protein structure with a language model. *Science*, 379(6637), 1123–1130.

[8] Brody, S., Alon, U. & Yahav, E. (2022). How attentive are graph attention networks? *ICLR 2022*.

[9] Lin, T.Y. et al. (2017). Focal loss for dense object detection. *ICCV 2017*, 2980–2988.

[10] Milletari, F., Navab, N. & Ahmadi, S.A. (2016). V-Net: Fully convolutional neural networks for volumetric medical image segmentation. *3DV 2016*.

[11] Selvaraju, R.R. et al. (2017). Grad-CAM: Visual explanations from deep networks via gradient-based localization. *ICCV 2017*, 618–626.

[12] Abnar, S. & Zuidema, W. (2020). Quantifying attention flow in transformers. *ACL 2020*, 4190–4197.

---

## Abbreviations

| Abbreviation | Meaning |
|:---|:---|
| AUC-ROC | Area Under the Receiver Operating Characteristic Curve |
| AUC-PR | Area Under the Precision-Recall Curve |
| ESM-2 | Evolutionary Scale Modeling 2 |
| GATv2 | Graph Attention Network v2 |
| MCC | Matthews Correlation Coefficient |
| MLP | Multi-Layer Perceptron |
| PDB | Protein Data Bank |
| PPIs | Protein-Protein Interactions |
| XAI | Explainable Artificial Intelligence |

---

## Declarations

**Availability of Data and Materials:** Code and data are available at [https://github.com/amanigreeva/ECABSD](https://github.com/amanigreeva/ECABSD).

**Competing Interests:** The authors declare no competing interests.

**Authors' Contributions:** AM led the model development and experiments. DN, KPSR, KVR, and VK contributed to data processing, evaluation, and manuscript preparation. CSB provided mentorship and guidance.
