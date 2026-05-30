# Biological Case Study — 1AY7 (RNase Sa / Barstar)

## Structure Overview

**PDB:** [1AY7](https://www.rcsb.org/structure/1AY7)  
**Complex:** Guanyl-specific Ribonuclease Sa (Chain A, 96 residues) bound to Barstar inhibitor (Chain B, 89 residues)  
**Resolution:** 1.8 Å X-ray crystallography  
**Organism:** *Streptomyces aureofaciens* (RNase Sa) / *Bacillus amyloliquefaciens* (Barstar)

## ECABSD V3 Prediction Results

| Metric | Value |
|---|---|
| True interface residues (≤4.5 Å) | 15 |
| Predicted binding residues | 16 |
| True Positives | 15 |
| False Positives | 1 (Arg31 — adjacent to interface) |
| False Negatives | 0 |
| **Precision** | **0.938** |
| **Recall** | **1.000** |
| **F1 Score** | **0.968** |

## Predicted Binding Residues (Chain A)

| Resid | Residue | Predicted Prob | True Interface |
|---|---|---|---|
| 32 | GLU | 0.8805 | ✅ |
| 37 | SER | 0.9025 | ✅ |
| 38 | GLU | 0.8561 | ✅ |
| 39 | ASN | 0.9096 | ✅ |
| 40 | GLY | 0.8470 | ✅ |
| 41 | LYS | 0.8750 | ✅ |
| 64 | THR | 0.8706 | ✅ |
| 65 | HIS | 0.9054 | ✅ |
| 66 | TYR | 0.8907 | ✅ |
| 67 | LYS | 0.7704 | ✅ |
| 69 | TRP | 0.7894 | ✅ |
| 84 | PRO | 0.9175 | ✅ |
| 85 | ARG | 0.8079 | ✅ |
| 86 | GLN | 0.7683 | ✅ |
| 87 | LEU | 0.9395 | ✅ |
| 31 | ARG | 0.5414 | ❌ (FP — adjacent residue) |

## Visualization

To reproduce the PyMOL figure:

```bash
# Copy 1AY7.pdb to your working directory, then:
pymol exports/ecabsd_1AY7_visualization.pml
```

This generates `ecabsd_1AY7_binding_site.png` — the binding site colored by predicted probability (white = low, red = high).

## Biological Interpretation

ECABSD correctly identifies the two main interface patches on RNase Sa:

- **Patch 1 (residues 37–41):** The β-strand loop region — a key electrostatic contact zone with Barstar Asp39
- **Patch 2 (residues 64–69):** The active site adjacent loop — hydrophobic core of the interface
- **Patch 3 (residues 84–87):** C-terminal helix contacts — contributing hydrogen bonds to Barstar

The single false positive (Arg31) is immediately adjacent to the interface and is a biologically reasonable borderline prediction; its distance to the nearest Barstar atom is ~5.1 Å (just above the 4.5 Å labeling cutoff).
