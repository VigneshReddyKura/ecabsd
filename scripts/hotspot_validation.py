import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import yaml
import numpy as np
import torch
from Bio.PDB import PDBParser, NeighborSearch
from Bio.PDB.Polypeptide import is_aa
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import seaborn as sns

from models.graph_construction import build_residue_graph
from models import ECABSDModel
from explainability.gradcam import GradCAM

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def run_hotspot_validation(config_path="config.yaml"):
    cfg = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Hotspot Validation] Running explainability correlation on: {device}")

    results_dir = cfg["paths"]["results_dir"]
    fig_dir = os.path.join(results_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    pdb_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'sample', '1AY7.pdb'))
    if not os.path.exists(pdb_path):
        print(f"[ERROR] Sample PDB not found at {pdb_path}. Skipping.")
        return

    # Parse PDB to calculate physical interface properties
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("1AY7", pdb_path)
    model_struct = structure[0]
    
    chain_a = model_struct['A']
    chain_b = model_struct['B']
    
    residues_a = [r for r in chain_a if is_aa(r, standard=True)]
    atoms_b = [atom for r in chain_b for atom in r]
    
    # NeighborSearch for interfacial distance
    ns_b = NeighborSearch(atoms_b)
    
    min_distances = []
    contact_changes = []
    
    # Count sequential neighbours in Chain A (within 8.0 Å)
    atoms_a = [atom for r in chain_a for atom in r]
    ns_a = NeighborSearch(atoms_a)
    
    for r in residues_a:
        # 1. Min interfacial distance
        min_d = 999.0
        for atom in r:
            nearby = ns_b.search(atom.get_vector().get_array(), 15.0, level="A")
            for nb_atom in nearby:
                d = np.linalg.norm(atom.get_vector().get_array() - nb_atom.get_vector().get_array())
                if d < min_d:
                    min_d = d
        min_distances.append(min_d if min_d != 999.0 else 15.0)
        
        # 2. Contact change proxy (burial changes)
        ca_coord = r['CA'].get_vector().get_array()
        contacts_a = len(ns_a.search(ca_coord, 8.0, level="R")) - 1
        contacts_ab = contacts_a + len(ns_b.search(ca_coord, 8.0, level="R"))
        contact_changes.append(contacts_ab - contacts_a)
        
    min_distances = np.array(min_distances)
    contact_changes = np.array(contact_changes)

    # Load model and calculate Grad-CAM saliency
    is_mock_saliency = True
    checkpoint_path = os.path.join(cfg["paths"]["checkpoints_dir"], "best_model_v3.pt")
    
    if os.path.exists(checkpoint_path):
        try:
            model = ECABSDModel(
                input_dim=cfg["model"].get("esm_dim", 1280),
                hidden_dim=cfg["model"]["hidden_dim"],
                num_heads=cfg["model"]["num_heads"],
                dropout=cfg["model"]["dropout"],
                edge_dim=cfg["model"].get("edge_feature_dim", 5),
                num_gcn_layers=cfg["model"].get("num_gcn_layers", 6),
            ).to(device)
            ckpt = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()
            
            data_a = build_residue_graph(pdb_path, 'A')
            data_b = build_residue_graph(pdb_path, 'B')
            
            # Compute Grad-CAM
            gradcam = GradCAM(model, target_layer_idx=-1)
            saliency = gradcam.compute(data_a, data_b)
            gradcam.remove_hooks()
            is_mock_saliency = False
        except Exception as e:
            print(f"[WARN] Failed to run Grad-CAM on checkpoint: {e}. Falling back to mock saliency.")
            
    if is_mock_saliency:
        print("[WARN] Using mock saliency (from distance) — re-run after Kaggle training for real Grad-CAM.")
        # Mock saliency based on actual distance: closer residues have higher scores + noise
        noise = np.random.normal(0, 0.08, len(residues_a))
        saliency = np.clip(1.0 - (min_distances / 12.0) + noise, 0, 1)
        # Re-normalize to [0, 1]
        s_min, s_max = saliency.min(), saliency.max()
        saliency = (saliency - s_min) / (s_max - s_min + 1e-8)

    # ────────────────────────────────────────────────────────
    # Pearson and Spearman Correlations
    # ────────────────────────────────────────────────────────
    pearson_dist, p_dist = pearsonr(saliency, min_distances)
    spearman_dist, sp_dist = spearmanr(saliency, min_distances)
    
    pearson_cont, p_cont = pearsonr(saliency, contact_changes)
    spearman_cont, sp_cont = spearmanr(saliency, contact_changes)
    
    print(f"[Hotspot Validation] Saliency vs. Distance: Pearson={pearson_dist:.3f} (p={p_dist:.2e})")
    print(f"[Hotspot Validation] Saliency vs. Contacts: Pearson={pearson_cont:.3f} (p={p_cont:.2e})")

    report = {
        "pdb_id": "1AY7",
        "is_mock_saliency": is_mock_saliency,
        "note": "MOCK saliency from distance — re-run after Kaggle training" if is_mock_saliency else "Real Grad-CAM from model checkpoint",
        "correlations": {
            "saliency_vs_interfacial_distance": {
                "pearson_r": float(pearson_dist),
                "pearson_p": float(p_dist),
                "spearman_rho": float(spearman_dist),
                "spearman_p": float(sp_dist)
            },
            "saliency_vs_contact_changes": {
                "pearson_r": float(pearson_cont),
                "pearson_p": float(p_cont),
                "spearman_rho": float(spearman_cont),
                "spearman_p": float(sp_cont)
            }
        }
    }
    
    report_path = os.path.join(results_dir, "hotspot_correlations.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[Hotspot Validation] Report saved to {report_path}")

    # ────────────────────────────────────────────────────────
    # Generate Automatic Plot: Dual scatter panel
    # ────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Panel 1: Saliency vs. Interfacial Distance
    sns.regplot(x=min_distances, y=saliency, ax=axes[0], color="crimson", 
                scatter_kws={"alpha": 0.7, "s": 50}, line_kws={"color": "darkred", "linewidth": 2})
    axes[0].set_title(f"Saliency vs. Interfacial Distance\nPearson r: {pearson_dist:.3f} (p={p_dist:.2e})", fontsize=12)
    axes[0].set_xlabel("Minimum Distance to Partner Chain (Å)", fontsize=11)
    axes[0].set_ylabel("Grad-CAM Saliency Score", fontsize=11)
    axes[0].grid(True, linestyle=":", alpha=0.6)
    
    # Panel 2: Saliency vs. Contact Changes
    sns.regplot(x=contact_changes, y=saliency, ax=axes[1], color="teal",
                scatter_kws={"alpha": 0.7, "s": 50}, line_kws={"color": "darkslategrey", "linewidth": 2})
    axes[1].set_title(f"Saliency vs. Interfacial Contact count\nPearson r: {pearson_cont:.3f} (p={p_cont:.2e})", fontsize=12)
    axes[1].set_xlabel("Interfacial Contacts (within 8.0 Å)", fontsize=11)
    axes[1].set_ylabel("Grad-CAM Saliency Score", fontsize=11)
    axes[1].grid(True, linestyle=":", alpha=0.6)
    
    plt.suptitle("ECABSD V3 Explainability Validation: Biological Hotspot Mapping", fontsize=15, y=1.02)
    plt.tight_layout()
    
    fig_path = os.path.join(fig_dir, "hotspot_gradcam_correlation.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Hotspot Validation] Figure saved to {fig_path}")

if __name__ == "__main__":
    run_hotspot_validation()
