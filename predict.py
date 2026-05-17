"""
ECABSD Single-Structure Prediction.

Predicts binding site residues for a single PDB file.
Outputs per-residue probabilities and highlighted binding residues.
"""

import os
import json
import yaml
import torch
import numpy as np

from models.ecabsd_model import ECABSDModel
from models.ecabsd_v3_model import ECABSDModelV3
from models.graph_construction import build_residue_graph, get_residues
from Bio.PDB import PDBParser


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def run_prediction(
    pdb_path: str,
    chain_a: str,
    chain_b: str = None,
    checkpoint_path: str = None,
    threshold: float = None,
    output_path: str = None,
    config_path: str = "config.yaml",
    model_version: str = "v3",
):
    """
    Predict binding sites for a single PDB structure.

    Parameters
    ----------
    pdb_path : str
        Path to the PDB file.
    chain_a : str
        Chain ID of the target protein.
    chain_b : str, optional
        Chain ID of the partner protein (for cross-attention).
    checkpoint_path : str
        Path to model checkpoint.
    threshold : float
        Probability threshold for binding site classification.
    output_path : str, optional
        Path to save results JSON. If None, saves to results/ dir.
    config_path : str
        Path to config file.
    """
    cfg = load_config(config_path)
    mcfg = cfg["model"]
    results_dir = cfg["paths"]["results_dir"]
    os.makedirs(results_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if checkpoint_path is None:
        checkpoint_path = "checkpoints/best_model_v3.pt" if model_version == "v3" else "checkpoints/best_model.pt"

    # Load model
    if model_version == "v3":
        model = ECABSDModelV3(
            input_dim=mcfg.get("input_dim", mcfg.get("esm_dim", 33)),
            hidden_dim=mcfg["hidden_dim"],
            num_heads=mcfg["num_heads"],
            dropout=0.0,
            edge_dim=mcfg.get("edge_feature_dim", 5),
            num_gcn_layers=mcfg.get("num_gcn_layers", 6)
        ).to(device)
    else:
        model = ECABSDModel(
            esm_dim=mcfg.get("esm_dim", 1280),
            hidden_dim=mcfg["hidden_dim"],
            num_heads=mcfg["num_heads"],
            dropout=0.0,
            num_layers=mcfg.get("num_gcn_layers", 3),
            cross_attention=True,
        ).to(device)

    # Resolve threshold: CLI arg > checkpoint value > config value
    cfg_threshold = cfg["prediction"].get("threshold", 0.5)
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        ckpt_threshold = checkpoint.get("best_threshold", cfg_threshold)
        print(f"[ECABSD] Loaded model from: {checkpoint_path}")
        print(f"[ECABSD] Checkpoint threshold: {ckpt_threshold:.4f}")
    else:
        print(f"[ECABSD] WARNING: No checkpoint at {checkpoint_path}. Using random weights.")
        ckpt_threshold = cfg_threshold

    is_auto = False
    if threshold is None:
        is_auto = True
        threshold = ckpt_threshold
        print(f"[ECABSD] Using threshold: {threshold:.4f}")

    # Automatic PDB downloading if the file does not exist locally
    if not os.path.exists(pdb_path):
        pdb_base = os.path.basename(pdb_path)
        pdb_id, ext = os.path.splitext(pdb_base)
        # If it looks like a PDB code (e.g. "1BRS" or "1BRS.pdb")
        if len(pdb_id) == 4 and (not ext or ext.lower() == '.pdb'):
            pdb_id = pdb_id.upper()
            os.makedirs("data/raw/pdbs", exist_ok=True)
            download_path = f"data/raw/pdbs/{pdb_id}.pdb"
            if not os.path.exists(download_path):
                print(f"[ECABSD] PDB file not found. Attempting to download {pdb_id} from RCSB PDB...")
                import urllib.request
                try:
                    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
                    urllib.request.urlretrieve(url, download_path)
                    print(f"[ECABSD] Successfully downloaded {pdb_id}.pdb to {download_path}")
                except Exception as e:
                    raise FileNotFoundError(f"Could not download PDB {pdb_id} from RCSB: {e}")
            pdb_path = download_path
        else:
            raise FileNotFoundError(f"PDB file not found at: {pdb_path}")

    # Build graphs
    print(f"[ECABSD] Building graph for chain {chain_a}...")
    data_a = build_residue_graph(pdb_path, chain_a).to(device)

    data_b = None
    if chain_b:
        print(f"[ECABSD] Building graph for chain {chain_b}...")
        try:
            data_b = build_residue_graph(pdb_path, chain_b).to(device)
        except (ValueError, KeyError) as e:
            print(f"[ECABSD] WARNING: Could not build graph for chain {chain_b}: {e}")
            print(f"[ECABSD] Falling back to self-attention on chain {chain_a}.")

    # Predict
    probs, labels, attn = model.predict(data_a, data_b, threshold=threshold)
    probs_np = probs.squeeze(-1).cpu().numpy()
    
    # Dynamic range adjustment if auto: keep binding ratio in 10% - 40%
    if is_auto:
        total_res = len(probs_np)
        binding_count = int((probs_np >= threshold).sum())
        ratio = binding_count / total_res if total_res > 0 else 0.0
        
        if ratio < 0.10:
            # Too strict, lower the threshold to get more binding residues
            for t in [0.55, 0.50, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20, 0.15, 0.10, 0.05]:
                new_count = int((probs_np >= t).sum())
                new_ratio = new_count / total_res
                if new_ratio >= 0.10:
                    threshold = t
                    labels = (probs >= t).int()
                    print(f"[ECABSD] Dynamic threshold adjustment: lowered to {t:.4f} to reach binding ratio of {new_ratio*100:.2f}% (range 10%-40%)")
                    break
        elif ratio > 0.40:
            # Too broad, raise the threshold to filter false positives
            for t in [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]:
                new_count = int((probs_np >= t).sum())
                new_ratio = new_count / total_res
                if new_ratio <= 0.40:
                    threshold = t
                    labels = (probs >= t).int()
                    print(f"[ECABSD] Dynamic threshold adjustment: raised to {t:.4f} to reach binding ratio of {new_ratio*100:.2f}% (range 10%-40%)")
                    break
                    
    probs = probs_np
    labels = labels.cpu().numpy()

    # Get residue info for output
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_path)
    chain = structure[0][chain_a]
    residues, _ = get_residues(chain)

    # Build per-residue results
    residue_results = []
    binding_residues = []
    for i, r in enumerate(residues):
        res_info = {
            "index": i,
            "resname": r.get_resname(),
            "resid": r.get_id()[1],
            "chain": chain_a,
            "probability": float(probs[i]),
            "is_binding": bool(labels[i]),
        }
        residue_results.append(res_info)
        if labels[i]:
            binding_residues.append(res_info)

    # Calculate quality
    total_residues = len(residues)
    binding_residues_count = len(binding_residues)
    quality = "Unknown"
    
    if total_residues > 0:
        binding_ratio = binding_residues_count / total_residues
        if binding_ratio < 0.08:
            quality = "Too strict / too few predicted binding residues"
        elif 0.08 <= binding_ratio <= 0.20:
            quality = "Good realistic range"
        elif 0.21 <= binding_ratio <= 0.40:
            quality = "Broad interface prediction"
        else:
            quality = "Overprediction - use higher threshold or exclude"
    else:
        binding_ratio = 0.0

    # Summary
    results = {
        "pdb_file": os.path.basename(pdb_path),
        "chain_a": chain_a,
        "chain_b": chain_b,
        "threshold": threshold,
        "total_residues": total_residues,
        "binding_residues_count": binding_residues_count,
        "binding_ratio": binding_ratio,
        "prediction_quality": quality,
        "residues": residue_results,
    }

    # Print summary
    print(f"\n{'='*60}")
    print(f"  ECABSD Prediction Results")
    print(f"{'='*60}")
    print(f"  PDB:               {os.path.basename(pdb_path)}")
    print(f"  Chain A:           {chain_a}")
    print(f"  Chain B:           {chain_b or 'None (self-attention)'}")
    print(f"  Total residues:    {total_residues}")
    print(f"  Threshold:         {threshold}")
    print(f"  Binding residues:  {binding_residues_count}")
    print(f"  Binding Ratio:     {binding_ratio*100:.2f}%")
    print(f"  Prediction Quality: {quality}")
    print(f"{'='*60}")

    if binding_residues:
        print(f"\n  Predicted Binding Site Residues:")
        print(f"  {'Idx':>4s}  {'Res':>4s}  {'ID':>5s}  {'Prob':>6s}")
        print(f"  {'-'*25}")
        for br in binding_residues:
            print(f"  {br['index']:4d}  {br['resname']:>4s}  {br['resid']:5d}  {br['probability']:.4f}")
    print()

    # Save results
    pdb_name = os.path.splitext(os.path.basename(pdb_path))[0]
    out_dir = os.path.join(results_dir, pdb_name)
    os.makedirs(out_dir, exist_ok=True)

    if output_path is None:
        output_path = os.path.join(out_dir, "predictions.json")

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved to: {output_path}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ECABSD Prediction")
    parser.add_argument("--pdb", required=True, help="PDB file path")
    parser.add_argument("--chain-a", required=True, help="Target chain ID")
    parser.add_argument("--chain-b", default=None, help="Partner chain ID")
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Decision threshold (default: loaded from checkpoint)")
    parser.add_argument("--output", default=None)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--model-version", type=str, choices=["v2", "v3"], default="v3", help="Which model architecture to use")
    args = parser.parse_args()

    run_prediction(
        pdb_path=args.pdb,
        chain_a=args.chain_a,
        chain_b=args.chain_b,
        checkpoint_path=args.checkpoint,
        threshold=args.threshold,
        output_path=args.output,
        config_path=args.config,
        model_version=args.model_version,
    )
