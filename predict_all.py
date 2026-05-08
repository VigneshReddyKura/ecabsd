"""
ECABSD — Full Pipeline in One Command.

Auto-downloads PDB, auto-finds best threshold, runs prediction, generates all visualizations.

Usage:
    python predict_all.py --pdb-id 1AY7 --chain-a A --chain-b B
    python predict_all.py --pdb-id 1AY7 --chain-a A --chain-b B --auto-threshold
    python predict_all.py --pdb 1AY7.pdb --chain-a A --chain-b B --threshold 0.8
"""

import os
import json
import yaml
import argparse
import urllib.request
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch

from Bio.PDB import PDBParser
from models.ecabsd_model import ECABSDModel
from models.graph_construction import build_residue_graph, get_residues


# ── Config ────────────────────────────────────────────────────────────────────
def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


# ── Auto-download PDB ─────────────────────────────────────────────────────────
def download_pdb(pdb_id, save_dir="data/raw/pdbs"):
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{pdb_id}.pdb")
    if os.path.exists(save_path):
        print(f"[ECABSD] PDB already exists: {save_path}")
        return save_path
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    print(f"[ECABSD] Downloading {pdb_id} from RCSB...")
    urllib.request.urlretrieve(url, save_path)
    print(f"[ECABSD] Saved to: {save_path}")
    return save_path


# ── Load model once ───────────────────────────────────────────────────────────
def load_model(checkpoint_path, cfg):
    mcfg = cfg["model"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ECABSDModel(
        input_dim=mcfg["input_dim"],
        hidden_dim=mcfg["hidden_dim"],
        num_heads=mcfg["num_heads"],
        dropout=0.0,
    ).to(device)
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"[ECABSD] Loaded checkpoint: {checkpoint_path}")
    else:
        print(f"[ECABSD] WARNING: No checkpoint at {checkpoint_path}. Using random weights.")
    return model, device


# ── Prediction ────────────────────────────────────────────────────────────────
def run_prediction(pdb_path, chain_a, chain_b, threshold, cfg, model=None, device=None, data_a=None, data_b=None, silent=False):
    """Run prediction. Accepts pre-built model and graphs to avoid reloading."""

    # Build graphs only if not passed in
    if data_a is None:
        if not silent:
            print(f"[ECABSD] Building graph for chain {chain_a}...")
        data_a = build_residue_graph(pdb_path, chain_a).to(device)

    if data_b is None and chain_b:
        if not silent:
            print(f"[ECABSD] Building graph for chain {chain_b}...")
        try:
            data_b = build_residue_graph(pdb_path, chain_b).to(device)
        except Exception as e:
            if not silent:
                print(f"[ECABSD] WARNING: Could not build chain {chain_b}: {e}")

    probs, labels, _ = model.predict(data_a, data_b, threshold=threshold)
    probs = probs.squeeze(-1).cpu().numpy()
    labels = labels.cpu().numpy()

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_path)
    chain = structure[0][chain_a]
    residues, _ = get_residues(chain)

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

    total = len(residues)
    binding_count = len(binding_residues)
    binding_ratio = binding_count / total if total > 0 else 0.0

    if binding_ratio < 0.08:
        quality = "Too strict — try lower threshold"
    elif binding_ratio <= 0.20:
        quality = "Good realistic range"
    elif binding_ratio <= 0.40:
        quality = "Broad interface prediction"
    else:
        quality = "Overprediction — try higher threshold"

    results = {
        "pdb_file": os.path.basename(pdb_path),
        "chain_a": chain_a,
        "chain_b": chain_b,
        "threshold": threshold,
        "total_residues": total,
        "binding_residues_count": binding_count,
        "binding_ratio": binding_ratio,
        "prediction_quality": quality,
        "residues": residue_results,
    }

    if not silent:
        print(f"\n{'='*60}")
        print(f"  ECABSD Prediction Results")
        print(f"{'='*60}")
        print(f"  PDB:               {os.path.basename(pdb_path)}")
        print(f"  Chain A (target):  {chain_a}")
        print(f"  Chain B (partner): {chain_b or 'None'}")
        print(f"  Total residues:    {total}")
        print(f"  Threshold:         {threshold}")
        print(f"  Binding residues:  {binding_count} ({binding_ratio*100:.1f}%)")
        print(f"  Quality:           {quality}")
        print(f"{'='*60}")

        if binding_residues:
            print(f"\n  Top binding residues:")
            print(f"  {'Idx':>4}  {'Res':>4}  {'ID':>5}  {'Prob':>6}")
            print(f"  {'-'*26}")
            sorted_br = sorted(binding_residues, key=lambda x: x['probability'], reverse=True)
            for br in sorted_br[:15]:
                print(f"  {br['index']:4d}  {br['resname']:>4s}  {br['resid']:5d}  {br['probability']:.4f}")
            if len(binding_residues) > 15:
                print(f"  ... and {len(binding_residues)-15} more")
        print()

    return results, probs, labels, data_a, data_b


# ── Auto Threshold ────────────────────────────────────────────────────────────
def find_best_threshold(pdb_path, chain_a, chain_b, cfg, model, device, target_ratio=0.15):
    """
    Try multiple thresholds and pick the one giving binding ratio
    closest to target_ratio (default 15% — biological sweet spot).
    Builds graphs only once, reuses them for all threshold attempts.
    """
    print(f"\n[ECABSD] Auto-finding best threshold (target ratio: {target_ratio*100:.0f}%)...")

    # Build graphs once
    print(f"[ECABSD] Building graphs...")
    data_a = build_residue_graph(pdb_path, chain_a).to(device)
    data_b = None
    if chain_b:
        try:
            data_b = build_residue_graph(pdb_path, chain_b).to(device)
        except Exception as e:
            print(f"[ECABSD] WARNING: Could not build chain {chain_b}: {e}")

    thresholds = [0.30, 0.40, 0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.92, 0.95, 0.97, 0.99]
    best_threshold = 0.5
    best_diff = float('inf')

    print(f"\n  {'Threshold':>10}  {'Ratio':>8}  {'Quality'}")
    print(f"  {'-'*45}")

    for t in thresholds:
        results, _, _, _, _ = run_prediction(
            pdb_path=pdb_path,
            chain_a=chain_a,
            chain_b=chain_b,
            threshold=t,
            cfg=cfg,
            model=model,
            device=device,
            data_a=data_a,
            data_b=data_b,
            silent=True,
        )
        ratio = results["binding_ratio"]
        quality = results["prediction_quality"]
        diff = abs(ratio - target_ratio)
        marker = " <-- BEST" if diff < best_diff else ""
        print(f"  {t:>10.2f}  {ratio*100:>7.1f}%  {quality}{marker}")

        if diff < best_diff:
            best_diff = diff
            best_threshold = t

    print(f"\n[ECABSD] Best threshold selected: {best_threshold}")
    print(f"[ECABSD] Binding ratio at best threshold: {(target_ratio - best_diff + best_diff)*100:.0f}% approx\n")
    return best_threshold, data_a, data_b


# ── Grad-CAM ──────────────────────────────────────────────────────────────────
def compute_gradcam(model, data_a, data_b):
    model.eval()
    data_a.x = data_a.x.float()
    data_a.x.requires_grad_(True)

    pred, _ = model(data_a, data_b)
    pred = pred.squeeze(-1)
    score = pred.sum()
    model.zero_grad()
    score.backward()

    grads = data_a.x.grad.detach().cpu().numpy()
    saliency = np.abs(grads).mean(axis=1)
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
    return saliency


# ── Visualizations ────────────────────────────────────────────────────────────
def generate_all_visualizations(results, saliency, out_dir, sample_id, chain_a):
    os.makedirs(out_dir, exist_ok=True)

    residues = results["residues"]
    probs = np.array([r["probability"] for r in residues])
    labels = np.array([r["is_binding"] for r in residues])
    indices = np.arange(len(residues))

    fig = plt.figure(figsize=(16, 14))
    fig.suptitle(f"ECABSD Results — {sample_id} Chain {chain_a}", fontsize=14, fontweight='bold', y=0.98)
    gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.5, wspace=0.35)

    # Plot 1 — Binding probability heatmap
    ax1 = fig.add_subplot(gs[0, :])
    im1 = ax1.imshow(probs.reshape(1, -1), aspect="auto", cmap="viridis", vmin=0, vmax=1)
    plt.colorbar(im1, ax=ax1, label="Binding Probability")
    ax1.set_title("Binding Probability Heatmap", fontsize=11)
    ax1.set_xlabel("Residue Index")
    ax1.set_yticks([])

    # Plot 2 — Grad-CAM heatmap
    ax2 = fig.add_subplot(gs[1, :])
    im2 = ax2.imshow(saliency.reshape(1, -1), aspect="auto", cmap="plasma", vmin=0, vmax=1)
    plt.colorbar(im2, ax=ax2, label="Grad-CAM Importance")
    ax2.set_title("Grad-CAM Explainability Heatmap", fontsize=11)
    ax2.set_xlabel("Residue Index")
    ax2.set_yticks([])

    # Plot 3 — Probability bar chart
    ax3 = fig.add_subplot(gs[2, :])
    colors = ['#D85A30' if l else '#378ADD' for l in labels]
    ax3.bar(indices, probs, color=colors, alpha=0.8, width=1.0)
    ax3.axhline(y=results["threshold"], color='black', linestyle='--', linewidth=1.2)
    ax3.set_title("Per-Residue Binding Probability", fontsize=11)
    ax3.set_xlabel("Residue Index")
    ax3.set_ylabel("Probability")
    ax3.set_ylim(0, 1.05)
    legend_elements = [
        Patch(facecolor='#D85A30', label='Binding'),
        Patch(facecolor='#378ADD', label='Non-binding'),
        plt.Line2D([0], [0], color='black', linestyle='--', label=f'Threshold ({results["threshold"]})')
    ]
    ax3.legend(handles=legend_elements, fontsize=9)

    # Plot 4 — Top binding residues
    ax4 = fig.add_subplot(gs[3, 0])
    binding = [(r["resname"], r["resid"], r["probability"]) for r in residues if r["is_binding"]]
    binding_sorted = sorted(binding, key=lambda x: x[2], reverse=True)[:10]
    if binding_sorted:
        names = [f"{r[0]}{r[1]}" for r in binding_sorted]
        scores = [r[2] for r in binding_sorted]
        bars = ax4.barh(names, scores, color='#D85A30', alpha=0.8)
        ax4.set_title("Top 10 Binding Residues", fontsize=11)
        ax4.set_xlabel("Probability")
        ax4.set_xlim(0, 1.05)
        for bar, score in zip(bars, scores):
            ax4.text(score + 0.01, bar.get_y() + bar.get_height()/2, f'{score:.3f}', va='center', fontsize=8)
    else:
        ax4.text(0.5, 0.5, "No binding residues\ndetected", ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title("Top Binding Residues", fontsize=11)

    # Plot 5 — Summary stats
    ax5 = fig.add_subplot(gs[3, 1])
    ax5.axis('off')
    total = results["total_residues"]
    binding_count = results["binding_residues_count"]
    ratio = results["binding_ratio"] * 100
    quality = results["prediction_quality"]
    summary_text = (
        f"Summary\n"
        f"{'─'*28}\n"
        f"Total residues:    {total}\n"
        f"Binding residues:  {binding_count}\n"
        f"Non-binding:       {total - binding_count}\n"
        f"Binding ratio:     {ratio:.1f}%\n"
        f"Threshold:         {results['threshold']}\n"
        f"Quality:           {quality}\n"
        f"Chain A:           {results['chain_a']}\n"
        f"Chain B:           {results['chain_b'] or 'None'}"
    )
    ax5.text(0.05, 0.95, summary_text, transform=ax5.transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='#E1F5EE', alpha=0.5))

    out_png = os.path.join(out_dir, f"full_results_{sample_id}_{chain_a}.png")
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"[ECABSD] Visualization saved to: {out_png}")
    return out_png


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="ECABSD — Full Pipeline")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--pdb-id", help="PDB ID to auto-download (e.g. 1AY7)")
    group.add_argument("--pdb", help="Path to local PDB file")
    parser.add_argument("--chain-a", required=True, help="Target chain ID")
    parser.add_argument("--chain-b", default=None, help="Partner chain ID")
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    parser.add_argument("--threshold", type=float, default=0.5, help="Manual threshold (ignored if --auto-threshold)")
    parser.add_argument("--auto-threshold", action="store_true", help="Auto-find best threshold for this protein")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    results_dir = cfg["paths"]["results_dir"]

    # Step 1 — Get PDB file
    if args.pdb_id:
        pdb_path = download_pdb(args.pdb_id.upper())
        sample_id = args.pdb_id.upper()
    else:
        pdb_path = args.pdb
        sample_id = os.path.splitext(os.path.basename(pdb_path))[0]

    # Step 2 — Load model once
    model, device = load_model(args.checkpoint, cfg)

    # Step 3 — Auto threshold or manual
    if args.auto_threshold:
        threshold, data_a, data_b = find_best_threshold(
            pdb_path=pdb_path,
            chain_a=args.chain_a,
            chain_b=args.chain_b,
            cfg=cfg,
            model=model,
            device=device,
        )
    else:
        threshold = args.threshold
        data_a = None
        data_b = None

    # Step 4 — Final prediction
    results, probs, labels, data_a, data_b = run_prediction(
        pdb_path=pdb_path,
        chain_a=args.chain_a,
        chain_b=args.chain_b,
        threshold=threshold,
        cfg=cfg,
        model=model,
        device=device,
        data_a=data_a,
        data_b=data_b,
    )

    # Step 5 — Grad-CAM
    print("[ECABSD] Computing Grad-CAM explainability...")
    saliency = compute_gradcam(model, data_a, data_b)

    # Step 6 — Save JSONs
    out_dir = os.path.join(results_dir, sample_id)
    os.makedirs(out_dir, exist_ok=True)

    pred_json = os.path.join(out_dir, "predictions.json")
    with open(pred_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[ECABSD] Predictions saved to: {pred_json}")

    gradcam_json = os.path.join(out_dir, f"gradcam_{sample_id}_{args.chain_a}.json")
    with open(gradcam_json, "w") as f:
        json.dump({
            "pdb": sample_id,
            "chain": args.chain_a,
            "residues": [{"index": int(i), "score": float(s)} for i, s in enumerate(saliency)]
        }, f, indent=2)
    print(f"[ECABSD] Grad-CAM saved to: {gradcam_json}")

    # Step 7 — Visualizations
    print("[ECABSD] Generating visualizations...")
    generate_all_visualizations(results, saliency, out_dir, sample_id, args.chain_a)

    print(f"\n[ECABSD] All done! Results saved in: {out_dir}")
    print(f"   predictions.json")
    print(f"   gradcam_{sample_id}_{args.chain_a}.json")
    print(f"   full_results_{sample_id}_{args.chain_a}.png")


if __name__ == "__main__":
    main()
