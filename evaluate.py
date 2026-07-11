"""
ECABSD Evaluation Pipeline.

Evaluates a trained model on the test set and produces:
- Accuracy, Precision, Recall, F1, MCC, AUC-ROC, AUC-PR
- Confusion matrix plot
- Per-structure breakdown
"""

import os
import json
import yaml
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    brier_score_loss,
)

from models import ECABSDModel


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def plot_confusion_matrix(cm, output_path):
    """Save confusion matrix as an image."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns

        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Non-binding", "Binding"],
            yticklabels=["Non-binding", "Binding"],
            ax=ax,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("ECABSD — Confusion Matrix")
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        print(f"  Confusion matrix saved to: {output_path}")
    except ImportError:
        print("  [WARN] matplotlib/seaborn not available; skipping confusion matrix plot.")


def run_evaluation(config_path: str = "config.yaml", checkpoint_path: str = "checkpoints/best_model_v3.pt"):
    """Run full evaluation on test set."""
    cfg = load_config(config_path)
    mcfg = cfg["model"]
    results_dir = cfg["paths"]["results_dir"]
    os.makedirs(results_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[ECABSD] Evaluating on device: {device}")

    # Load model — must match training architecture exactly (V3)
    model = ECABSDModel(
        input_dim=mcfg.get("esm_dim", 33),
        hidden_dim=mcfg["hidden_dim"],
        num_heads=mcfg["num_heads"],
        dropout=0.0,  # No dropout during evaluation
        edge_dim=mcfg.get("edge_feature_dim", 5),
        num_gcn_layers=mcfg.get("num_gcn_layers", 6),
    ).to(device)

    # Load checkpoint and recover saved threshold
    saved_threshold = cfg["prediction"].get("threshold", 0.5)
    norm_path = os.path.normpath(checkpoint_path)
    default_norm_path = os.path.normpath("checkpoints/best_model_v3.pt")
    if not os.path.exists(norm_path) and norm_path == default_norm_path:
        print(f"[ECABSD] Checkpoint not found. Automatically downloading best_model_v3.pt...")
        try:
            from download_weights import download
            download()
        except Exception as e:
            print(f"[ECABSD] WARNING: Failed to automatically download checkpoint: {e}")

    if os.path.exists(norm_path):
        checkpoint = torch.load(norm_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        saved_threshold = checkpoint.get("best_threshold", saved_threshold)
        print(f"[ECABSD] Loaded checkpoint from: {norm_path}")
        print(f"[ECABSD] Using saved threshold: {saved_threshold:.4f}")
    else:
        print(f"[ECABSD] WARNING: No checkpoint found at {norm_path}")
        print(f"[ECABSD] Running with random weights for demonstration.")

    model.eval()

    # Load test data
    processed_dir = cfg["data"]["processed_dir"]
    splits_csv = cfg["data"]["splits_csv"]

    all_probs = []
    all_labels = []
    per_pdb_results = {}

    if os.path.exists(processed_dir) and os.path.exists(splits_csv):
        from data.dataset import BindingSiteDataset, collate_fn
        from torch.utils.data import DataLoader

        test_dataset = BindingSiteDataset(processed_dir, splits_csv, split="test")
        test_loader = DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_fn
        )

        with torch.no_grad():
            for batch in test_loader:
                data_a = batch["data_a"].to(device)
                data_b = batch["data_b"].to(device)   # always a Batch (collate_fn guarantees)
                labels = batch["labels"]
                pdb_ids = batch["pdb_id"]

                logits, _ = model(data_a, data_b)
                probs = torch.sigmoid(logits).squeeze(-1).cpu().numpy()
                labels_np = labels.cpu().numpy()
                
                all_probs.extend(probs.tolist())
                all_labels.extend(labels_np.tolist())
                
                # For batch size 1
                pid = pdb_ids[0]
                per_pdb_results[pid] = {
                    "probs": probs.tolist(),
                    "labels": labels_np.tolist()
                }
    else:
        print(f"[ECABSD] No processed test data found. Please run scripts/prepare_dataset.py.")
        return

    # Compute metrics
    all_probs  = np.array(all_probs)
    all_labels = np.array(all_labels)

    best_threshold = saved_threshold  # comes from checkpoint["best_threshold"]
    print(f"  [Threshold] Using val-optimised threshold: {best_threshold:.4f}")

    all_preds = (all_probs >= best_threshold).astype(int)

    metrics = {
        "accuracy":              float(accuracy_score(all_labels, all_preds)),
        "precision":             float(precision_score(all_labels, all_preds, zero_division=0)),
        "recall":                float(recall_score(all_labels, all_preds, zero_division=0)),
        "f1":                    float(f1_score(all_labels, all_preds, zero_division=0)),
        "mcc":                   float(matthews_corrcoef(all_labels, all_preds)),
        "threshold":             float(best_threshold),
        "num_samples":           len(all_labels),
        "num_positive":          int(all_labels.sum()),
        "num_predicted_positive": int(all_preds.sum()),
    }

    # AUC metrics (need both classes present)
    if len(np.unique(all_labels)) > 1:
        metrics["auc_roc"] = float(roc_auc_score(all_labels, all_probs))
        metrics["auc_pr"] = float(average_precision_score(all_labels, all_probs))
        metrics["brier_score"] = float(brier_score_loss(all_labels, all_probs))
        
        # Precision@15%
        k = int(len(all_probs) * 0.15)
        top_k_indices = np.argsort(all_probs)[-k:]
        top_k_labels = all_labels[top_k_indices]
        metrics["precision@15"] = float(top_k_labels.sum() / max(k, 1))
    else:
        metrics["auc_roc"] = None
        metrics["auc_pr"] = None
        metrics["brier_score"] = None
        metrics["precision@15"] = None

    # Print results
    print(f"\n{'='*50}")
    print("  ECABSD V3 Evaluation Results")
    print(f"{'='*50}")
    for key, val in metrics.items():
        if isinstance(val, float):
            print(f"  {key:>25s}: {val:.4f}")
        else:
            print(f"  {key:>25s}: {val}")
    print(f"{'='*50}\n")

    # Save metrics
    metrics_path = os.path.join(results_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Metrics saved to: {metrics_path}")

    # Failure Analysis
    failures = []
    for pid, res in per_pdb_results.items():
        p_probs = np.array(res["probs"])
        p_labels = np.array(res["labels"])
        p_preds = (p_probs >= best_threshold).astype(int)
        
        if len(np.unique(p_labels)) > 1:
            p_f1 = f1_score(p_labels, p_preds, zero_division=0)
            if p_f1 < 0.3:
                num_pred_pos = int(p_preds.sum())
                num_true_pos = int(p_labels.sum())
                failures.append({
                    "pdb_id": pid,
                    "f1_score": float(p_f1),
                    "issue": "Underprediction" if num_pred_pos < (0.5 * num_true_pos) else ("Overprediction" if num_pred_pos > (2.0 * num_true_pos) else "Poor Accuracy"),
                    "predicted_residues": num_pred_pos,
                    "actual_residues": num_true_pos
                })
    
    if failures:
        fail_path = os.path.join(results_dir, "failed_structures.json")
        with open(fail_path, "w") as f:
            json.dump(failures, f, indent=2)
        print(f"  [Failure Analysis] Found {len(failures)} poorly performing structures. Logged to: {fail_path}")

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    cm_path = os.path.join(results_dir, "confusion_matrix.png")
    plot_confusion_matrix(cm, cm_path)

    return metrics


if __name__ == "__main__":
    run_evaluation()
