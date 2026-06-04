"""
train_kfold.py
==============
K-fold cross-validation training for ECABSD.

Performs grouped k-fold splitting (by PDB complex ID) so no complex
appears in both train and validation within a fold. Reports mean ± std
metrics across all folds — required for peer-reviewed submissions.

Usage
-----
    python scripts/train_kfold.py \
        --config config.yaml \
        --splits data/splits_homology.csv \
        --folds 5 \
        --output results/kfold_results.json

Output
------
    results/kfold_results.json  — per-fold + summary metrics
    results/kfold_summary.txt   — human-readable summary table
    checkpoints/fold_k/         — best checkpoint per fold

Publication Note
----------------
    Report the mean ± std metrics from kfold_summary.txt.
    Use splits_homology.csv (MMseqs2-clustered) for full rigor.
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import yaml
    import pandas as pd
    import torch
    from torch.utils.data import DataLoader
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import (
        f1_score, roc_auc_score, average_precision_score,
        precision_score, recall_score, accuracy_score, matthews_corrcoef,
    )
except ImportError as e:
    print(f"ERROR: Missing dependency: {e}")
    print("Install with: pip install -r requirements.txt")
    sys.exit(1)

from train import (
    run_training, set_seed, load_config, compute_pos_weight,
    build_criterion, train_one_epoch, validate,
)
from models import ECABSDModel
from data.dataset import BindingSiteDataset, collate_fn


# ─────────────────────────────────────────────────────────────────────────────
# K-fold driver
# ─────────────────────────────────────────────────────────────────────────────

def run_kfold(
    config_path: str,
    splits_csv: str,
    n_folds: int = 5,
    output_path: str = "results/kfold_results.json",
    seed: int = 42,
):
    cfg  = load_config(config_path)
    tcfg = cfg["training"]
    mcfg = cfg["model"]
    pcfg = cfg["paths"]

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[K-Fold] Running {n_folds}-fold CV on device: {device}")

    df = pd.read_csv(splits_csv)
    # Use all non-test complexes for cross-validation
    df_cv = df[df["split"] != "test"].reset_index(drop=True)
    pdb_ids = df_cv["pdb_id"].values
    groups  = pdb_ids  # group = PDB ID (ensures no complex appears in both folds)

    processed_dir = cfg["data"]["processed_dir"]

    gkf = GroupKFold(n_splits=n_folds)
    fold_results = []

    for fold_idx, (train_idx, val_idx) in enumerate(
            gkf.split(df_cv, groups=groups)):
        fold_num = fold_idx + 1
        print(f"\n{'='*50}")
        print(f"  FOLD {fold_num}/{n_folds}")
        print(f"{'='*50}")

        train_pdbs = set(df_cv.iloc[train_idx]["pdb_id"].tolist())
        val_pdbs   = set(df_cv.iloc[val_idx]["pdb_id"].tolist())
        print(f"  Train: {len(train_pdbs)} complexes | Val: {len(val_pdbs)} complexes")

        # Write fold-specific splits CSV
        fold_dir  = os.path.join(pcfg["checkpoints_dir"], f"fold_{fold_num}")
        os.makedirs(fold_dir, exist_ok=True)

        fold_splits_path = os.path.join(fold_dir, "fold_splits.csv")
        df_fold = df.copy()
        df_fold.loc[df_fold["pdb_id"].isin(train_pdbs), "split"] = "train"
        df_fold.loc[df_fold["pdb_id"].isin(val_pdbs),   "split"] = "val"
        df_fold.to_csv(fold_splits_path, index=False)

        # Build datasets
        train_dataset = BindingSiteDataset(processed_dir, fold_splits_path, split="train")
        val_dataset   = BindingSiteDataset(processed_dir, fold_splits_path, split="val")

        if len(train_dataset) == 0 or len(val_dataset) == 0:
            print(f"  [WARN] Fold {fold_num} has empty dataset — skipping.")
            continue

        train_loader = DataLoader(
            train_dataset, batch_size=tcfg["batch_size"], shuffle=True,
            num_workers=tcfg.get("num_workers", 0), collate_fn=collate_fn,
        )
        val_loader = DataLoader(
            val_dataset, batch_size=tcfg["batch_size"], shuffle=False,
            num_workers=tcfg.get("num_workers", 0), collate_fn=collate_fn,
        )

        # Fresh model per fold
        set_seed(seed + fold_idx)
        model = ECABSDModel(
            input_dim=mcfg.get("esm_dim", 33),
            hidden_dim=mcfg["hidden_dim"],
            num_heads=mcfg["num_heads"],
            dropout=mcfg["dropout"],
            edge_dim=mcfg.get("edge_feature_dim", 5),
            num_gcn_layers=mcfg.get("num_gcn_layers", 6),
        ).to(device)

        from torch.optim import AdamW
        from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
        import torch.nn as nn

        optimizer = AdamW(
            model.parameters(),
            lr=tcfg["learning_rate"],
            weight_decay=tcfg["weight_decay"],
        )

        warmup_epochs = tcfg.get("warmup_epochs", 15)
        cosine_epochs = max(tcfg["epochs"] - warmup_epochs, 1)
        scheduler = SequentialLR(
            optimizer,
            schedulers=[
                LinearLR(optimizer, start_factor=0.01, end_factor=1.0,
                         total_iters=warmup_epochs),
                CosineAnnealingLR(optimizer, T_max=cosine_epochs, eta_min=1e-6),
            ],
            milestones=[warmup_epochs],
        )

        pos_weight  = compute_pos_weight(train_dataset)
        criterion   = build_criterion(tcfg, pos_weight, device)
        scaler      = torch.amp.GradScaler("cuda", enabled=False)

        best_val_f1  = -1.0
        best_metrics = {}
        patience     = 0
        patience_max = tcfg.get("early_stopping_patience", 60)

        for epoch in range(tcfg["epochs"]):
            train_one_epoch(
                model, train_loader, optimizer, criterion, device,
                tcfg["gradient_clip"], scaler,
                chain_swap_prob=tcfg.get("chain_swap_prob", 0.5),
            )
            val_metrics = validate(model, val_loader, criterion, device)
            scheduler.step()

            current_f1 = val_metrics["f1"]
            if current_f1 > best_val_f1:
                best_val_f1  = current_f1
                best_metrics = val_metrics.copy()
                patience     = 0
                ckpt_path = os.path.join(fold_dir, "best_model.pt")
                torch.save(model.state_dict(), ckpt_path)
                print(f"  Epoch {epoch+1:03d} | F1={current_f1:.4f} "
                      f"AUC-ROC={val_metrics.get('auc_roc', 0):.4f} ← best")
            else:
                patience += 1
                if patience >= patience_max:
                    print(f"  Early stopping at epoch {epoch+1}")
                    break

        print(f"\n  Fold {fold_num} best → F1={best_val_f1:.4f} "
              f"AUC-ROC={best_metrics.get('auc_roc', 0):.4f}")

        fold_results.append({
            "fold":      fold_num,
            "n_train":   len(train_dataset),
            "n_val":     len(val_dataset),
            "metrics":   best_metrics,
        })

    # ── Aggregate ──────────────────────────────────────────────────────────
    metric_keys = ["f1", "auc_roc", "auc_pr", "precision", "recall",
                   "accuracy", "mcc"]
    summary = {}
    for k in metric_keys:
        vals = [r["metrics"].get(k, np.nan) for r in fold_results]
        vals = [v for v in vals if not np.isnan(v)]
        if vals:
            summary[k] = {
                "mean":   float(np.mean(vals)),
                "std":    float(np.std(vals)),
                "values": vals,
            }

    output_data = {
        "n_folds":      n_folds,
        "splits_csv":   splits_csv,
        "config":       config_path,
        "seed":         seed,
        "fold_results": fold_results,
        "summary":      summary,
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\n[K-Fold] Results saved to {output_path}")

    # ── Human-readable summary ─────────────────────────────────────────────
    summary_txt = output_path.replace(".json", ".txt")
    with open(summary_txt, "w") as f:
        f.write(f"ECABSD — {n_folds}-Fold Cross-Validation Summary\n")
        f.write(f"Splits: {splits_csv}\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"{'Metric':<15} {'Mean':>8} {'±Std':>8}\n")
        f.write("-" * 35 + "\n")
        for k in metric_keys:
            if k in summary:
                f.write(f"{k:<15} {summary[k]['mean']:>8.4f} "
                        f"{summary[k]['std']:>8.4f}\n")
        f.write("\n")
        f.write("Per-fold F1 scores:\n")
        for r in fold_results:
            f.write(f"  Fold {r['fold']}: {r['metrics'].get('f1', 0):.4f}\n")

    print(f"\n{'='*50}")
    print(f"  ECABSD {n_folds}-Fold CV Summary")
    print(f"{'='*50}")
    print(f"  {'Metric':<15} {'Mean':>8} {'±Std':>8}")
    print(f"  {'-'*35}")
    for k in metric_keys:
        if k in summary:
            print(f"  {k:<15} {summary[k]['mean']:>8.4f} "
                  f"{summary[k]['std']:>8.4f}")
    print(f"\n  Full results: {output_path}")
    print(f"  Summary:      {summary_txt}\n")

    return output_data


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="K-fold cross-validation training for ECABSD."
    )
    parser.add_argument("--config",  default="config.yaml",
                        help="Path to config.yaml")
    parser.add_argument("--splits",  default="data/splits.csv",
                        help="Splits CSV (use splits_homology.csv for publication)")
    parser.add_argument("--folds",   type=int, default=5,
                        help="Number of CV folds (default: 5)")
    parser.add_argument("--output",  default="results/kfold_results.json",
                        help="Output path for results JSON")
    parser.add_argument("--seed",    type=int, default=42)
    args = parser.parse_args()

    run_kfold(
        config_path=args.config,
        splits_csv=args.splits,
        n_folds=args.folds,
        output_path=args.output,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
