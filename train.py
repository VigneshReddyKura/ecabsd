"""
ECABSD Training Pipeline — v3 "Best Model"

Handles:
- Config loading
- Dataset construction
- Model initialization
- Training loop with Weighted BCE Loss, early stopping
- Checkpoint saving
- Metric logging
"""

import os
import json
import time
import random
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
)

from models.ecabsd_model import ECABSDModel
from data.dataset import BindingSiteDataset, collate_fn


# ── Weighted BCE Loss ─────────────────────────────────────────────────────────
class WeightedBCELoss(nn.Module):
    """
    Weighted Binary Cross Entropy Loss for class imbalance.
    pos_weight=4.67 matches the dataset ratio (non-binding:binding).
    """
    def __init__(self, pos_weight=4.67):
        super().__init__()
        self.pos_weight = pos_weight

    def forward(self, pred, target):
        loss = nn.functional.binary_cross_entropy(
            pred, target, reduction='none'
        )
        weights = torch.where(
            target == 1,
            torch.tensor(self.pos_weight, device=pred.device),
            torch.tensor(1.0, device=pred.device)
        )
        return (loss * weights).mean()


# ── Helpers ───────────────────────────────────────────────────────────────────
def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_pos_weight(dataset) -> float:
    total_pos = total_neg = 0
    for sample in dataset:
        labels = sample["labels"]
        total_pos += int(labels.sum().item())
        total_neg += int((labels == 0).sum().item())
    return (total_neg / total_pos) if total_pos > 0 else 7.0


# ── Train / Validate ──────────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion, device, gradient_clip):
    model.train()
    total_loss = 0.0
    all_labels, all_preds = [], []
    total_gnorm = 0.0
    n_batches   = 0

    for sample in loader:
        data_a = sample["data_a"].to(device)
        data_b = sample["data_b"].to(device) if sample["data_b"] is not None else None
        labels = sample["labels"].to(device)

        optimizer.zero_grad()
        logits, _ = model(data_a, data_b)
        logits    = logits.squeeze(-1)

        loss = criterion(pred, labels.float())
        loss.backward()

        if gradient_clip > 0:
            gnorm = nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            total_gnorm += gnorm.item()

        optimizer.step()

        total_loss   += loss.item() * labels.size(0)
        probs         = torch.sigmoid(logits)
        binary_preds  = (probs >= 0.5).long().cpu().numpy()
        all_labels.extend(labels.cpu().numpy().tolist())
        all_preds.extend(binary_preds.tolist())
        n_batches += 1

    avg_loss        = total_loss / max(len(all_labels), 1)
    metrics         = compute_metrics(all_labels, all_preds)
    metrics["loss"] = avg_loss
    metrics["grad_norm"] = total_gnorm / max(n_batches, 1)
    return metrics


@torch.no_grad()
def validate(model, loader, criterion, device):
    """Run validation."""
    model.eval()
    total_loss = 0.0
    all_labels, all_preds, all_probs = [], [], []

    for sample in loader:
        data_a = sample["data_a"].to(device)
        data_b = sample["data_b"].to(device) if sample["data_b"] is not None else None
        labels = sample["labels"].to(device)

        logits, _ = model(data_a, data_b)
        logits    = logits.squeeze(-1)

        loss = criterion(pred, labels.float())
        total_loss += loss.item() * labels.size(0)

        probs        = torch.sigmoid(logits)
        binary_preds = (probs >= 0.5).long().cpu().numpy()
        all_labels.extend(labels.cpu().numpy().tolist())
        all_preds.extend(binary_preds.tolist())
        all_probs.extend(probs.cpu().numpy().tolist())

    avg_loss        = total_loss / max(len(all_labels), 1)
    metrics         = compute_metrics(all_labels, all_preds)
    metrics["loss"] = avg_loss
    return metrics, np.array(all_labels), np.array(all_probs)


# ── Main Training Function ────────────────────────────────────────────────────
def run_training(config_path: str = "config.yaml", resume_from: str = None):
    cfg  = load_config(config_path)
    tcfg = cfg["training"]
    mcfg = cfg["model"]
    pcfg = cfg["paths"]

    set_seed(tcfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[ECABSD] Training on device: {device}")

    os.makedirs(pcfg["checkpoints_dir"], exist_ok=True)
    os.makedirs(pcfg["logs_dir"],        exist_ok=True)

    # ── Build model ────────────────────────────────────────────────────────────
    model = ECABSDModel(
        input_dim=mcfg["input_dim"],
        hidden_dim=mcfg["hidden_dim"],
        num_heads=mcfg["num_heads"],
        dropout=mcfg["dropout"],
        edge_dim=mcfg["edge_feature_dim"],
        num_ca_layers=mcfg.get("num_ca_layers", 4),
    ).to(device)

    print(f"[ECABSD] Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ── Optimizer — AdamW with decoupled weight decay ─────────────────────────
    optimizer = AdamW(
        model.parameters(),
        lr=tcfg["learning_rate"],
        weight_decay=tcfg["weight_decay"],
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    # LR scheduler
    if tcfg["lr_scheduler"] == "plateau":
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            patience=tcfg["lr_patience"],
            factor=tcfg["lr_factor"],
        )
    elif tcfg["lr_scheduler"] == "step":
        scheduler = StepLR(optimizer, step_size=tcfg["lr_patience"], gamma=tcfg["lr_factor"])
    elif tcfg["lr_scheduler"] == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=tcfg["epochs"])
    else:
        scheduler = None

    # Loss function — Weighted BCE for class imbalance
    criterion = WeightedBCELoss(pos_weight=4.67)

    # Dataset & loaders
    processed_dir = cfg["data"]["processed_dir"]
    splits_csv    = cfg["data"]["splits_csv"]

    if os.path.exists(processed_dir) and os.path.exists(splits_csv):
        train_dataset = BindingSiteDataset(processed_dir, splits_csv, split="train")
        val_dataset   = BindingSiteDataset(processed_dir, splits_csv, split="val")

        train_loader = DataLoader(
            train_dataset, batch_size=1, shuffle=True,
            num_workers=0, collate_fn=collate_fn,
        )
        val_loader = DataLoader(
            val_dataset, batch_size=1, shuffle=False,
            num_workers=0, collate_fn=collate_fn,
        )
    else:
        print(f"[ECABSD] WARNING: Processed data not found at '{processed_dir}'.")
        print(f"[ECABSD] Run 'python scripts/prepare_dataset.py' first.")
        from models.graph_construction import build_residue_graph
        sample_pdb = "1AY7.pdb"
        if os.path.exists(sample_pdb):
            data_a = build_residue_graph(sample_pdb, "A")
            data_a.y = torch.zeros(data_a.num_residues)
            data_a.y[:10] = 1.0

            class DummyBatch:
                def __init__(self, data):
                    self.data = data
                def __iter__(self):
                    yield {"data_a": self.data, "data_b": None, "labels": self.data.y}
                def __len__(self):
                    return 1

            train_loader = DummyBatch(data_a)
            val_loader   = DummyBatch(data_a)
            pos_weight_val = 7.0
        else:
            print("[ECABSD] ERROR: No PDB file found. Cannot train.")
            return
        data_a    = build_residue_graph(sample_pdb, "A")
        data_a.y  = torch.zeros(data_a.num_residues)
        data_a.y[:10] = 1.0

        class DummyBatch:
            def __iter__(self):
                yield {"data_a": data_a, "data_b": None, "labels": data_a.y}
            def __len__(self):
                return 1

        train_loader = val_loader = DummyBatch()
        train_dataset = list(train_loader)

    # ── Loss ──────────────────────────────────────────────────────────────────
    criterion = build_criterion(tcfg, device)

    # ── LR scheduler: warmup → cosine ─────────────────────────────────────────
    scheduler = build_scheduler(optimizer, tcfg, steps_per_epoch=len(train_loader))

    # ── Resume ────────────────────────────────────────────────────────────────
    start_epoch   = 0
    best_val_f1   = 0.0
    best_threshold = 0.5

    if resume_from and os.path.exists(resume_from):
        ckpt = torch.load(resume_from, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch    = ckpt.get("epoch", 0) + 1
        best_val_f1    = ckpt.get("best_val_f1", 0.0)
        best_threshold = ckpt.get("best_threshold", 0.5)
        print(f"[ECABSD] Resumed from epoch {start_epoch}  best_val_f1={best_val_f1:.4f}")

    # ── Training loop ─────────────────────────────────────────────────────────
    patience_counter = 0
    history          = []

    print(f"\n{'='*60}")
    print(f"  ECABSD Training v3 — {tcfg['epochs']} epochs")
    print(f"{'='*60}\n")

    for epoch in range(start_epoch, tcfg["epochs"]):
        t0 = time.time()

        train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion, device, tcfg["gradient_clip"]
        )
        val_metrics = validate(model, val_loader, criterion, device)

        scheduler.step()
        elapsed = time.time() - t0

        # Threshold sweep every 5 epochs after epoch 5
        if (epoch + 1) >= 5 and (epoch + 1) % 5 == 0 and len(np.unique(val_labels)) > 1:
            best_threshold, best_t_f1 = find_best_threshold(val_labels, val_probs)
            print(
                f"  [Threshold] Best val threshold: {best_threshold:.2f}"
                f"  F1={best_t_f1:.4f}"
            )

        lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch+1:03d}/{tcfg['epochs']} | "
            f"Train Loss: {train_metrics['loss']:.4f} F1: {train_metrics['f1']:.4f} "
            f"GNorm: {train_metrics['grad_norm']:.2f} | "
            f"Val Loss: {val_metrics['loss']:.4f} F1: {val_metrics['f1']:.4f} "
            f"MCC: {val_metrics['mcc']:.4f} | "
            f"LR: {lr:.2e} | {elapsed:.1f}s"
        )

        history.append({
            "epoch":     epoch + 1,
            "train":     train_metrics,
            "val":       val_metrics,
            "lr":        lr,
            "time":      elapsed,
            "threshold": best_threshold,
        })

        # ── Save best model by val F1 ─────────────────────────────────────────
        val_f1 = val_metrics["f1"]
        if val_f1 > best_val_f1:
            best_val_f1      = val_f1
            patience_counter = 0
            ckpt_path        = os.path.join(pcfg["checkpoints_dir"], "best_model.pt")
            torch.save({
                "epoch":                epoch,
                "model_state_dict":     model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_f1":          best_val_f1,
                "best_threshold":       best_threshold,
                "config":               cfg,
            }, ckpt_path)
            print(f"  -> Saved best model (val_F1={best_val_f1:.4f}  threshold={best_threshold:.2f})")
        else:
            patience_counter += 1

        # Save periodic checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save({
                "epoch":                epoch,
                "model_state_dict":     model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_f1":          best_val_f1,
                "best_threshold":       best_threshold,
                "config":               cfg,
            }, os.path.join(pcfg["checkpoints_dir"], f"epoch_{epoch+1}.pt"))

        # Early stopping
        if patience_counter >= tcfg["early_stopping_patience"]:
            print(f"\n[ECABSD] Early stopping at epoch {epoch+1}")
            break

    # ── Save history ──────────────────────────────────────────────────────────
    history_path = os.path.join(pcfg["logs_dir"], "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    # Write best threshold to config
    cfg_out = load_config(config_path)
    cfg_out["prediction"]["threshold"] = round(best_threshold, 4)
    with open(config_path, "w") as f:
        yaml.dump(cfg_out, f, default_flow_style=False, sort_keys=False)

    print(f"\n{'='*60}")
    print(f"  Training complete. Best val F1:  {best_val_f1:.4f}")
    print(f"  Best threshold:                  {best_threshold:.4f}")
    print(f"  History saved to:                {history_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    run_training()