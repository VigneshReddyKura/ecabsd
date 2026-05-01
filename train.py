"""
ECABSD Training Pipeline.

Handles:
- Config loading
- Dataset construction
- Model initialization
- Training loop with BCE loss, class weighting, early stopping
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
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CosineAnnealingLR
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


def load_config(config_path: str) -> dict:
    """Load YAML configuration."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_metrics(all_labels, all_preds):
    """Compute classification metrics."""
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    mcc = matthews_corrcoef(all_labels, all_preds)
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "mcc": mcc}


def train_one_epoch(model, loader, optimizer, criterion, device, gradient_clip, pos_weight):
    """Run one training epoch (processes one graph per step)."""
    model.train()
    total_loss = 0.0
    all_labels = []
    all_preds = []

    for sample in loader:
        data_a = sample["data_a"].to(device)
        data_b = sample["data_b"].to(device) if sample["data_b"] is not None else None
        labels = sample["labels"].to(device)

        optimizer.zero_grad()
        pred, _ = model(data_a, data_b)
        pred = pred.squeeze(-1)

        raw_loss = criterion(pred, labels.float())
        weights = torch.where(
            labels == 1,
            torch.tensor(pos_weight, device=device),
            torch.tensor(1.0, device=device)
        )
        loss = (raw_loss * weights).mean()
        loss.backward()

        # Gradient clipping
        if gradient_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)

        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        binary_preds = (pred >= 0.3).long().cpu().numpy()
        all_labels.extend(labels.cpu().numpy().tolist())
        all_preds.extend(binary_preds.tolist())

    avg_loss = total_loss / max(len(all_labels), 1)
    metrics = compute_metrics(all_labels, all_preds)
    metrics["loss"] = avg_loss
    return metrics


@torch.no_grad()
def validate(model, loader, criterion, device, pos_weight):
    """Run validation."""
    model.eval()
    total_loss = 0.0
    all_labels = []
    all_preds = []

    for sample in loader:
        data_a = sample["data_a"].to(device)
        data_b = sample["data_b"].to(device) if sample["data_b"] is not None else None
        labels = sample["labels"].to(device)

        pred, _ = model(data_a, data_b)
        pred = pred.squeeze(-1)

        raw_loss = criterion(pred, labels.float())
        weights = torch.where(
            labels == 1,
            torch.tensor(pos_weight, device=device),
            torch.tensor(1.0, device=device)
        )
        loss = (raw_loss * weights).mean()
        total_loss += loss.item() * labels.size(0)

        binary_preds = (pred >= 0.3).long().cpu().numpy()
        all_labels.extend(labels.cpu().numpy().tolist())
        all_preds.extend(binary_preds.tolist())

    avg_loss = total_loss / max(len(all_labels), 1)
    metrics = compute_metrics(all_labels, all_preds)
    metrics["loss"] = avg_loss
    return metrics


def run_training(config_path: str = "config.yaml", resume_from: str = None):
    """Main training function."""
    cfg = load_config(config_path)
    tcfg = cfg["training"]
    mcfg = cfg["model"]
    pcfg = cfg["paths"]

    set_seed(tcfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[ECABSD] Training on device: {device}")

    # Create output directories
    os.makedirs(pcfg["checkpoints_dir"], exist_ok=True)
    os.makedirs(pcfg["logs_dir"], exist_ok=True)

    # Build model
    model = ECABSDModel(
        input_dim=mcfg["input_dim"],
        hidden_dim=mcfg["hidden_dim"],
        num_heads=mcfg["num_heads"],
        dropout=mcfg["dropout"],
    ).to(device)

    print(f"[ECABSD] Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer
    optimizer = Adam(
        model.parameters(),
        lr=tcfg["learning_rate"],
        weight_decay=tcfg["weight_decay"],
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

    # Loss function — classifier outputs probabilities (sigmoid), so use BCELoss
    criterion = nn.BCELoss(reduction="none")
    pos_weight = 7.74

    # Dataset & loaders — use batch_size=1 with custom collate to handle variable-size graphs
    processed_dir = cfg["data"]["processed_dir"]
    splits_csv = cfg["data"]["splits_csv"]

    if os.path.exists(processed_dir) and os.path.exists(splits_csv):
        train_dataset = BindingSiteDataset(processed_dir, splits_csv, split="train")
        val_dataset = BindingSiteDataset(processed_dir, splits_csv, split="val")

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
        print(f"[ECABSD] Using dummy data for demonstration...")

        # Create minimal dummy data for demonstration
        from models.graph_construction import build_residue_graph
        sample_pdb = "1AY7.pdb"
        if os.path.exists(sample_pdb):
            data_a = build_residue_graph(sample_pdb, "A")
            # Create dummy labels
            data_a.y = torch.zeros(data_a.num_residues)
            data_a.y[:10] = 1.0  # First 10 residues as dummy binding sites

            # Wrap in simple list-based loader
            class DummyBatch:
                def __init__(self, data):
                    self.data = data
                def __iter__(self):
                    yield {
                        "data_a": self.data,
                        "data_b": None,
                        "labels": self.data.y,
                    }
                def __len__(self):
                    return 1

            train_loader = DummyBatch(data_a)
            val_loader = DummyBatch(data_a)
        else:
            print(f"[ECABSD] ERROR: No PDB file found. Cannot train.")
            return

    # Resume from checkpoint
    start_epoch = 0
    best_val_loss = float("inf")
    if resume_from and os.path.exists(resume_from):
        checkpoint = torch.load(resume_from, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint.get("epoch", 0) + 1
        best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        print(f"[ECABSD] Resumed from epoch {start_epoch}")

    # Training loop
    patience_counter = 0
    history = []

    print(f"\n{'='*60}")
    print(f"  ECABSD Training — {tcfg['epochs']} epochs")
    print(f"{'='*60}\n")

    for epoch in range(start_epoch, tcfg["epochs"]):
        t0 = time.time()

        train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion, device, tcfg["gradient_clip"], pos_weight
        )
        val_metrics = validate(model, val_loader, criterion, device, pos_weight)

        elapsed = time.time() - t0

        # LR scheduler step
        if scheduler is not None:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_metrics["loss"])
            else:
                scheduler.step()

        # Logging
        lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch+1:03d}/{tcfg['epochs']} | "
            f"Train Loss: {train_metrics['loss']:.4f} F1: {train_metrics['f1']:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} F1: {val_metrics['f1']:.4f} | "
            f"LR: {lr:.6f} | {elapsed:.1f}s"
        )

        epoch_record = {
            "epoch": epoch + 1,
            "train": train_metrics,
            "val": val_metrics,
            "lr": lr,
            "time": elapsed,
        }
        history.append(epoch_record)

        # Save best model
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            patience_counter = 0
            ckpt_path = os.path.join(pcfg["checkpoints_dir"], "best_model.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_loss": best_val_loss,
                    "config": cfg,
                },
                ckpt_path,
            )
            print(f"  -> Saved best model (val_loss={best_val_loss:.4f})")
        else:
            patience_counter += 1

        # Save periodic checkpoint
        if (epoch + 1) % 10 == 0:
            ckpt_path = os.path.join(pcfg["checkpoints_dir"], f"epoch_{epoch+1}.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_loss": best_val_loss,
                    "config": cfg,
                },
                ckpt_path,
            )

        # Early stopping
        if patience_counter >= tcfg["early_stopping_patience"]:
            print(f"\n[ECABSD] Early stopping at epoch {epoch+1}")
            break

    # Save training history
    history_path = os.path.join(pcfg["logs_dir"], "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  Training complete. Best val loss: {best_val_loss:.4f}")
    print(f"  History saved to: {history_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    run_training()
