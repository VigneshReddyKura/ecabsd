"""
ECABSD Training Pipeline - v3 Architecture
Dataset returns dicts with keys: data_a, data_b, labels, pdb_id
Model takes two PyG graphs (chain A and chain B) as input.
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
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from models.ecabsd_model import ECABSDModel
from data.dataset import BindingSiteDataset


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target):
        pred = pred.squeeze(-1)
        target = target.squeeze(-1)
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        pt = torch.exp(-bce)
        focal = self.alpha * (1.0 - pt) ** self.gamma * bce
        return focal.mean()


def load_config(path="config.yaml"):
    with open(path, encoding="utf-8", errors="ignore") as f:
        return yaml.safe_load(f)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collate_fn(batch):
    """Keep samples as a list — each item is a dict with data_a, data_b, labels."""
    return batch


def move_to_device(data, device):
    """Move a PyG Data object to device."""
    return data.to(device)


def train_one_epoch(model, loader, optimizer, criterion, device, threshold=0.3):
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for batch in loader:
        for sample in batch:
            data_a  = sample['data_a'].to(device)
            data_b  = sample['data_b'].to(device)
            labels  = sample['labels'].float().to(device)

            optimizer.zero_grad()
            pred, _ = model(data_a, data_b)
            pred = pred.squeeze(-1)

            loss = criterion(pred, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()

            probs = torch.sigmoid(pred.detach())
            binary_preds = (probs >= threshold).long().cpu().numpy()
            all_preds.extend(binary_preds.tolist())
            all_labels.extend(labels.long().cpu().numpy().tolist())

    avg_loss = total_loss / max(len(loader), 1)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    return avg_loss, f1


@torch.no_grad()
def validate(model, loader, criterion, device, threshold=0.3):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for batch in loader:
        for sample in batch:
            data_a  = sample['data_a'].to(device)
            data_b  = sample['data_b'].to(device)
            labels  = sample['labels'].float().to(device)

            pred, _ = model(data_a, data_b)
            pred = pred.squeeze(-1)

            loss = criterion(pred, labels)
            total_loss += loss.item()

            probs = torch.sigmoid(pred)
            binary_preds = (probs >= threshold).long().cpu().numpy()
            all_preds.extend(binary_preds.tolist())
            all_labels.extend(labels.long().cpu().numpy().tolist())

    avg_loss = total_loss / max(len(loader), 1)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    return avg_loss, f1


def run_training(config):
    set_seed(config.get("seed", 42))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[ECABSD] Using device: {device}")

    dcfg = config.get("data", {})
    tcfg = config.get("training", {})
    mcfg = config.get("model", {})

    train_dataset = BindingSiteDataset(
        processed_dir=dcfg.get("processed_dir", "data/processed"),
        splits_csv=dcfg.get("splits_csv", "data/splits.csv"),
        split="train",
    )
    val_dataset = BindingSiteDataset(
        processed_dir=dcfg.get("processed_dir", "data/processed"),
        splits_csv=dcfg.get("splits_csv", "data/splits.csv"),
        split="val",
    )

    batch_size = tcfg.get("batch_size", 1)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=0, collate_fn=collate_fn)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)

    model = ECABSDModel(
        input_dim=mcfg.get("input_dim", 23),
        hidden_dim=mcfg.get("hidden_dim", 256),
        num_heads=mcfg.get("num_heads", 8),
        dropout=mcfg.get("dropout", 0.3),
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[ECABSD] Model parameters: {total_params:,}")

    epochs    = tcfg.get("epochs", 100)
    patience  = tcfg.get("early_stopping_patience", 20)
    lr        = tcfg.get("learning_rate", 1e-3)
    threshold = tcfg.get("threshold", 0.3)

    criterion = FocalLoss(
        alpha=tcfg.get("focal_alpha", 0.75),
        gamma=tcfg.get("focal_gamma", 2.0),
    )

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=tcfg.get("weight_decay", 1e-4))
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=7, min_lr=1e-5)

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    best_val_loss = float("inf")
    patience_counter = 0
    history = []

    print("=" * 60)
    print(f"  ECABSD Training - {epochs} epochs")
    print("=" * 60)

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        train_loss, train_f1 = train_one_epoch(model, train_loader, optimizer, criterion, device, threshold)
        val_loss,   val_f1   = validate(model, val_loader, criterion, device, threshold)

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0

        print(
            f"Epoch {epoch:03d}/{epochs} | "
            f"Train Loss: {train_loss:.4f} F1: {train_f1:.4f} | "
            f"Val Loss: {val_loss:.4f} F1: {val_f1:.4f} | "
            f"LR: {current_lr:.6f} | {elapsed:.1f}s"
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_f1": train_f1,
            "val_loss": val_loss,
            "val_f1": val_f1,
            "lr": current_lr,
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_f1": val_f1,
                    "config": config,
                },
                "checkpoints/best_model.pt",
            )
            print(f"  -> Saved best model (val_loss={val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"[ECABSD] Early stopping at epoch {epoch}")
                break

    with open("logs/training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print("=" * 60)
    print(f"  Training complete. Best val loss: {best_val_loss:.4f}")
    print(f"  History saved to: logs/training_history.json")
    print("=" * 60)


if __name__ == "__main__":
    cfg = load_config("config.yaml")
    run_training(cfg)
