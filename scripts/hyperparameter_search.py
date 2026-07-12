import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import random
import yaml
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from data.dataset import BindingSiteDataset, collate_fn
from models import ECABSDModel
from train import train_one_epoch, validate, compute_pos_weight, build_criterion, set_seed

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def run_search(num_trials=5, epochs_per_trial=2):
    cfg = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Hyperparameter Search] Running on device: {device}")
    
    # Load dataset
    processed_dir = cfg["data"]["processed_dir"]
    splits_csv = cfg["data"]["splits_csv"]
    
    if not (os.path.exists(processed_dir) and os.path.exists(splits_csv)):
        print("Processed dataset not found. Please run prepare_db5.py first.")
        return

    train_dataset = BindingSiteDataset(processed_dir, splits_csv, split="train")
    val_dataset = BindingSiteDataset(processed_dir, splits_csv, split="val")
    
    train_loader = DataLoader(
        train_dataset, batch_size=cfg["training"]["batch_size"], shuffle=True,
        num_workers=cfg["training"]["num_workers"], collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg["training"]["batch_size"], shuffle=False,
        num_workers=cfg["training"]["num_workers"], collate_fn=collate_fn
    )
    
    pos_weight = compute_pos_weight(train_dataset)
    
    # Search space definition
    learning_rates = [1e-4, 3e-4, 5e-4]
    hidden_dims = [128, 256]
    gat_layers = [4, 6]
    attention_heads = [4, 8]
    dropouts = [0.2, 0.3, 0.4]
    weight_decays = [1e-4, 5e-3, 1e-2]
    
    results = []
    
    for trial in range(num_trials):
        set_seed(42 + trial)
        
        # Sample parameters
        lr = random.choice(learning_rates)
        hd = random.choice(hidden_dims)
        layers = random.choice(gat_layers)
        heads = random.choice(attention_heads)
        dropout = random.choice(dropouts)
        wd = random.choice(weight_decays)
        
        # Verify hidden_dim divisible by heads
        if hd % heads != 0:
            heads = 4 if hd == 128 else 8
            
        print(f"\n--- Trial {trial + 1}/{num_trials} ---")
        print(f"Params: lr={lr}, hidden_dim={hd}, layers={layers}, heads={heads}, dropout={dropout}, wd={wd}")
        
        model = ECABSDModel(
            input_dim=cfg["model"].get("esm_dim", 1280),
            hidden_dim=hd,
            num_heads=heads,
            dropout=dropout,
            edge_dim=cfg["model"].get("edge_feature_dim", 5),
            num_gcn_layers=layers,
        ).to(device)
        
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=wd)
        
        # Loss criterion
        tcfg = cfg["training"].copy()
        criterion = build_criterion(tcfg, pos_weight, device)
        
        best_f1 = -1.0
        best_mcc = -1.0
        
        for epoch in range(epochs_per_trial):
            train_metrics = train_one_epoch(
                model, train_loader, optimizer, criterion, device,
                gradient_clip=1.0, chain_swap_prob=0.5
            )
            val_metrics = validate(model, val_loader, criterion, device)
            
            print(f"  Epoch {epoch+1:02d} | Train F1: {train_metrics['f1']:.4f} | Val F1: {val_metrics['f1']:.4f} MCC: {val_metrics['mcc']:.4f}")
            
            if val_metrics["f1"] > best_f1:
                best_f1 = val_metrics["f1"]
                best_mcc = val_metrics["mcc"]
                
        results.append({
            "trial": trial + 1,
            "params": {
                "learning_rate": lr,
                "hidden_dim": hd,
                "gat_layers": layers,
                "attention_heads": heads,
                "dropout": dropout,
                "weight_decay": wd
            },
            "metrics": {
                "best_val_f1": best_f1,
                "best_val_mcc": best_mcc
            }
        })
        
    # Save search logs
    output_path = os.path.join(cfg["paths"]["results_dir"], "hyperparameter_search_results.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Hyperparameter Search] Completed. Results written to {output_path}")

if __name__ == "__main__":
    run_search(num_trials=3, epochs_per_trial=2)
