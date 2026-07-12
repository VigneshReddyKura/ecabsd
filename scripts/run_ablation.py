import os
import csv
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

# Wrapper to simulate GNN-only model (no cross-attention partner chain)
class GNNOnlyWrapper(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        
    def forward(self, data_a, data_b=None):
        # Pass data_a as both A and B to run self-attention only (GNN only)
        return self.base_model(data_a, data_b=None)

def train_ablation_variant(variant_name, use_esm=True, use_cross_attn=True, epochs=2):
    cfg = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(42)
    
    # Load dataset
    processed_dir = cfg["data"]["processed_dir"]
    splits_csv = cfg["data"]["splits_csv"]
    
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
    
    input_dim = 1280 if use_esm else 33
    
    # If structural-only, we mock the first 33 dimensions of the features
    class FeatureFilterDatasetWrapper(torch.utils.data.Dataset):
        def __init__(self, dataset, feature_dim):
            self.dataset = dataset
            self.feature_dim = feature_dim
        def __len__(self):
            return len(self.dataset)
        def __getitem__(self, idx):
            sample = self.dataset[idx]
            # Copy sample
            data_a = sample["data_a"].clone()
            data_b = sample["data_b"].clone()
            if self.feature_dim == 33:
                # Slice or pad/truncate to 33-dim
                data_a.x = data_a.x[:, :33] if data_a.x.shape[1] >= 33 else torch.cat([data_a.x, torch.zeros(data_a.x.shape[0], 33 - data_a.x.shape[1])], dim=1)
                data_b.x = data_b.x[:, :33] if data_b.x.shape[1] >= 33 else torch.cat([data_b.x, torch.zeros(data_b.x.shape[0], 33 - data_b.x.shape[1])], dim=1)
            return {"data_a": data_a, "data_b": data_b, "labels": sample["labels"]}

    if not use_esm:
        train_dataset = FeatureFilterDatasetWrapper(train_dataset, 33)
        val_dataset = FeatureFilterDatasetWrapper(val_dataset, 33)
        
    base_model = ECABSDModel(
        input_dim=input_dim,
        hidden_dim=cfg["model"]["hidden_dim"],
        num_heads=cfg["model"]["num_heads"],
        dropout=cfg["model"]["dropout"],
        edge_dim=cfg["model"].get("edge_feature_dim", 5),
        num_gcn_layers=cfg["model"].get("num_gcn_layers", 6),
    )
    
    if not use_cross_attn:
        model = GNNOnlyWrapper(base_model).to(device)
    else:
        model = base_model.to(device)
        
    optimizer = AdamW(model.parameters(), lr=cfg["training"]["learning_rate"])
    tcfg = cfg["training"].copy()
    criterion = build_criterion(tcfg, pos_weight, device)
    
    best_f1 = -1.0
    best_mcc = -1.0
    
    print(f"\n[Ablation] Training variant: {variant_name} (ESM={use_esm}, Cross-Attention={use_cross_attn})")
    
    for epoch in range(epochs):
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            gradient_clip=1.0, chain_swap_prob=0.5 if use_cross_attn else 0.0
        )
        val_metrics = validate(model, val_loader, criterion, device)
        print(f"  Epoch {epoch+1:02d} | Train F1: {train_metrics['f1']:.4f} | Val F1: {val_metrics['f1']:.4f}")
        
        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            best_mcc = val_metrics["mcc"]
            
    return best_f1, best_mcc

def run_ablation_study():
    results = []
    
    # 1. GNN Only
    f1, mcc = train_ablation_variant("GNN Only (No Partner Chain)", use_esm=True, use_cross_attn=False, epochs=2)
    results.append({"Variant": "GNN Only (No Partner Chain)", "Val F1": f1, "Val MCC": mcc})
    
    # 2. GNN + Cross Attention (Structural Only)
    f1, mcc = train_ablation_variant("GNN + Cross Attention (Structural Features)", use_esm=False, use_cross_attn=True, epochs=2)
    results.append({"Variant": "GNN + Cross Attention (Structural Features)", "Val F1": f1, "Val MCC": mcc})
    
    # 3. Full V3 Model
    f1, mcc = train_ablation_variant("Full V3 Model (ESM-2 + Cross Attention)", use_esm=True, use_cross_attn=True, epochs=2)
    results.append({"Variant": "Full V3 Model (ESM-2 + Cross Attention)", "Val F1": f1, "Val MCC": mcc})
    
    output_path = "results/ablation_study.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Variant", "Val F1", "Val MCC"])
        writer.writeheader()
        writer.writerows(results)
        
    print(f"\n[Ablation Study] Complete. Comparison written to {output_path}")

if __name__ == "__main__":
    run_ablation_study()
