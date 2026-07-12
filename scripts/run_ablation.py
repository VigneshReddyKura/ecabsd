import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import csv
import yaml
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from data.dataset import BindingSiteDataset, collate_fn
from models import ECABSDModel
from train import train_one_epoch, validate, compute_pos_weight, build_criterion, set_seed

_esm_models_cache = {}

def get_esm_embeddings_by_model(sequence: str, model_name: str) -> torch.Tensor:
    """Load a specific ESM model and extract embeddings."""
    global _esm_models_cache
    if model_name not in _esm_models_cache:
        import sys
        # Prevent Windows DLL crash and Kaggle import crash during torchvision loading
        sys.modules['torchvision'] = None
        sys.modules['torchvision.transforms'] = None
        from transformers import EsmModel, AutoTokenizer
        print(f"[Ablation] Loading model weights: {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = EsmModel.from_pretrained(model_name).to(device)
        model.eval()
        _esm_models_cache[model_name] = (model, tokenizer)

    model, tokenizer = _esm_models_cache[model_name]
    device = model.device
    inputs = tokenizer(sequence, return_tensors="pt", add_special_tokens=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Extract last hidden state, ignoring <cls> and <eos>
    embeddings = outputs.last_hidden_state[0, 1:-1, :].cpu()
    return embeddings

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

# Wrapper to simulate GNN-only model (no cross-attention partner chain)
class GNNOnlyWrapper(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        
    def forward(self, data_a, data_b=None):
        return self.base_model(data_a, data_b=None)

class AblationDatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset, variant_mode):
        """
        variant_mode can be:
          - "GNN_ONLY": uses ESM-2 650M embeddings
          - "STRUCTURAL_ONLY": uses 33-dim structural features
          - "ESM_8M": extracts 320-dim ESM-2 8M embeddings
          - "ESM_650M": uses 1280-dim ESM-2 650M embeddings (default)
        """
        self.dataset = dataset
        self.variant_mode = variant_mode
        self.esm_cache = {}
        
        self.STANDARD_AA = [
            'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY',
            'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER',
            'THR', 'TRP', 'TYR', 'VAL'
        ]
        self.three_to_one = {aa: aa[0] if aa != 'ARG' else 'R' for aa in self.STANDARD_AA}
        # Correct custom mappings where first letter is not unique
        self.three_to_one.update({
            'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLN': 'Q',
            'GLU': 'E', 'GLY': 'G', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K',
            'PHE': 'F', 'PRO': 'P', 'SER': 'S', 'THR': 'T', 'TRP': 'W',
            'TYR': 'Y', 'VAL': 'V'
        })

    def __len__(self):
        return len(self.dataset)

    def _recover_sequence_from_features(self, x_structural):
        """Recover sequence from the 20-dim one-hot features."""
        indices = x_structural[:, :20].argmax(dim=1).tolist()
        return "".join([self.three_to_one.get(self.STANDARD_AA[idx], 'X') for idx in indices])

    def _process_graph(self, data):
        if data is None:
            return None
        data = data.clone()
        
        # 1. Structural Only Mode
        if self.variant_mode == "STRUCTURAL_ONLY":
            # If x_structural is saved in data, use it; otherwise slice/truncate data.x
            data.x = getattr(data, "x_structural", data.x[:, :33])
            
        # 2. ESM 8M Mode (Extracts 320-dim features)
        elif self.variant_mode == "ESM_8M":
            # Find sequence
            seq = getattr(data, "sequence", None)
            if seq is None:
                # Recover sequence from structural features (either data.x_structural or data.x)
                x_struct = getattr(data, "x_structural", data.x if data.x.shape[1] == 33 else None)
                if x_struct is not None:
                    seq = self._recover_sequence_from_features(x_struct)
                else:
                    seq = "X" * data.x.shape[0]
            
            if seq not in self.esm_cache:
                self.esm_cache[seq] = get_esm_embeddings_by_model(seq, "facebook/esm2_t6_8M_UR50D")
            data.x = self.esm_cache[seq]
            
        # 3. ESM 650M Mode (1280-dim) or GNN_ONLY
        else:
            # Assumes data.x contains 1280-dim embeddings
            pass
            
        return data

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        data_a = self._process_graph(sample["data_a"])
        data_b = self._process_graph(sample["data_b"])
        return {
            "data_a": data_a,
            "data_b": data_b,
            "labels": sample["labels"],
            "pdb_id": sample["pdb_id"]
        }

def train_ablation_variant(variant_mode, epochs=2):
    cfg = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(42)
    
    # Load dataset
    processed_dir = cfg["data"]["processed_dir"]
    splits_csv = cfg["data"]["splits_csv"]
    
    train_dataset = BindingSiteDataset(processed_dir, splits_csv, split="train")
    val_dataset = BindingSiteDataset(processed_dir, splits_csv, split="val")
    
    # Wrap datasets to apply custom representations
    train_wrapped = AblationDatasetWrapper(train_dataset, variant_mode)
    val_wrapped = AblationDatasetWrapper(val_dataset, variant_mode)
    
    train_loader = DataLoader(
        train_wrapped, batch_size=cfg["training"]["batch_size"], shuffle=True,
        num_workers=cfg["training"]["num_workers"], collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_wrapped, batch_size=cfg["training"]["batch_size"], shuffle=False,
        num_workers=cfg["training"]["num_workers"], collate_fn=collate_fn
    )
    
    pos_weight = compute_pos_weight(train_dataset)
    
    # Determine input dimensions
    if variant_mode == "STRUCTURAL_ONLY":
        input_dim = 33
    elif variant_mode == "ESM_8M":
        input_dim = 320
    else:
        input_dim = 1280
        
    base_model = ECABSDModel(
        input_dim=input_dim,
        hidden_dim=cfg["model"]["hidden_dim"],
        num_heads=cfg["model"]["num_heads"],
        dropout=cfg["model"]["dropout"],
        edge_dim=cfg["model"].get("edge_feature_dim", 5),
        num_gcn_layers=cfg["model"].get("num_gcn_layers", 6),
    )
    
    use_cross_attn = (variant_mode != "GNN_ONLY")
    
    if not use_cross_attn:
        model = GNNOnlyWrapper(base_model).to(device)
    else:
        model = base_model.to(device)
        
    optimizer = AdamW(model.parameters(), lr=cfg["training"]["learning_rate"])
    tcfg = cfg["training"].copy()
    criterion = build_criterion(tcfg, pos_weight, device)
    
    best_f1 = -1.0
    best_mcc = -1.0
    
    print(f"\n[Ablation] Training variant: {variant_mode} (input_dim={input_dim}, Cross-Attention={use_cross_attn})")
    
    for epoch in range(epochs):
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            gradient_clip=1.0, chain_swap_prob=0.5 if use_cross_attn else 0.0
        )
        val_metrics = validate(model, val_loader, criterion, device)
        print(f"  Epoch {epoch+1:02d} | Train F1: {train_metrics['f1']:.4f} | Val F1: {val_metrics['f1']:.4f} MCC: {val_metrics['mcc']:.4f}")
        
        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            best_mcc = val_metrics["mcc"]
            
    return best_f1, best_mcc

def run_ablation_study():
    results = []
    
    # 1. GNN Only (ESM2-650M)
    f1, mcc = train_ablation_variant("GNN_ONLY", epochs=2)
    results.append({"Variant": "GNN Only (No Partner Chain - ESM2-650M)", "Val F1": f1, "Val MCC": mcc})
    
    # 2. GNN + Cross Attention (Structural Only)
    f1, mcc = train_ablation_variant("STRUCTURAL_ONLY", epochs=2)
    results.append({"Variant": "GNN + Cross Attention (Structural Features Only)", "Val F1": f1, "Val MCC": mcc})
    
    # 3. GNN + Cross Attention (ESM2-8M)
    f1, mcc = train_ablation_variant("ESM_8M", epochs=2)
    results.append({"Variant": "GNN + Cross Attention (ESM2-8M Embeddings)", "Val F1": f1, "Val MCC": mcc})
    
    # 4. Full V3 Model (ESM2-650M)
    f1, mcc = train_ablation_variant("ESM_650M", epochs=2)
    results.append({"Variant": "Full V3 Model (ESM-2 650M + Cross Attention)", "Val F1": f1, "Val MCC": mcc})
    
    output_path = "results/ablation_study.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Variant", "Val F1", "Val MCC"])
        writer.writeheader()
        writer.writerows(results)
        
    print(f"\n[Ablation Study] Complete. Comparison written to {output_path}")

if __name__ == "__main__":
    run_ablation_study()
