import os
import sys
import gc

try:
    import psutil
    def get_mem():
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024) # MB
except ImportError:
    def get_mem():
        return 0.0

print(f"Memory before imports: {get_mem():.2f} MB")

import torch
print(f"Memory after importing torch: {get_mem():.2f} MB")

from models.ecabsd_model import ECABSDModel
from models.graph_construction import build_residue_graph
print(f"Memory after other imports: {get_mem():.2f} MB")

# Load V3 model and stripped checkpoint
device = torch.device("cpu")
model = ECABSDModel(
    input_dim=33,
    hidden_dim=256,
    num_heads=4,
    dropout=0.0,
    edge_dim=5,
    num_gcn_layers=6,
).to(device)
print(f"Memory after model creation: {get_mem():.2f} MB")

ckpt_path = r"checkpoints/best_model_v3.pt"
ckpt = torch.load(ckpt_path, map_location=device)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()
print(f"Memory after loading stripped checkpoint (which replaces original): {get_mem():.2f} MB")

del ckpt
gc.collect()
print(f"Memory after deleting ckpt & gc.collect(): {get_mem():.2f} MB")

# Run prediction
print("\n--- Testing Prediction Graph Building & Inference ---")
pdb_path = r"data/raw/pdbs/1AY7.pdb"
chain_a = "A"
chain_b = "B"

print(f"Memory before graph build: {get_mem():.2f} MB")
data_a = build_residue_graph(pdb_path, chain_a).to(device)
data_b = build_residue_graph(pdb_path, chain_b).to(device)
print(f"Memory after graph build: {get_mem():.2f} MB")

with torch.no_grad():
    logits, attn = model(data_a, data_b)
    probs = torch.sigmoid(logits).squeeze(-1)
print(f"Memory after inference: {get_mem():.2f} MB")
print("Sample prediction probabilities (first 10):", probs[:10].tolist())

# Clean up memory
del logits, attn, probs, data_a, data_b
gc.collect()
print(f"Memory after prediction cleanup & gc: {get_mem():.2f} MB")
