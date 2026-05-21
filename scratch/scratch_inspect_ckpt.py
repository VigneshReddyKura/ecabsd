import torch
import os

ckpt_path = r"c:\Users\amani\Downloads\ecabsd\checkpoints\best_model_v3.pt"
if os.path.exists(ckpt_path):
    print("Checkpoint exists.")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    print("Keys in checkpoint:", ckpt.keys())
    for k, v in ckpt.items():
        if isinstance(v, dict):
            print(f"Key '{k}' is a dict with {len(v)} elements.")
        elif torch.is_tensor(v):
            print(f"Key '{k}' is a tensor of shape {v.shape}.")
        else:
            print(f"Key '{k}': {type(v)} = {v}")
else:
    print("Checkpoint does not exist.")
