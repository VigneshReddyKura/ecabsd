import torch
import os

ckpt_path = r"c:\Users\amani\Downloads\ecabsd\checkpoints\best_model_v3.pt"
stripped_path = r"c:\Users\amani\Downloads\ecabsd\checkpoints\best_model_v3_stripped.pt"

if os.path.exists(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    stripped_ckpt = {
        "model_state_dict": ckpt["model_state_dict"],
        "best_threshold": ckpt.get("best_threshold", 0.52),
    }
    
    # Save stripped checkpoint
    torch.save(stripped_ckpt, stripped_path)
    
    original_size = os.path.getsize(ckpt_path) / (1024 * 1024)
    stripped_size = os.path.getsize(stripped_path) / (1024 * 1024)
    
    print(f"Original checkpoint size: {original_size:.2f} MB")
    print(f"Stripped checkpoint size: {stripped_size:.2f} MB")
else:
    print("Checkpoint not found.")
