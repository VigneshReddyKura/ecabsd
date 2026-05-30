from huggingface_hub import hf_hub_download
import os

def download():
    os.makedirs("checkpoints", exist_ok=True)
    hf_hub_download(
        repo_id="manigreeva01/ECABSD",
        filename="best_model_v3.pt",
        local_dir="checkpoints"
    )
    print("Done! Weights saved to checkpoints/best_model_v3.pt")

if __name__ == "__main__":
    download()