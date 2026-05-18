import os
import urllib.request
import tarfile
import argparse

def download_db5(output_dir):
    """
    Downloads the Docking Benchmark 5 (DB5) dataset structures.
    This script downloads the rigid-body docking benchmark structures from the Weng Lab.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # URL for Benchmark 5 structures
    url = "https://zlab.umassmed.edu/benchmark/benchmark5.5_structures.tgz"
    tar_path = os.path.join(output_dir, "benchmark5.5_structures.tgz")
    
    if not os.path.exists(tar_path):
        print(f"Downloading DB5 from {url}...")
        urllib.request.urlretrieve(url, tar_path)
        print("Download complete.")
    else:
        print(f"Archive {tar_path} already exists. Skipping download.")
        
    print(f"Extracting to {output_dir}...")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=output_dir)
        
    print(f"Successfully downloaded and extracted DB5 to {output_dir}.")
    print("Run `python scripts/prepare_dataset.py` pointing to the extracted structures directory to process them.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data/raw/db5")
    args = parser.parse_args()
    download_db5(args.output_dir)
