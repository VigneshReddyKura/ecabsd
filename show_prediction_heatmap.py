import os
import json
import numpy as np
import matplotlib.pyplot as plt
import argparse

def main():
    parser = argparse.ArgumentParser(description="Generate Heatmap from ECABSD Prediction")
    parser.add_argument("--pdb-id", required=True, help="PDB ID (folder name in results/)")
    args = parser.parse_args()

    json_path = f"results/{args.pdb_id}/predictions.json"

    if not os.path.exists(json_path):
        print(f"Error: {json_path} not found. Run predict.py first.")
        return

    with open(json_path, "r") as f:
        data = json.load(f)

    residues = data["residues"]
    probs = np.array([r["probability"] for r in residues])

    # make 1-row heatmap
    heatmap = probs.reshape(1, -1)

    plt.figure(figsize=(14, 2))
    plt.imshow(heatmap, aspect="auto", cmap="viridis")
    plt.colorbar(label="Binding Probability")
    plt.title(f"ECABSD Binding Probability Heatmap - {data['pdb_file']} Chain {data['chain_a']}")
    plt.xlabel("Residue Index")
    plt.yticks([])
    plt.tight_layout()

    # Define output directory based on PDB name
    pdb_name = os.path.basename(data["pdb_file"]).replace(".pdb", "")
    out_dir = f"results/{pdb_name}"
    os.makedirs(out_dir, exist_ok=True)

    out_path = f"{out_dir}/heatmap.png"
    plt.savefig(out_path, dpi=300)
    # plt.show()

    print(f"Saved: {out_path}")

if __name__ == "__main__":
    main()