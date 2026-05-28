import subprocess
import argparse
import os

def run_command(command):
    print(f"\n[RUNNING]: {' '.join(command)}")
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[ERROR]:\n{result.stderr}")
    else:
        print(result.stdout)
    return result.returncode

def main():
    parser = argparse.ArgumentParser(description="Run Full ECABSD Analysis Pipeline")
    parser.add_argument("--pdb", required=True, help="Path to PDB file")
    parser.add_argument("--chain-a", required=True, help="Target chain ID")
    parser.add_argument("--chain-b", default=None, help="Partner chain ID")
    parser.add_argument("--threshold", type=float, default=0.75, help="Prediction threshold")
    args = parser.parse_args()

    pdb_id = os.path.splitext(os.path.basename(args.pdb))[0]

    # Step 1: Predict
    predict_cmd = ["python", "predict.py", "--pdb", args.pdb, "--chain-a", args.chain_a, "--threshold", str(args.threshold)]
    if args.chain_b:
        predict_cmd += ["--chain-b", args.chain_b]
    
    if run_command(predict_cmd) != 0:
        return

    # Step 2: Heatmap
    heatmap_cmd = ["python", "show_prediction_heatmap.py", "--pdb-id", pdb_id]
    run_command(heatmap_cmd)

    # Step 3: Grad-CAM
    gradcam_cmd = ["python", "gradcam_explain.py", "--pdb", args.pdb, "--chain-a", args.chain_a]
    if args.chain_b:
        gradcam_cmd += ["--chain-b", args.chain_b]
    run_command(gradcam_cmd)

    print(f"\n{'='*60}")
    print(f" Pipeline Complete for {pdb_id}")
    print(f" All results stored in: results/{pdb_id}/")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
