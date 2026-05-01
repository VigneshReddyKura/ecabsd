"""
ECABSD — CSV Export.

Exports per-residue binding site predictions to CSV format.

Usage:
    from exports.csv_export import export_csv
    export_csv("results/predictions_1AY7_A.json", "results/predictions_1AY7_A.csv")
"""

import os
import csv
import json
from typing import Optional


def export_csv(results_path: str, output_path: Optional[str] = None) -> str:
    """
    Export prediction results from JSON to CSV.

    Parameters
    ----------
    results_path : str
        Path to the prediction results JSON file.
    output_path : str, optional
        Output CSV file path. If None, replaces .json extension with .csv.

    Returns
    -------
    str : Path to the saved CSV file.
    """
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"Results file not found: {results_path}")

    with open(results_path, "r") as f:
        results = json.load(f)

    if output_path is None:
        output_path = results_path.replace(".json", ".csv")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    fieldnames = [
        "pdb_file",
        "chain",
        "residue_index",
        "residue_id",
        "resname",
        "probability",
        "is_binding",
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        pdb_file = results.get("pdb_file", "")
        chain = results.get("chain_a", "")

        for res in results.get("residues", []):
            writer.writerow({
                "pdb_file": pdb_file,
                "chain": chain,
                "residue_index": res["index"],
                "residue_id": res["resid"],
                "resname": res["resname"],
                "probability": f"{res['probability']:.6f}",
                "is_binding": int(res["is_binding"]),
            })

    print(f"[Export] CSV saved to: {output_path}")
    print(f"[Export] {len(results.get('residues', []))} residues exported.")
    return output_path


def export_batch_csv(results_dir: str, output_path: Optional[str] = None) -> str:
    """
    Merge multiple prediction JSON files into a single CSV.

    Parameters
    ----------
    results_dir : str
        Directory containing prediction JSON files.
    output_path : str, optional
        Output CSV path.

    Returns
    -------
    str : Path to the merged CSV file.
    """
    import glob

    json_files = sorted(glob.glob(os.path.join(results_dir, "predictions_*.json")))
    if not json_files:
        raise FileNotFoundError(f"No prediction JSON files found in: {results_dir}")

    if output_path is None:
        output_path = os.path.join(results_dir, "all_predictions.csv")

    fieldnames = [
        "pdb_file", "chain", "residue_index", "residue_id",
        "resname", "probability", "is_binding",
    ]

    total_rows = 0
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for jf in json_files:
            with open(jf, "r") as jfile:
                results = json.load(jfile)

            pdb_file = results.get("pdb_file", "")
            chain = results.get("chain_a", "")

            for res in results.get("residues", []):
                writer.writerow({
                    "pdb_file": pdb_file,
                    "chain": chain,
                    "residue_index": res["index"],
                    "residue_id": res["resid"],
                    "resname": res["resname"],
                    "probability": f"{res['probability']:.6f}",
                    "is_binding": int(res["is_binding"]),
                })
                total_rows += 1

    print(f"[Export] Merged CSV saved to: {output_path} ({total_rows} rows)")
    return output_path
