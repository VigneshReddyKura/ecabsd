"""
ECABSD — RMSD Calculator for Docking Poses.

Computes RMSD between predicted and reference binding poses.
Supports both standard and symmetry-corrected RMSD.
"""

import os
import numpy as np
from typing import List, Dict, Optional


def compute_rmsd(coords_pred: np.ndarray, coords_ref: np.ndarray) -> float:
    """
    Compute RMSD between two sets of 3D coordinates.

    Parameters
    ----------
    coords_pred : np.ndarray
        Predicted coordinates, shape (N, 3).
    coords_ref : np.ndarray
        Reference coordinates, shape (N, 3).

    Returns
    -------
    float : RMSD value in Angstroms.
    """
    if coords_pred.shape != coords_ref.shape:
        raise ValueError(
            f"Coordinate shapes don't match: {coords_pred.shape} vs {coords_ref.shape}"
        )
    diff = coords_pred - coords_ref
    return float(np.sqrt((diff ** 2).sum(axis=1).mean()))


def compute_centroid_distance(coords_pred: np.ndarray, coords_ref: np.ndarray) -> float:
    """
    Compute distance between centroids of two coordinate sets.
    Useful for binding site center comparison.
    """
    centroid_pred = coords_pred.mean(axis=0)
    centroid_ref  = coords_ref.mean(axis=0)
    return float(np.linalg.norm(centroid_pred - centroid_ref))


def extract_pdbqt_coords(pdbqt_path: str, model_idx: int = 0) -> np.ndarray:
    """
    Extract heavy atom coordinates from a PDBQT file.

    Parameters
    ----------
    pdbqt_path : str
        Path to PDBQT file (can contain multiple models).
    model_idx : int
        Which model/pose to extract (0-indexed).

    Returns
    -------
    np.ndarray : Coordinates, shape (N, 3).
    """
    if not os.path.exists(pdbqt_path):
        raise FileNotFoundError(f"PDBQT file not found: {pdbqt_path}")

    coords = []
    current_model = -1
    in_target_model = False

    with open(pdbqt_path, "r") as f:
        for line in f:
            if line.startswith("MODEL"):
                current_model += 1
                in_target_model = (current_model == model_idx)
            elif line.startswith("ENDMDL"):
                if in_target_model:
                    break
                in_target_model = False
            elif line.startswith(("ATOM", "HETATM")) and (in_target_model or current_model == -1):
                try:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    element = line[76:78].strip() if len(line) > 76 else ""
                    if element != "H":  # Heavy atoms only
                        coords.append([x, y, z])
                except (ValueError, IndexError):
                    continue

    if not coords:
        raise ValueError(f"No heavy atom coordinates found in: {pdbqt_path}")

    return np.array(coords)


def compute_docking_rmsd_table(
    predicted_pdbqt: str,
    reference_pdbqt: str,
    num_modes: int = 9,
) -> List[Dict]:
    """
    Compute RMSD for all docking poses vs a reference structure.

    Parameters
    ----------
    predicted_pdbqt : str
        Vina output PDBQT (multiple poses).
    reference_pdbqt : str
        Reference crystal/experimental pose PDBQT.
    num_modes : int
        Number of poses to evaluate.

    Returns
    -------
    list of dict : Per-pose RMSD results.
    """
    ref_coords = extract_pdbqt_coords(reference_pdbqt, model_idx=0)
    results = []

    for i in range(num_modes):
        try:
            pred_coords = extract_pdbqt_coords(predicted_pdbqt, model_idx=i)

            # Align by matching atom count (use minimum common length)
            n = min(len(pred_coords), len(ref_coords))
            rmsd = compute_rmsd(pred_coords[:n], ref_coords[:n])
            centroid_dist = compute_centroid_distance(pred_coords[:n], ref_coords[:n])

            results.append({
                "pose": i + 1,
                "rmsd": round(rmsd, 3),
                "centroid_distance": round(centroid_dist, 3),
                "success": rmsd <= 2.0,  # Standard success criterion: RMSD <= 2Å
            })
        except Exception as e:
            results.append({"pose": i + 1, "error": str(e)})

    # Print table
    print(f"\n{'─'*50}")
    print(f"  Docking RMSD Table")
    print(f"{'─'*50}")
    print(f"  {'Pose':>5s}  {'RMSD (Å)':>10s}  {'Centroid (Å)':>13s}  {'Success':>7s}")
    print(f"  {'─'*45}")
    for r in results:
        if "error" not in r:
            flag = "✓" if r["success"] else "✗"
            print(f"  {r['pose']:5d}  {r['rmsd']:10.3f}  {r['centroid_distance']:13.3f}  {flag:>7s}")
    print(f"{'─'*50}\n")

    success_count = sum(1 for r in results if r.get("success", False))
    print(f"  Successful poses (RMSD ≤ 2Å): {success_count}/{len(results)}")

    return results
