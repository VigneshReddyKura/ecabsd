"""
ECABSD — Docking Input Preparation.

Converts predicted binding residues to:
1. A Vina docking box (center + dimensions)
2. Receptor PDBQT file (via MGLTools/meeko)
3. Vina config file
"""

import os
import subprocess
import numpy as np
from typing import List, Tuple, Dict, Optional


def binding_residues_to_box(
    binding_residues: List[Dict],
    pdb_path: str,
    chain_id: str,
    padding: float = 5.0,
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    """
    Compute a docking box from predicted binding site residues.

    Parameters
    ----------
    binding_residues : list of dict
        List of binding residue dicts with 'resid' key.
    pdb_path : str
        Path to PDB file (to get CA coordinates).
    chain_id : str
        Chain ID.
    padding : float
        Padding in Angstroms around the binding site.

    Returns
    -------
    center : tuple
        (x, y, z) center of the docking box.
    box_size : tuple
        (size_x, size_y, size_z) dimensions of the box.
    """
    from Bio.PDB import PDBParser
    from Bio.PDB.Polypeptide import is_aa

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_path)
    model = structure[0]
    chain = model[chain_id]

    binding_resids = {r["resid"] for r in binding_residues}
    coords = []

    for residue in chain:
        if not is_aa(residue, standard=True):
            continue
        resid = residue.get_id()[1]
        if resid in binding_resids:
            try:
                ca = residue["CA"].get_vector().get_array()
                # Also add all heavy atoms for better box coverage
                for atom in residue:
                    coords.append(atom.get_vector().get_array())
            except KeyError:
                pass

    if not coords:
        raise ValueError("No coordinates found for binding residues.")

    coords = np.array(coords)
    min_coords = coords.min(axis=0) - padding
    max_coords = coords.max(axis=0) + padding

    center = tuple(((min_coords + max_coords) / 2).tolist())
    box_size = tuple((max_coords - min_coords).tolist())

    print(f"[DockingInput] Box center: ({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f})")
    print(f"[DockingInput] Box size:   ({box_size[0]:.2f}, {box_size[1]:.2f}, {box_size[2]:.2f})")

    return center, box_size


def prepare_receptor_pdbqt(
    pdb_path: str,
    output_path: Optional[str] = None,
    chain_id: Optional[str] = None,
) -> str:
    """
    Convert receptor PDB to PDBQT using meeko or MGLTools.

    Parameters
    ----------
    pdb_path : str
        Path to receptor PDB file.
    output_path : str, optional
        Output PDBQT path.
    chain_id : str, optional
        If specified, only keep this chain.

    Returns
    -------
    str : Path to receptor PDBQT file.
    """
    if output_path is None:
        output_path = pdb_path.replace(".pdb", "_receptor.pdbqt")

    # Try meeko first (modern, pip-installable)
    try:
        cmd = ["mk_prepare_receptor.py", "-i", pdb_path, "-o", output_path]
        if chain_id:
            cmd += ["--keep_nonpolar_hydrogens"]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode == 0 and os.path.exists(output_path):
            print(f"[DockingInput] Receptor PDBQT saved: {output_path} (via meeko)")
            return output_path
    except FileNotFoundError:
        pass

    # Try prepare_receptor4.py (MGLTools)
    try:
        cmd = ["prepare_receptor4.py", "-r", pdb_path, "-o", output_path, "-A", "hydrogens"]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode == 0 and os.path.exists(output_path):
            print(f"[DockingInput] Receptor PDBQT saved: {output_path} (via MGLTools)")
            return output_path
    except FileNotFoundError:
        pass

    raise RuntimeError(
        "Could not prepare receptor PDBQT. Install meeko: pip install meeko\n"
        "Or MGLTools: https://ccsb.scripps.edu/mgltools/downloads/"
    )


def write_vina_config(
    receptor_pdbqt: str,
    ligand_pdbqt: str,
    center: Tuple[float, float, float],
    box_size: Tuple[float, float, float],
    output_path: str,
    exhaustiveness: int = 8,
    num_modes: int = 9,
) -> str:
    """
    Write an AutoDock Vina configuration file.

    Returns
    -------
    str : Path to the config file.
    """
    config = f"""receptor = {receptor_pdbqt}
ligand = {ligand_pdbqt}

center_x = {center[0]:.3f}
center_y = {center[1]:.3f}
center_z = {center[2]:.3f}

size_x = {box_size[0]:.3f}
size_y = {box_size[1]:.3f}
size_z = {box_size[2]:.3f}

exhaustiveness = {exhaustiveness}
num_modes = {num_modes}
energy_range = 3
"""
    with open(output_path, "w") as f:
        f.write(config)

    print(f"[DockingInput] Vina config written: {output_path}")
    return output_path
