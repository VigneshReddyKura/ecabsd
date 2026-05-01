"""
ECABSD — AutoDock Vina Runner.

Wraps AutoDock Vina CLI to perform docking using predicted binding sites
as the search box definition.

Requirements:
    - AutoDock Vina installed and on PATH (or path specified in config)
    - Receptor and ligand in PDBQT format

Usage:
    from docking.vina_runner import VinaRunner
    runner = VinaRunner(vina_executable="vina")
    result = runner.dock(receptor_pdbqt, ligand_pdbqt, center, box_size)
"""

import os
import subprocess
import re
from typing import Optional, Tuple, List, Dict


class VinaRunner:
    """
    AutoDock Vina wrapper for ECABSD binding site docking.

    Parameters
    ----------
    vina_executable : str
        Path to or name of the Vina executable.
    exhaustiveness : int
        Vina exhaustiveness parameter (higher = more thorough, slower).
    num_modes : int
        Maximum number of docking modes to generate.
    energy_range : float
        Maximum energy difference between best and worst mode (kcal/mol).
    """

    def __init__(
        self,
        vina_executable: str = "vina",
        exhaustiveness: int = 8,
        num_modes: int = 9,
        energy_range: float = 3.0,
    ):
        self.vina_executable = vina_executable
        self.exhaustiveness = exhaustiveness
        self.num_modes = num_modes
        self.energy_range = energy_range
        self._check_vina()

    def _check_vina(self):
        """Check if Vina is accessible."""
        try:
            result = subprocess.run(
                [self.vina_executable, "--version"],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0 or "AutoDock Vina" in (result.stdout + result.stderr):
                print(f"[Vina] Found: {self.vina_executable}")
            else:
                print(f"[Vina] WARNING: Vina may not be installed correctly.")
        except (FileNotFoundError, subprocess.TimeoutExpired):
            print(f"[Vina] WARNING: '{self.vina_executable}' not found on PATH.")
            print(f"[Vina] Install via: conda install -c conda-forge autodock-vina")

    def dock(
        self,
        receptor_pdbqt: str,
        ligand_pdbqt: str,
        center: Tuple[float, float, float],
        box_size: Tuple[float, float, float],
        output_pdbqt: Optional[str] = None,
        log_file: Optional[str] = None,
        cpu: int = 1,
    ) -> Dict:
        """
        Run Vina docking.

        Parameters
        ----------
        receptor_pdbqt : str
            Path to receptor PDBQT file.
        ligand_pdbqt : str
            Path to ligand PDBQT file.
        center : tuple
            (x, y, z) center of the docking box in Angstroms.
        box_size : tuple
            (size_x, size_y, size_z) dimensions of the docking box.
        output_pdbqt : str, optional
            Output PDBQT path. Defaults to ligand path with '_out' suffix.
        log_file : str, optional
            Path to save Vina log output.
        cpu : int
            Number of CPUs to use.

        Returns
        -------
        dict : Results containing output path, log, and parsed scores.
        """
        if not os.path.exists(receptor_pdbqt):
            raise FileNotFoundError(f"Receptor not found: {receptor_pdbqt}")
        if not os.path.exists(ligand_pdbqt):
            raise FileNotFoundError(f"Ligand not found: {ligand_pdbqt}")

        if output_pdbqt is None:
            base = os.path.splitext(ligand_pdbqt)[0]
            output_pdbqt = f"{base}_out.pdbqt"

        if log_file is None:
            base = os.path.splitext(ligand_pdbqt)[0]
            log_file = f"{base}_vina.log"

        cmd = [
            self.vina_executable,
            "--receptor", receptor_pdbqt,
            "--ligand", ligand_pdbqt,
            "--center_x", str(round(center[0], 3)),
            "--center_y", str(round(center[1], 3)),
            "--center_z", str(round(center[2], 3)),
            "--size_x", str(round(box_size[0], 3)),
            "--size_y", str(round(box_size[1], 3)),
            "--size_z", str(round(box_size[2], 3)),
            "--out", output_pdbqt,
            "--log", log_file,
            "--exhaustiveness", str(self.exhaustiveness),
            "--num_modes", str(self.num_modes),
            "--energy_range", str(self.energy_range),
            "--cpu", str(cpu),
        ]

        print(f"[Vina] Running docking...")
        print(f"[Vina] Center: {center}")
        print(f"[Vina] Box:    {box_size}")

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=600
            )
            stdout = result.stdout + result.stderr

            scores = self._parse_scores(stdout)

            if result.returncode == 0:
                print(f"[Vina] Docking complete.")
                if scores:
                    print(f"[Vina] Best score: {scores[0]['affinity']} kcal/mol")
            else:
                print(f"[Vina] ERROR: {stdout}")

            return {
                "output_pdbqt": output_pdbqt,
                "log_file": log_file,
                "scores": scores,
                "returncode": result.returncode,
                "stdout": stdout,
            }

        except subprocess.TimeoutExpired:
            print("[Vina] ERROR: Docking timed out after 600 seconds.")
            return {"error": "timeout", "scores": []}
        except FileNotFoundError:
            print(f"[Vina] ERROR: Vina executable not found: {self.vina_executable}")
            return {"error": "vina_not_found", "scores": []}

    def _parse_scores(self, vina_output: str) -> List[Dict]:
        """Parse Vina output scores table."""
        scores = []
        lines = vina_output.split("\n")
        in_table = False
        for line in lines:
            if "mode |" in line.lower() or "-----" in line:
                in_table = True
                continue
            if in_table:
                parts = line.strip().split()
                if len(parts) >= 4:
                    try:
                        mode = int(parts[0])
                        affinity = float(parts[1])
                        rmsd_lb = float(parts[2])
                        rmsd_ub = float(parts[3])
                        scores.append({
                            "mode": mode,
                            "affinity": affinity,
                            "rmsd_lb": rmsd_lb,
                            "rmsd_ub": rmsd_ub,
                        })
                    except ValueError:
                        continue
        return scores
