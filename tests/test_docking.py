"""
tests/test_docking.py
=====================
Unit tests for ECABSD Docking preprocessing utilities.

NOTE: These tests focus on pure-Python logic and do not require
AutoDock Vina, meeko, or MGLTools to be installed. External tools
are mocked where necessary.
"""

import os
import math
import tempfile
import unittest.mock as mock
import pytest
import numpy as np

from docking.docking_input import binding_residues_to_box, write_vina_config
from docking.vina_runner import VinaRunner


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SAMPLE_PDB = """\
ATOM      1  N   ALA A   5      10.000  10.000  10.000  1.00  0.00           N  
ATOM      2  CA  ALA A   5      11.000  10.500  10.500  1.00  0.00           C  
ATOM      3  C   ALA A   5      12.000  11.000  11.000  1.00  0.00           C  
ATOM      4  O   ALA A   5      12.500  11.500  10.500  1.00  0.00           O  
ATOM      5  CB  ALA A   5      10.500  10.000  12.000  1.00  0.00           C  
ATOM      6  N   GLY A   6      13.000  11.000  11.000  1.00  0.00           N  
ATOM      7  CA  GLY A   6      14.000  11.500  11.500  1.00  0.00           C  
ATOM      8  C   GLY A   6      15.000  12.000  12.000  1.00  0.00           C  
ATOM      9  O   GLY A   6      15.500  12.500  11.500  1.00  0.00           O  
END
"""


@pytest.fixture
def sample_pdb_path(tmp_path):
    """Write a minimal PDB to a temp file and return its path."""
    p = tmp_path / "sample.pdb"
    p.write_text(SAMPLE_PDB)
    return str(p)


# ---------------------------------------------------------------------------
# binding_residues_to_box
# ---------------------------------------------------------------------------

class TestBindingResiduesBox:

    def test_returns_center_and_box_size(self, sample_pdb_path):
        """Function must return two 3-tuples: (center, box_size)."""
        binding_residues = [{"resid": 5}]
        center, box_size = binding_residues_to_box(
            binding_residues, sample_pdb_path, chain_id="A", padding=5.0
        )
        assert len(center)   == 3
        assert len(box_size) == 3

    def test_box_size_always_positive(self, sample_pdb_path):
        """Box dimensions must all be strictly positive."""
        binding_residues = [{"resid": 5}, {"resid": 6}]
        _, box_size = binding_residues_to_box(
            binding_residues, sample_pdb_path, chain_id="A", padding=5.0
        )
        assert all(s > 0 for s in box_size)

    def test_padding_increases_box_size(self, sample_pdb_path):
        """A larger padding must produce a larger or equal box."""
        binding_residues = [{"resid": 5}]
        _, box_small = binding_residues_to_box(
            binding_residues, sample_pdb_path, chain_id="A", padding=2.0
        )
        _, box_large = binding_residues_to_box(
            binding_residues, sample_pdb_path, chain_id="A", padding=10.0
        )
        for s, l in zip(box_small, box_large):
            assert l >= s

    def test_empty_residues_raises(self, sample_pdb_path):
        """Providing a resid that doesn't exist must raise ValueError."""
        binding_residues = [{"resid": 999}]  # not in SAMPLE_PDB
        with pytest.raises(ValueError, match="No coordinates found"):
            binding_residues_to_box(
                binding_residues, sample_pdb_path, chain_id="A", padding=5.0
            )


# ---------------------------------------------------------------------------
# write_vina_config
# ---------------------------------------------------------------------------

class TestWriteVinaConfig:

    def test_config_file_is_written(self, tmp_path):
        config_path = str(tmp_path / "vina.conf")
        write_vina_config(
            receptor_pdbqt="receptor.pdbqt",
            ligand_pdbqt="ligand.pdbqt",
            center=(1.0, 2.0, 3.0),
            box_size=(20.0, 20.0, 20.0),
            output_path=config_path,
        )
        assert os.path.exists(config_path)

    def test_config_contains_correct_values(self, tmp_path):
        config_path = str(tmp_path / "vina.conf")
        write_vina_config(
            receptor_pdbqt="my_receptor.pdbqt",
            ligand_pdbqt="my_ligand.pdbqt",
            center=(5.5, 6.6, 7.7),
            box_size=(18.0, 19.0, 21.0),
            output_path=config_path,
            exhaustiveness=16,
            num_modes=5,
        )
        content = open(config_path).read()
        assert "my_receptor.pdbqt" in content
        assert "my_ligand.pdbqt"   in content
        assert "5.500"             in content
        assert "exhaustiveness = 16" in content
        assert "num_modes = 5"      in content


# ---------------------------------------------------------------------------
# VinaRunner
# ---------------------------------------------------------------------------

class TestVinaRunner:

    def test_init_stores_executable(self):
        runner = VinaRunner(vina_executable="vina")
        assert runner.vina_executable == "vina"

    def test_run_docking_callable(self):
        runner = VinaRunner(vina_executable="vina")
        assert callable(runner.run_docking)

    def test_run_docking_raises_on_missing_executable(self, tmp_path):
        """If the Vina binary doesn't exist, run_docking must raise an informative error."""
        runner = VinaRunner(vina_executable="nonexistent_vina_binary_xyz")
        config_path = str(tmp_path / "vina.conf")
        with open(config_path, "w") as f:
            f.write("receptor = r.pdbqt\n")

        with pytest.raises(Exception):
            runner.run_docking(
                config_path=config_path,
                output_dir=str(tmp_path),
            )
