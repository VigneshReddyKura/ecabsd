"""
tests/test_predict.py
=====================
Unit tests for predict.py and exports.
"""

import pytest
import os
import sys
import json
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

SAMPLE_PDB = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'sample', '1AY7.pdb'))
CHECKPOINT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'checkpoints', 'best_model_v3.pt'))

skip_no_checkpoint = pytest.mark.skipif(
    not os.path.exists(CHECKPOINT),
    reason="No model checkpoint found — skipping prediction tests in CI"
)

skip_no_pdb = pytest.mark.skipif(
    not os.path.exists(SAMPLE_PDB),
    reason="No sample PDB found"
)


# ---------------------------------------------------------------------------
# Graph construction sanity checks (no checkpoint needed)
# ---------------------------------------------------------------------------

class TestGraphConstruction:

    @skip_no_pdb
    def test_build_graph_chain_a(self):
        from models.graph_construction import build_residue_graph
        graph = build_residue_graph(SAMPLE_PDB, 'A')
        assert graph.x.shape[1] == 33
        assert graph.edge_index.shape[0] == 2
        assert graph.edge_attr.shape[1] == 5

    @skip_no_pdb
    def test_build_graph_chain_b(self):
        from models.graph_construction import build_residue_graph
        graph = build_residue_graph(SAMPLE_PDB, 'B')
        assert graph.num_residues > 0

    @skip_no_pdb
    def test_invalid_chain_raises(self):
        from models.graph_construction import build_residue_graph
        with pytest.raises(Exception):
            build_residue_graph(SAMPLE_PDB, 'Z')


# ---------------------------------------------------------------------------
# Export tests (no checkpoint needed)
# ---------------------------------------------------------------------------

class TestExports:

    @pytest.fixture
    def sample_results(self):
        return {
            "pdb_id": "1AY7",
            "chain": "A",
            "threshold": 0.5,
            "residues": [
                {"residue_id": 1, "residue_name": "ALA", "probability": 0.85, "is_binding": True},
                {"residue_id": 2, "residue_name": "GLY", "probability": 0.23, "is_binding": False},
                {"residue_id": 3, "residue_name": "VAL", "probability": 0.91, "is_binding": True},
            ]
        }

    def test_json_export(self, sample_results, tmp_path):
        from exports.json_export import export_json
        out_file = str(tmp_path / "test_output.json")
        export_json(sample_results, out_file)
        assert os.path.exists(out_file)
        with open(out_file) as f:
            data = json.load(f)
        assert "residues" in data
        assert len(data["residues"]) == 3

    def test_csv_export(self, sample_results, tmp_path):
        from exports.csv_export import export_csv
        out_file = str(tmp_path / "test_output.csv")
        export_csv(sample_results, out_file)
        assert os.path.exists(out_file)
        with open(out_file) as f:
            content = f.read()
        assert "probability" in content.lower() or "residue" in content.lower()

    def test_pymol_export(self, sample_results, tmp_path):
        from exports.pymol_export import export_pymol
        out_file = str(tmp_path / "test_output.pml")
        export_pymol(sample_results, out_file)
        assert os.path.exists(out_file)
        with open(out_file) as f:
            content = f.read()
        assert len(content) > 0


# ---------------------------------------------------------------------------
# Prediction pipeline (requires checkpoint)
# ---------------------------------------------------------------------------

class TestPredictionPipeline:

    @skip_no_checkpoint
    @skip_no_pdb
    def test_run_prediction_returns_dict(self):
        from predict import run_prediction
        results = run_prediction(SAMPLE_PDB, 'A', 'B')
        assert isinstance(results, dict)
        assert "residues" in results
        assert len(results["residues"]) > 0

    @skip_no_checkpoint
    @skip_no_pdb
    def test_prediction_probabilities_valid(self):
        from predict import run_prediction
        results = run_prediction(SAMPLE_PDB, 'A', 'B')
        for r in results["residues"]:
            assert 0.0 <= r["probability"] <= 1.0

    @skip_no_checkpoint
    @skip_no_pdb
    def test_prediction_has_binding_flag(self):
        from predict import run_prediction
        results = run_prediction(SAMPLE_PDB, 'A', 'B')
        for r in results["residues"]:
            assert "is_binding" in r
            assert isinstance(r["is_binding"], bool)
  
