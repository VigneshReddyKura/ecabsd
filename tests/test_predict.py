"""
tests/test_predict.py
=====================
Unit tests for graph construction and prediction pipeline.
"""

import pytest
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

SAMPLE_PDB = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'sample', '1AY7.pdb'))
CHECKPOINT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'checkpoints', 'best_model_v3.pt'))

skip_no_checkpoint = pytest.mark.skipif(
    not os.path.exists(CHECKPOINT),
    reason="No model checkpoint found — skipping in CI"
)

skip_no_pdb = pytest.mark.skipif(
    not os.path.exists(SAMPLE_PDB),
    reason="No sample PDB found"
)


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
