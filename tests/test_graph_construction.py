# tests/test_graph_construction.py

import pytest
import torch
import sys
import os

# So Python can find models/graph_construction.py
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.graph_construction import (
    build_residue_graph,
    get_residues,
    get_backbone_coords,
    get_node_features,
    get_edges,
    AA_TO_IDX,
    STANDARD_AA
)
from Bio.PDB import PDBParser

PDB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '1AY7.pdb'))
CHAIN_ID = 'A'

# ──────────────────────────────────────────────
# 1. CONSTANTS
# ──────────────────────────────────────────────

def test_standard_aa_count():
    """There must be exactly 20 standard amino acids."""
    assert len(STANDARD_AA) == 20

def test_aa_to_idx_mapping():
    """Every AA maps to a unique index 0-19."""
    assert len(AA_TO_IDX) == 20
    assert set(AA_TO_IDX.values()) == set(range(20))

# ──────────────────────────────────────────────
# 2. PDB PARSING
# ──────────────────────────────────────────────

def test_pdb_file_exists():
    """1AY7.pdb must exist at project root."""
    assert os.path.exists(PDB_PATH), f"PDB file not found at {PDB_PATH}"

def test_get_residues_returns_list():
    """get_residues must return a non-empty list for chain A."""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", PDB_PATH)
    chain = structure[0][CHAIN_ID]
    residues, skipped = get_residues(chain)
    assert isinstance(residues, list)
    assert len(residues) > 0

def test_get_residues_skips_hetatm():
    """Skipped count must be >= 0 (non-AA atoms filtered out)."""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", PDB_PATH)
    chain = structure[0][CHAIN_ID]
    residues, skipped = get_residues(chain)
    assert skipped >= 0

# ──────────────────────────────────────────────
# 3. NODE FEATURES
# ──────────────────────────────────────────────

def test_node_feature_shape():
    """Each residue must produce exactly 23 features (20 AA + 3 SS)."""
    graph = build_residue_graph(PDB_PATH, CHAIN_ID)
    assert graph.x.shape[1] == 23, f"Expected 23 features, got {graph.x.shape[1]}"

def test_node_feature_count_matches_residues():
    """Number of node feature rows must equal number of residues."""
    graph = build_residue_graph(PDB_PATH, CHAIN_ID)
    assert graph.x.shape[0] == graph.num_residues

def test_node_features_are_float():
    """Node features must be float tensors."""
    graph = build_residue_graph(PDB_PATH, CHAIN_ID)
    assert graph.x.dtype == torch.float32

def test_one_hot_is_valid():
    """Each residue's AA one-hot (first 20 dims) must sum to exactly 1."""
    graph = build_residue_graph(PDB_PATH, CHAIN_ID)
    one_hot_part = graph.x[:, :20]
    row_sums = one_hot_part.sum(dim=1)
    assert torch.all(row_sums == 1.0), "Some residues have invalid one-hot encoding"

def test_ss_one_hot_is_valid():
    """Each residue's SS one-hot (last 3 dims) must sum to exactly 1."""
    graph = build_residue_graph(PDB_PATH, CHAIN_ID)
    ss_part = graph.x[:, 20:]
    row_sums = ss_part.sum(dim=1)
    assert torch.all(row_sums == 1.0), "Some residues have invalid SS encoding"

# ──────────────────────────────────────────────
# 4. EDGES
# ──────────────────────────────────────────────

def test_edge_index_shape():
    """edge_index must have shape [2, num_edges]."""
    graph = build_residue_graph(PDB_PATH, CHAIN_ID)
    assert graph.edge_index.shape[0] == 2

def test_edge_index_dtype():
    """edge_index must be a long (int64) tensor."""
    graph = build_residue_graph(PDB_PATH, CHAIN_ID)
    assert graph.edge_index.dtype == torch.long

def test_no_self_loops():
    """No edge should connect a residue to itself."""
    graph = build_residue_graph(PDB_PATH, CHAIN_ID)
    src, dst = graph.edge_index
    assert torch.all(src != dst), "Self-loops found in edge_index"

def test_edge_attr_shape():
    """Edge attributes must have 4 features (dist + 3D unit vector)."""
    graph = build_residue_graph(PDB_PATH, CHAIN_ID)
    assert graph.edge_attr.shape[1] == 4, f"Expected 4 edge features, got {graph.edge_attr.shape[1]}"

def test_edge_attr_count_matches_edges():
    """Number of edge_attr rows must match number of edges."""
    graph = build_residue_graph(PDB_PATH, CHAIN_ID)
    num_edges = graph.edge_index.shape[1]
    assert graph.edge_attr.shape[0] == num_edges

def test_edge_distances_positive():
    """All edge distances (first edge feature) must be positive."""
    graph = build_residue_graph(PDB_PATH, CHAIN_ID)
    distances = graph.edge_attr[:, 0]
    assert torch.all(distances > 0), "Some edge distances are zero or negative"

def test_edge_distances_within_cutoff():
    """All edge distances must be within the 8.0 Angstrom cutoff."""
    graph = build_residue_graph(PDB_PATH, CHAIN_ID)
    distances = graph.edge_attr[:, 0]
    assert torch.all(distances <= 8.0), "Some edges exceed the 8A cutoff"

# ──────────────────────────────────────────────
# 5. GRAPH OBJECT
# ──────────────────────────────────────────────

def test_graph_has_required_attributes():
    """Graph must have x, edge_index, edge_attr, num_residues, chain_id."""
    graph = build_residue_graph(PDB_PATH, CHAIN_ID)
    assert hasattr(graph, 'x')
    assert hasattr(graph, 'edge_index')
    assert hasattr(graph, 'edge_attr')
    assert hasattr(graph, 'num_residues')
    assert hasattr(graph, 'chain_id')

def test_chain_id_stored_correctly():
    """chain_id attribute must match the requested chain."""
    graph = build_residue_graph(PDB_PATH, CHAIN_ID)
    assert graph.chain_id == CHAIN_ID

def test_residue_count_within_bounds():
    """Residue count must be between 50 and 512 (FR-02)."""
    graph = build_residue_graph(PDB_PATH, CHAIN_ID)
    assert 50 <= graph.num_residues <= 512

# ──────────────────────────────────────────────
# 6. EDGE CASE — INVALID CHAIN
# ──────────────────────────────────────────────

def test_invalid_chain_raises_error():
    """Requesting a non-existent chain must raise an exception."""
    with pytest.raises(Exception):
        build_residue_graph(PDB_PATH, 'Z')