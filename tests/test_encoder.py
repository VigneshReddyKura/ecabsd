"""
Test script for ECABSD V3 Encoder.
Verifies GCNEncoderV3 forward pass on 1AY7 chain A.
"""
import torch
from models.encoder import GCNEncoderV3
from models.graph_construction import build_residue_graph

import os

def test_encoder_forward():
    pdb_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'sample', '1AY7.pdb'))
    data = build_residue_graph(pdb_path, "A")
    
    # Initialize V3 encoder (matches config.yaml)
    model = GCNEncoderV3(
        input_dim=33,
        hidden_dim=256,
        edge_dim=5,
        num_heads=4,
        dropout=0.0,  # 0 for testing
        num_layers=6,
    )
    
    # Forward pass
    output = model(data.x, data.edge_index, data.edge_attr)
    
    assert output.shape == (data.num_nodes, 256), f"Expected ({data.num_nodes}, 256), got {output.shape}"