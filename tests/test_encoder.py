"""
Test script for ECABSD V3 Encoder.
Verifies GCNEncoderV3 forward pass on 1AY7 chain A.
"""
import torch
from models.encoder import GCNEncoderV3
from models.graph_construction import build_residue_graph

# Load graph — update path to your local 1AY7.pdb
data = build_residue_graph("1AY7.pdb", "A")

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

print("Input node features:", data.x.shape)
print("Edge index shape:   ", data.edge_index.shape)
print("Edge attr shape:    ", data.edge_attr.shape)
print("Output shape:       ", output.shape)
assert output.shape == (data.num_nodes, 256), f"Expected ({data.num_nodes}, 256), got {output.shape}"
print("✅ V3 Encoder test passed!")