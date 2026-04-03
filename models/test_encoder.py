import torch
from encoder import Encoder
from graph_construction import build_residue_graph

# Load graph
data = build_residue_graph(r"C:\Users\vitta\ecabsd\1AY7.pdb", "A")

# Initialize model
model = Encoder()

# Forward pass
output = model(data)

print("Output shape:", output.shape)