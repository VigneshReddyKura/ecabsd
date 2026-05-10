from models.graph_construction import build_residue_graph
import torch
d = build_residue_graph('1AY7.pdb', 'A')
print('Feature dim:', d.x.shape[1])