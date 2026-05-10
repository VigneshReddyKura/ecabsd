from models.ecabsd_model import ECABSDModel
import torch
m = ECABSDModel(input_dim=23, hidden_dim=256, num_heads=8, dropout=0.3)
print('Params:', sum(p.numel() for p in m.parameters()))
x = torch.randn(10, 23)
ei = torch.tensor([[0,1,2,3,4],[1,2,3,4,5]], dtype=torch.long)
ea = torch.randn(5, 4)
from torch_geometric.data import Data
d = Data(x=x, edge_index=ei, edge_attr=ea)
out, _ = m(d, None)
print('Output shape:', out.shape)
print('Model OK')