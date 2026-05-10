from models.ecabsd_model import ECABSDModel
from data.dataset import BindingSiteDataset
import torch
ds = BindingSiteDataset('data/processed','data/splits.csv','train')
s = ds[0]
m = ECABSDModel(input_dim=23, hidden_dim=256, num_heads=8, dropout=0.3)
out = m(s['data_a'], s['data_b'])
print(type(out))
if isinstance(out, tuple):
    for i,o in enumerate(out):
        print(i, type(o), o.shape if hasattr(o,'shape') else o)