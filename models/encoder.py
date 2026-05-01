import torch
from .gcn_model import GCNEncoder
from .se3_model import SE3Transformer

class Encoder(torch.nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.gcn = GCNEncoder()
        self.se3 = SE3Transformer()

    def forward(self, data):
        x = self.gcn(data.x, data.edge_index)
        x = self.se3(x)
        return x