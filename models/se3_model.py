import torch
import torch.nn as nn

class SE3Transformer(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=128):
        super(SE3Transformer, self).__init__()

        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x