import torch
import torch.nn as nn

class BindingSiteClassifier(nn.Module):
    def __init__(self, input_dim=128):
        super(BindingSiteClassifier, self).__init__()

        self.linear1 = nn.Linear(input_dim, 64)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.sigmoid(x)
        return x