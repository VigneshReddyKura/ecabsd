import torch
import torch.nn as nn

class CrossAttention(nn.Module):
    def __init__(self, embed_dim=128, num_heads=8):
        super(CrossAttention, self).__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True
        )

    def forward(self, x1, x2):
        # Cross attention: x1 attends to x2
        out, attn_weights = self.attention(x1, x2, x2)
        return out, attn_weights