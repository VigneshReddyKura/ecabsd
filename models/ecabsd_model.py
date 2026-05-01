"""
ECABSDModel — End-to-end Equivariant Cross-Attention Binding Site Detection model.

Architecture:
    Protein A  ─→ [GCN Encoder → SE3 Refinement] ─→  repr_A  ─┐
                                                                ├─→ CrossAttention ─→ Classifier ─→ per-residue prob
    Protein B  ─→ [GCN Encoder → SE3 Refinement] ─→  repr_B  ─┘
"""

import torch
import torch.nn as nn

from .gcn_model import GCNEncoder
from .se3_model import SE3Transformer
from .cross_attention import CrossAttention
from .classifier import BindingSiteClassifier


class ECABSDModel(nn.Module):
    """
    Full ECABSD pipeline.

    Parameters
    ----------
    input_dim : int
        Node feature dimension (default 23: 20 AA + 3 SS).
    hidden_dim : int
        Hidden representation dimension.
    num_heads : int
        Number of attention heads in cross-attention.
    dropout : float
        Dropout probability.
    """

    def __init__(self, input_dim=23, hidden_dim=128, num_heads=8, dropout=0.1):
        super(ECABSDModel, self).__init__()

        # Shared encoder for both chains (weight sharing)
        self.gcn_encoder = GCNEncoder(input_dim=input_dim, hidden_dim=hidden_dim)
        self.se3_refine = SE3Transformer(input_dim=hidden_dim, hidden_dim=hidden_dim)

        # Cross-attention: chain A attends to chain B
        self.cross_attention = CrossAttention(embed_dim=hidden_dim, num_heads=num_heads)

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)

        # Layer normalization
        self.norm_a = nn.LayerNorm(hidden_dim)
        self.norm_b = nn.LayerNorm(hidden_dim)
        self.norm_cross = nn.LayerNorm(hidden_dim)

        # Per-residue binding site classifier
        self.classifier = BindingSiteClassifier(input_dim=hidden_dim)

    def encode_chain(self, x, edge_index):
        """Encode a single protein chain through GCN + SE3."""
        h = self.gcn_encoder(x, edge_index)
        h = self.se3_refine(h)
        return h

    def forward(self, data_a, data_b=None):
        """
        Forward pass.

        Parameters
        ----------
        data_a : torch_geometric.data.Data
            Graph for protein chain A (the target chain for binding prediction).
        data_b : torch_geometric.data.Data or None
            Graph for protein chain B (interaction partner).
            If None, self-attention on chain A is used.

        Returns
        -------
        pred : torch.Tensor
            Per-residue binding site probabilities for chain A, shape (N_a, 1).
        attn_weights : torch.Tensor
            Cross-attention weight matrix, shape (N_a, N_b).
        """
        # Encode chain A
        h_a = self.encode_chain(data_a.x, data_a.edge_index)
        h_a = self.norm_a(h_a)

        # Encode chain B (or use chain A for self-attention)
        if data_b is not None:
            h_b = self.encode_chain(data_b.x, data_b.edge_index)
            h_b = self.norm_b(h_b)
        else:
            h_b = h_a

        # Cross-attention: chain A attends to chain B
        # Add batch dimension for nn.MultiheadAttention: (1, N, D)
        h_a_seq = h_a.unsqueeze(0)
        h_b_seq = h_b.unsqueeze(0)

        cross_out, attn_weights = self.cross_attention(h_a_seq, h_b_seq)
        cross_out = cross_out.squeeze(0)  # (N_a, D)

        # Residual connection + norm
        h_fused = self.norm_cross(h_a + self.dropout(cross_out))

        # Per-residue classification
        pred = self.classifier(h_fused)  # (N_a, 1)

        # Squeeze attention weights
        attn_weights = attn_weights.squeeze(0)  # (N_a, N_b)

        return pred, attn_weights

    def predict(self, data_a, data_b=None, threshold=0.5):
        """
        Convenience method: returns binary predictions + probabilities.

        Returns
        -------
        probs : torch.Tensor   — (N_a, 1) probabilities
        labels : torch.Tensor  — (N_a,) binary labels
        attn : torch.Tensor    — attention weights
        """
        self.eval()
        with torch.no_grad():
            probs, attn = self.forward(data_a, data_b)
            labels = (probs.squeeze(-1) >= threshold).long()
        return probs, labels, attn
