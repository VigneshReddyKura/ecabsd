"""
ECABSDModel — End-to-end Equivariant Cross-Attention Binding Site Detection.

Architecture (v3 — "best model"):
    Protein A  →  GATv2 Encoder (4L, edge-aware, residual)
               →  SE3 Refinement (3× residual MLP blocks)
               →  LayerNorm
               ─┐
                ├──→  CrossAttentionStack (4× cross-attn + 1× self-attn)
               ─┘                        ↑ Protein B (same pipeline)
               →  Classifier (5-layer residual MLP) → logit

Notes
-----
- Outputs RAW LOGITS. Apply torch.sigmoid() at inference.
- Use BCEWithLogitsLoss or FocalLoss during training.
- Edge features (distance + 3D unit vector) are consumed by GATv2Conv.
- CrossAttentionStack stacks 4 cross-attention Transformer blocks + 1 self-attention
  so chain A integrates both partner context and its own sequence context.
"""

import torch
import torch.nn as nn

from .gcn_model       import GCNEncoder
from .se3_model       import SE3Transformer
from .cross_attention import CrossAttention
from .classifier      import BindingSiteClassifier


class ECABSDModel(nn.Module):
    """
    Full ECABSD v3 pipeline.

    Parameters
    ----------
    input_dim   : int   — Node feature dimension (23 with current data; 26 after reprocess)
    hidden_dim  : int   — Hidden representation dimension
    num_heads   : int   — Attention heads in GATv2 and cross-attention
    dropout     : float — Dropout probability
    edge_dim    : int   — Edge feature dimension (4: distance + 3D unit vector)
    num_ca_layers : int — Number of cross-attention Transformer blocks
    """

    def __init__(
        self,
        input_dim:    int   = 23,
        hidden_dim:   int   = 256,
        num_heads:    int   = 8,
        dropout:      float = 0.3,
        edge_dim:     int   = 4,
        num_ca_layers: int  = 4,
    ):
        super().__init__()

        # ── Shared encoder (weight-sharing across both chains) ───────────────
        self.gcn_encoder = GCNEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            edge_dim=edge_dim,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.se3_refine = SE3Transformer(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_blocks=3,
            dropout=dropout,
        )

        # ── Per-chain norms ───────────────────────────────────────────────────
        self.norm_a = nn.LayerNorm(hidden_dim)
        self.norm_b = nn.LayerNorm(hidden_dim)

        # ── Cross-attention stack ─────────────────────────────────────────────
        self.cross_attention = CrossAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_ca_layers,
            ffn_mult=4,
            dropout=dropout,
        )

        # ── Fusion residual + norm ────────────────────────────────────────────
        self.dropout    = nn.Dropout(dropout)
        self.norm_cross = nn.LayerNorm(hidden_dim)

        # ── Binding site classifier (outputs raw logits) ──────────────────────
        self.classifier = BindingSiteClassifier(
            input_dim=hidden_dim,
            dropout=dropout,
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def encode_chain(
        self,
        x:          torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr:  torch.Tensor,
    ) -> torch.Tensor:
        """GATv2 encoding + SE3 residual refinement."""
        h = self.gcn_encoder(x, edge_index, edge_attr)
        h = self.se3_refine(h)
        return h

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, data_a, data_b=None):
        """
        Parameters
        ----------
        data_a : torch_geometric.data.Data — target chain (binding site predicted here)
        data_b : torch_geometric.data.Data | None — partner chain; uses self-attn if None

        Returns
        -------
        logits       : (N_a, 1) — raw logits; sigmoid → probabilities
        attn_weights : (N_a, N_b) — last cross-attention layer weights
        """
        # Encode chain A
        h_a = self.encode_chain(data_a.x, data_a.edge_index, data_a.edge_attr)
        h_a = self.norm_a(h_a)

        # Encode chain B (or reuse A for self-attention)
        if data_b is not None:
            h_b = self.encode_chain(data_b.x, data_b.edge_index, data_b.edge_attr)
            h_b = self.norm_b(h_b)
        else:
            h_b = h_a

        # Cross-attention stack — add batch dim: (1, N, D)
        cross_out, attn_weights = self.cross_attention(
            h_a.unsqueeze(0),
            h_b.unsqueeze(0),
        )
        cross_out = cross_out.squeeze(0)          # (N_a, D)

        # Residual + norm
        h_fused = self.norm_cross(h_a + self.dropout(cross_out))

        # Per-residue logits
        logits = self.classifier(h_fused)         # (N_a, 1)

        return logits, attn_weights.squeeze(0)    # (N_a, N_b)

    def predict(self, data_a, data_b=None, threshold: float = 0.5):
        """Inference convenience: returns (probs, binary_labels, attn_weights)."""
        self.eval()
        with torch.no_grad():
            logits, attn = self.forward(data_a, data_b)
            probs  = torch.sigmoid(logits)
            labels = (probs.squeeze(-1) >= threshold).long()
        return probs, labels, attn
