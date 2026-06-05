"""
ECABSD V3 Model — Explainable Cross-Attention Binding Site Discovery.

Architecture:
  GCNEncoderV3   : 6-layer GATv2 (256-dim, 4 heads, residual connections)
  CrossAttention : Bidirectional CrossPPI-style cross-attention (from cross_attention.py)
  Global Pooling : Mean pool → projected context added to local residue features
  Classifier     : 3-layer MLP (LayerNorm → ReLU → Dropout → Sigmoid)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import unbatch
from torch_geometric.nn import GATv2Conv

class CrossAttentionV3(nn.Module):
    def __init__(self, embed_dim, num_heads=4, dropout=0.3):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key_val):
        attn_out, attn_weights = self.mha(query, key_val, key_val)
        out1 = self.norm1(query + self.dropout(attn_out))
        ffn_out = self.ffn(out1)
        out2 = self.norm2(out1 + self.dropout(ffn_out))
        return out2, attn_weights



class GCNEncoderV3(nn.Module):
    """
    6-layer GATv2 encoder with residual connections, LayerNorm, and GELU.
    Input: 33-dim ESM-2 + geometric node features, 5-dim edge features.
    Output: 256-dim per-residue embeddings.
    """
    def __init__(
        self,
        input_dim: int = 33,
        hidden_dim: int = 256,
        edge_dim: int = 5,
        num_heads: int = 4,
        dropout: float = 0.3,
        num_layers: int = 6,
    ):
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        head_dim = hidden_dim // num_heads
        self.drop = nn.Dropout(dropout)
        self.num_layers = num_layers

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            if i == num_layers - 1:
                self.convs.append(
                    GATv2Conv(in_dim, hidden_dim, heads=1, edge_dim=edge_dim, dropout=dropout, concat=False)
                )
            else:
                self.convs.append(
                    GATv2Conv(in_dim, head_dim, heads=num_heads, edge_dim=edge_dim, dropout=dropout, concat=True)
                )
                self.norms.append(nn.LayerNorm(hidden_dim))

    def forward(self, x, edge_index, edge_attr):
        h = x
        for i, conv in enumerate(self.convs):
            h_new = conv(h, edge_index, edge_attr)
            if i < self.num_layers - 1:
                h_new = F.gelu(self.norms[i](h_new))
                if i > 0:
                    h_new = h_new + h
                h = self.drop(h_new)
            else:
                h = h_new + h
        return h


class ECABSDModelV3(nn.Module):
    """
    Full ECABSD V3 model.

    Parameters
    ----------
    input_dim     : Node feature dimension (33 = ESM-2 projected)
    hidden_dim    : Hidden dimension throughout (256)
    num_heads     : Attention heads for GATv2 and cross-attention (4)
    dropout       : Dropout probability (0.3)
    edge_dim      : Edge feature dimension (5)
    num_gcn_layers: Number of GATv2 layers (6)
    xattn_layers  : Number of bidirectional cross-attention layers (4)
    """
    def __init__(
        self,
        input_dim: int = 33,
        hidden_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.3,
        edge_dim: int = 5,
        num_gcn_layers: int = 6,
        xattn_layers: int = 4,
    ):
        super().__init__()

        # Structural encoder — shared weights for both chains
        self.gcn_encoder = GCNEncoderV3(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            edge_dim=edge_dim,
            num_heads=num_heads,
            dropout=dropout,
            num_layers=num_gcn_layers,
        )

        # Single-layer cross-attention matching checkpoint keys
        self.cross_attention = CrossAttentionV3(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        # Global context: mean-pool the cross-attended chain A → project → add to local features
        self.global_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

        self.norm_fuse = nn.LayerNorm(hidden_dim)

        # Per-residue classifier: 3-layer MLP
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

    def encode_chain(self, x, edge_index, edge_attr):
        """Encode a single chain through GATv2 encoder."""
        return self.gcn_encoder(x, edge_index, edge_attr)

    def forward(self, data_a, data_b=None):
        """
        Forward pass.

        Parameters
        ----------
        data_a : PyG Data — target chain (Chain A), labels predicted for this chain
        data_b : PyG Data — partner chain (Chain B), used as context via cross-attention
                           If None, self-attention fallback (chain A queries itself)

        Returns
        -------
        logits   : (N_a, 1) raw logits per residue
        attn_list: list of attention weight tensors per sample in batch
        """
        # Encode both chains through shared GATv2 encoder
        h_a = self.encode_chain(data_a.x, data_a.edge_index, data_a.edge_attr)
        h_b = self.encode_chain(data_b.x, data_b.edge_index, data_b.edge_attr) \
              if data_b is not None else h_a

        # Handle batching
        batch_a = data_a.batch if hasattr(data_a, 'batch') and data_a.batch is not None \
                  else torch.zeros(data_a.num_nodes, dtype=torch.long, device=data_a.x.device)
        batch_b = data_b.batch if (data_b is not None and hasattr(data_b, 'batch') and data_b.batch is not None) \
                  else torch.zeros(h_b.shape[0], dtype=torch.long, device=h_b.device)

        h_a_list = unbatch(h_a, batch_a)
        h_b_list = unbatch(h_b, batch_b)

        cross_out_list = []
        attn_list = []

        for h_a_single, h_b_single in zip(h_a_list, h_b_list):
            # Add batch dim: (N, D) -> (1, N, D)
            h_a_seq = h_a_single.unsqueeze(0)
            h_b_seq = h_b_single.unsqueeze(0)

            # Bidirectional cross-attention
            # cross_out: (1, N_a, D) — chain A enriched by chain B context
            # attn_weights: (1, H, N_a, N_b) — attention map for interpretability
            cross_out, attn_weights = self.cross_attention(h_a_seq, h_b_seq)

            # Global context pooling: summarise full chain A → add back to local features
            global_ctx = self.global_proj(cross_out.mean(dim=1, keepdim=True))
            h_fused_single = self.norm_fuse(cross_out + global_ctx)

            cross_out_list.append(h_fused_single.squeeze(0))
            attn_list.append(attn_weights.squeeze(0) if attn_weights is not None else None)

        # Concatenate batch back together
        h_fused = torch.cat(cross_out_list, dim=0)

        # Per-residue classification
        logits = self.classifier(h_fused)

        return logits, attn_list

    def predict(self, data_a, data_b=None, threshold: float = 0.5):
        """Inference mode — returns probabilities, binary labels, and attention weights."""
        self.eval()
        with torch.no_grad():
            logits, attn = self.forward(data_a, data_b)
            probs = torch.sigmoid(logits)
            labels = (probs.squeeze(-1) >= threshold).long()
        return probs, labels, attn


# Alias for backward compatibility — both names work
ECABSDModel = ECABSDModelV3
