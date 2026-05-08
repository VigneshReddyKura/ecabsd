"""
CrossAttentionStack — stacked Transformer blocks for protein cross-attention.

Replaces the single nn.MultiheadAttention with a proper Transformer encoder
stack: N layers of (Pre-Norm Cross-Attention → Pre-Norm FFN) with residual
connections, plus a final self-attention block over chain A to let each
residue integrate its full context before classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class _CrossAttentionBlock(nn.Module):
    """
    One Transformer block: pre-norm cross-attention followed by pre-norm FFN.

    Q comes from chain A, K/V come from chain B (or self if B == A).
    """

    def __init__(self, embed_dim: int, num_heads: int, ffn_mult: int = 4, dropout: float = 0.1):
        super().__init__()
        # --- cross-attention ---
        self.norm_q   = nn.LayerNorm(embed_dim)
        self.norm_kv  = nn.LayerNorm(embed_dim)
        self.attn     = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.drop_attn = nn.Dropout(dropout)

        # --- feed-forward ---
        self.norm_ff = nn.LayerNorm(embed_dim)
        self.ff      = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * ffn_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * ffn_mult, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Pre-norm cross-attention with residual
        q_n  = self.norm_q(q)
        kv_n = self.norm_kv(kv)
        attn_out, attn_w = self.attn(q_n, kv_n, kv_n)
        q = q + self.drop_attn(attn_out)

        # Pre-norm FFN with residual
        q = q + self.ff(self.norm_ff(q))

        return q, attn_w


class _SelfAttentionBlock(nn.Module):
    """Standard self-attention Transformer block (pre-norm)."""

    def __init__(self, embed_dim: int, num_heads: int, ffn_mult: int = 4, dropout: float = 0.1):
        super().__init__()
        self.norm_sa = nn.LayerNorm(embed_dim)
        self.sa      = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.drop_sa = nn.Dropout(dropout)
        self.norm_ff = nn.LayerNorm(embed_dim)
        self.ff      = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * ffn_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * ffn_mult, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_n = self.norm_sa(x)
        sa_out, _ = self.sa(x_n, x_n, x_n)
        x = x + self.drop_sa(sa_out)
        x = x + self.ff(self.norm_ff(x))
        return x


class CrossAttention(nn.Module):
    """
    Stacked cross-attention module.

    Architecture:
        num_layers × _CrossAttentionBlock  (A attends to B)
        1 × _SelfAttentionBlock            (A integrates its full context)
        LayerNorm

    Returns the last layer's attention weights for interpretability.

    Parameters
    ----------
    embed_dim   : int   — token embedding dimension
    num_heads   : int   — attention heads per layer
    num_layers  : int   — number of cross-attention blocks (default 4)
    ffn_mult    : int   — FFN expansion factor (default 4)
    dropout     : float — dropout rate
    """

    def __init__(
        self,
        embed_dim:  int   = 256,
        num_heads:  int   = 8,
        num_layers: int   = 4,
        ffn_mult:   int   = 4,
        dropout:    float = 0.1,
    ):
        super().__init__()

        self.cross_blocks = nn.ModuleList([
            _CrossAttentionBlock(embed_dim, num_heads, ffn_mult, dropout)
            for _ in range(num_layers)
        ])

        # Final self-attention over chain A to integrate full context
        self.self_block = _SelfAttentionBlock(embed_dim, num_heads, ffn_mult, dropout)

        self.out_norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x1 : (1, N_a, D) — chain A queries
        x2 : (1, N_b, D) — chain B keys/values

        Returns
        -------
        out         : (1, N_a, D)
        attn_weights: (1, N_a, N_b) — from the last cross-attention block
        """
        last_attn = None
        q = x1
        for block in self.cross_blocks:
            q, last_attn = block(q, x2)

        # Self-attention over chain A
        q = self.self_block(q)
        q = self.out_norm(q)

        return q, last_attn