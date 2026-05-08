"""
SE3Refinement — deep residual MLP that refines GATv2 node embeddings.

Replaces the original 2-linear-layer stub. Uses 3 residual blocks with
LayerNorm, GELU, and Dropout — providing significantly more expressive
spatial refinement of per-residue representations.
"""

import torch
import torch.nn as nn


class _ResBlock(nn.Module):
    """Single pre-norm residual feed-forward block."""

    def __init__(self, dim: int, expansion: int = 4, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.net  = nn.Sequential(
            nn.Linear(dim, dim * expansion),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * expansion, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(self.norm(x))


class SE3Transformer(nn.Module):
    """
    Deep residual MLP refinement block (replaces the original 2-layer stub).

    Parameters
    ----------
    input_dim : int
        Input/output feature dimension.
    hidden_dim : int
        Passed for API compatibility — must equal input_dim.
    num_blocks : int
        Number of residual blocks (default 3).
    dropout : float
        Dropout rate inside each block.
    """

    def __init__(
        self,
        input_dim:  int = 256,
        hidden_dim: int = 256,
        num_blocks: int = 3,
        dropout:    float = 0.1,
    ):
        super().__init__()
        assert input_dim == hidden_dim, (
            "SE3Transformer requires input_dim == hidden_dim for residual path."
        )
        self.blocks = nn.ModuleList(
            [_ResBlock(input_dim, expansion=4, dropout=dropout) for _ in range(num_blocks)]
        )
        self.norm = nn.LayerNorm(input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return self.norm(x)