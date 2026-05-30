"""
ECABSD V3 Encoder — backward-compatible wrapper.

The V3 model (ecabsd_v3_model.py) uses GCNEncoderV3 internally.
This module re-exports GCNEncoderV3 as Encoder for backward compatibility
with any code that imports from models.encoder.

V3 Architecture:
  - 6-layer GATv2 (was 4-layer GCN in V1)
  - 256-dim hidden (was 128-dim in V1)
  - ESM-2 33-dim node features (was 23-dim one-hot in V1)
  - 5-dim SE(3)-aware edge features
  - Residual connections + LayerNorm + GELU
"""

from .ecabsd_v3_model import GCNEncoderV3

# Backward-compatible alias
Encoder = GCNEncoderV3

__all__ = ["Encoder", "GCNEncoderV3"]
