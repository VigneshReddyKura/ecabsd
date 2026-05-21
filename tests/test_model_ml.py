"""
tests/test_model_ml.py
======================
Unit tests for ECABSD ML components.

All synthetic graphs use the exact dimensions from config.yaml:
    x          = 33   (esm_dim)
    edge_attr  = 5    (edge_feature_dim)
    hidden_dim = 256
    num_heads  = 4    (cross-attention heads)
    num_gcn_layers = 6

Run with:
    pytest tests/test_model_ml.py -v
"""

import pytest
import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch


# ---------------------------------------------------------------------------
# Helpers — build minimal synthetic residue graphs
# ---------------------------------------------------------------------------

ESM_DIM    = 33
EDGE_DIM   = 5
HIDDEN_DIM = 256
NUM_HEADS  = 4
NUM_GCN    = 6
DROPOUT    = 0.0   # deterministic for testing


def make_graph(num_nodes: int = 20, num_edges: int = 60,
               with_labels: bool = True) -> Data:
    """Return a synthetic residue graph matching ECABSD feature dimensions."""
    x         = torch.randn(num_nodes, ESM_DIM)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    edge_attr  = torch.randn(num_edges, EDGE_DIM)
    pos        = torch.randn(num_nodes, 3)
    data       = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, pos=pos)
    if with_labels:
        data.y = torch.randint(0, 2, (num_nodes,)).float()
    return data


def make_batch(num_graphs: int = 2, **kwargs) -> Batch:
    """Return a PyG Batch of synthetic graphs."""
    return Batch.from_data_list([make_graph(**kwargs) for _ in range(num_graphs)])


# ---------------------------------------------------------------------------
# Fixture — load model once per session
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def model():
    from models.ecabsd_model import ECABSDModel
    m = ECABSDModel(
        input_dim=ESM_DIM,
        hidden_dim=HIDDEN_DIM,
        num_heads=NUM_HEADS,
        dropout=DROPOUT,
        edge_dim=EDGE_DIM,
        num_gcn_layers=NUM_GCN,
    )
    m.eval()
    return m


# ---------------------------------------------------------------------------
# 1. ECABSDModel — forward pass shape and dtype
# ---------------------------------------------------------------------------

class TestECABSDModelForward:

    def test_output_shape_matches_chain_a_nodes(self, model):
        """logits must have one value per residue in chain A."""
        data_a = make_graph(num_nodes=30, num_edges=90)
        data_b = make_graph(num_nodes=25, num_edges=75)
        with torch.no_grad():
            logits, attn = model(data_a, data_b)
        assert logits.shape == (30, 1), (
            f"Expected logits shape (30, 1), got {logits.shape}")

    def test_output_dtype_is_float32(self, model):
        data_a = make_graph(num_nodes=20, num_edges=60)
        data_b = make_graph(num_nodes=15, num_edges=45)
        with torch.no_grad():
            logits, _ = model(data_a, data_b)
        assert logits.dtype == torch.float32

    def test_no_nan_in_output(self, model):
        """Forward pass must never produce NaNs on valid input."""
        data_a = make_graph(num_nodes=20, num_edges=60)
        data_b = make_graph(num_nodes=20, num_edges=60)
        with torch.no_grad():
            logits, _ = model(data_a, data_b)
        assert not torch.isnan(logits).any(), "NaN detected in model output"

    def test_probabilities_in_unit_interval(self, model):
        """Sigmoid of logits must be in [0, 1]."""
        data_a = make_graph(num_nodes=20, num_edges=60)
        data_b = make_graph(num_nodes=20, num_edges=60)
        with torch.no_grad():
            logits, _ = model(data_a, data_b)
        probs = torch.sigmoid(logits)
        assert (probs >= 0).all() and (probs <= 1).all()

    def test_batched_input(self, model):
        """Model must handle PyG Batch objects (as used during training)."""
        batch_a = make_batch(num_graphs=2, num_nodes=20, num_edges=60)
        batch_b = make_batch(num_graphs=2, num_nodes=15, num_edges=45)
        with torch.no_grad():
            logits, _ = model(batch_a, batch_b)
        # total nodes across both graphs in chain A = 2 * 20 = 40
        assert logits.shape[0] == 40

    def test_asymmetric_chain_sizes(self, model):
        """Chain A and B may have different numbers of residues."""
        data_a = make_graph(num_nodes=50, num_edges=150)
        data_b = make_graph(num_nodes=10, num_edges=30)
        with torch.no_grad():
            logits, _ = model(data_a, data_b)
        assert logits.shape == (50, 1)

    def test_gradient_flows(self, model):
        """At least some parameters must receive non-zero gradients."""
        # Use a fresh model copy in train mode so we don't pollute the fixture
        from models.ecabsd_model import ECABSDModel
        m = ECABSDModel(
            input_dim=ESM_DIM, hidden_dim=HIDDEN_DIM, num_heads=NUM_HEADS,
            dropout=DROPOUT, edge_dim=EDGE_DIM, num_gcn_layers=NUM_GCN,
        )
        m.train()
        data_a = make_graph(num_nodes=20, num_edges=60)
        data_b = make_graph(num_nodes=20, num_edges=60)
        labels = torch.randint(0, 2, (20,)).float()
        logits, _ = m(data_a, data_b)
        loss = nn.BCEWithLogitsLoss()(logits.squeeze(-1), labels)
        loss.backward()
        grads = [p.grad for p in m.parameters() if p.grad is not None]
        assert len(grads) > 0, "No gradients computed"
        assert any(g.abs().sum().item() > 0 for g in grads), \
            "All gradients are zero"


# ---------------------------------------------------------------------------
# 2. CombinedLoss — numerical stability
# ---------------------------------------------------------------------------

class TestCombinedLoss:

    @pytest.fixture
    def loss_fn(self):
        from train import CombinedLoss
        return CombinedLoss(focal_alpha=0.75, focal_gamma=2.0, dice_weight=0.5)

    def test_no_nan_on_random_input(self, loss_fn):
        logits  = torch.randn(100)
        targets = torch.randint(0, 2, (100,)).float()
        loss    = loss_fn(logits, targets)
        assert not torch.isnan(loss), "CombinedLoss produced NaN on random input"

    def test_no_nan_all_positives(self, loss_fn):
        """Edge case: all labels are 1."""
        logits  = torch.randn(50)
        targets = torch.ones(50)
        loss    = loss_fn(logits, targets)
        assert not torch.isnan(loss)

    def test_no_nan_all_negatives(self, loss_fn):
        """Edge case: all labels are 0."""
        logits  = torch.randn(50)
        targets = torch.zeros(50)
        loss    = loss_fn(logits, targets)
        assert not torch.isnan(loss)

    def test_loss_is_positive(self, loss_fn):
        logits  = torch.randn(100)
        targets = torch.randint(0, 2, (100,)).float()
        loss    = loss_fn(logits, targets)
        assert loss.item() > 0

    def test_loss_is_scalar(self, loss_fn):
        logits  = torch.randn(100)
        targets = torch.randint(0, 2, (100,)).float()
        loss    = loss_fn(logits, targets)
        assert loss.ndim == 0, "Loss must be a scalar tensor"

    def test_focal_loss_alone_no_nan(self):
        from train import FocalLoss
        fn      = FocalLoss(alpha=0.75, gamma=2.0)
        logits  = torch.randn(100)
        targets = torch.randint(0, 2, (100,)).float()
        loss    = fn(logits, targets)
        assert not torch.isnan(loss)

    def test_soft_dice_loss_alone_no_nan(self):
        from train import SoftDiceLoss
        fn      = SoftDiceLoss(smooth=1.0)
        logits  = torch.randn(100)
        targets = torch.randint(0, 2, (100,)).float()
        loss    = fn(logits, targets)
        assert not torch.isnan(loss)

    def test_dice_weight_respected(self):
        """dice_weight=0 should give pure focal; dice_weight=1 pure dice."""
        from train import CombinedLoss, FocalLoss, SoftDiceLoss
        logits  = torch.randn(100)
        targets = torch.randint(0, 2, (100,)).float()

        pure_focal = FocalLoss(0.75, 2.0)(logits, targets).item()
        pure_dice  = SoftDiceLoss()(logits, targets).item()

        combo_focal = CombinedLoss(0.75, 2.0, dice_weight=0.0)(logits, targets).item()
        combo_dice  = CombinedLoss(0.75, 2.0, dice_weight=1.0)(logits, targets).item()

        assert abs(combo_focal - pure_focal) < 1e-5, \
            "dice_weight=0 should equal pure focal"
        assert abs(combo_dice - pure_dice) < 1e-5, \
            "dice_weight=1 should equal pure dice"


# ---------------------------------------------------------------------------
# 3. GCN Encoder — sub-module tests
# ---------------------------------------------------------------------------

class ChainEncoderWrapper(nn.Module):
    def __init__(self, input_dim, hidden_dim, edge_dim, num_gcn_layers, dropout):
        super().__init__()
        from models.ecabsd_model import GCNEncoder
        self.encoder = GCNEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            edge_dim=edge_dim,
            num_layers=num_gcn_layers,
            dropout=dropout,
        )
    def forward(self, data):
        return self.encoder(data.x, data.edge_index, data.edge_attr)


class TestGCNEncoder:

    @pytest.fixture
    def encoder(self):
        return ChainEncoderWrapper(
            input_dim=ESM_DIM,
            hidden_dim=HIDDEN_DIM,
            edge_dim=EDGE_DIM,
            num_gcn_layers=NUM_GCN,
            dropout=DROPOUT,
        )

    def test_output_shape(self, encoder):
        data = make_graph(num_nodes=20, num_edges=60)
        with torch.no_grad():
            out = encoder(data)
        assert out.shape == (20, HIDDEN_DIM), \
            f"Expected (20, {HIDDEN_DIM}), got {out.shape}"

    def test_no_nan_output(self, encoder):
        data = make_graph(num_nodes=20, num_edges=60)
        with torch.no_grad():
            out = encoder(data)
        assert not torch.isnan(out).any()

    def test_different_graph_sizes(self, encoder):
        for n in [10, 30, 100]:
            data = make_graph(num_nodes=n, num_edges=n * 3)
            with torch.no_grad():
                out = encoder(data)
            assert out.shape == (n, HIDDEN_DIM)


# ---------------------------------------------------------------------------
# 4. CrossAttention — sub-module tests
# ---------------------------------------------------------------------------

class CrossAttentionModuleWrapper(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout):
        super().__init__()
        from models.ecabsd_model import CrossAttention
        self.m = CrossAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
        )
    def forward(self, feat_a, feat_b):
        # Add batch dim: (1, n_a, dim)
        q = feat_a.unsqueeze(0)
        kv = feat_b.unsqueeze(0)
        out, attn = self.m(q, kv)
        # Remove batch dim from out and attn
        return out.squeeze(0), attn.squeeze(0)


class TestCrossAttention:

    @pytest.fixture
    def cross_attn(self):
        return CrossAttentionModuleWrapper(
            hidden_dim=HIDDEN_DIM,
            num_heads=NUM_HEADS,
            dropout=DROPOUT,
        )

    def _make_embeddings(self, n_a: int = 20, n_b: int = 15):
        feat_a = torch.randn(n_a, HIDDEN_DIM)
        feat_b = torch.randn(n_b, HIDDEN_DIM)
        return feat_a, feat_b

    def test_output_shape(self, cross_attn):
        feat_a, feat_b = self._make_embeddings(20, 15)
        with torch.no_grad():
            out, attn = cross_attn(feat_a, feat_b)
        assert out.shape == (20, HIDDEN_DIM), \
            f"Expected (20, {HIDDEN_DIM}), got {out.shape}"

    def test_attention_weights_sum_to_one(self, cross_attn):
        feat_a, feat_b = self._make_embeddings(20, 15)
        with torch.no_grad():
            _, attn = cross_attn(feat_a, feat_b)
        if attn is not None:
            row_sums = attn.sum(dim=-1)
            assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-4), \
                "Attention weights do not sum to 1 along key dimension"

    def test_no_nan_output(self, cross_attn):
        feat_a, feat_b = self._make_embeddings(20, 15)
        with torch.no_grad():
            out, _ = cross_attn(feat_a, feat_b)
        assert not torch.isnan(out).any()

    def test_asymmetric_sequence_lengths(self, cross_attn):
        """Cross-attention must handle very different chain sizes."""
        for n_a, n_b in [(5, 100), (100, 5), (50, 50)]:
            feat_a, feat_b = self._make_embeddings(n_a, n_b)
            with torch.no_grad():
                out, _ = cross_attn(feat_a, feat_b)
            assert out.shape == (n_a, HIDDEN_DIM)
