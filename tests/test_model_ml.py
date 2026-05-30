"""
tests/test_model_ml.py
======================
Unit tests for ECABSD ML components.
"""

import pytest
import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch

ESM_DIM    = 33
EDGE_DIM   = 5
HIDDEN_DIM = 256
NUM_HEADS  = 4
NUM_GCN    = 6
DROPOUT    = 0.0


def make_graph(num_nodes: int = 20, num_edges: int = 60,
               with_labels: bool = True) -> Data:
    x          = torch.randn(num_nodes, ESM_DIM)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    edge_attr  = torch.randn(num_edges, EDGE_DIM)
    pos        = torch.randn(num_nodes, 3)
    data       = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, pos=pos)
    if with_labels:
        data.y = torch.randint(0, 2, (num_nodes,)).float()
    return data


def make_batch(num_graphs: int = 2, **kwargs) -> Batch:
    return Batch.from_data_list([make_graph(**kwargs) for _ in range(num_graphs)])


@pytest.fixture(scope="module")
def model():
    from models.ecabsd_v3_model import ECABSDModelV3
    m = ECABSDModelV3(
        input_dim=ESM_DIM,
        hidden_dim=HIDDEN_DIM,
        num_heads=NUM_HEADS,
        dropout=DROPOUT,
        edge_dim=EDGE_DIM,
        num_gcn_layers=NUM_GCN,
    )
    m.eval()
    return m


class TestECABSDModelForward:

    def test_output_shape_matches_chain_a_nodes(self, model):
        data_a = make_graph(num_nodes=30, num_edges=90)
        data_b = make_graph(num_nodes=25, num_edges=75)
        with torch.no_grad():
            logits, attn = model(data_a, data_b)
        assert logits.shape == (30, 1)

    def test_output_dtype_is_float32(self, model):
        data_a = make_graph(num_nodes=20, num_edges=60)
        data_b = make_graph(num_nodes=15, num_edges=45)
        with torch.no_grad():
            logits, _ = model(data_a, data_b)
        assert logits.dtype == torch.float32

    def test_no_nan_in_output(self, model):
        data_a = make_graph(num_nodes=20, num_edges=60)
        data_b = make_graph(num_nodes=20, num_edges=60)
        with torch.no_grad():
            logits, _ = model(data_a, data_b)
        assert not torch.isnan(logits).any()

    def test_probabilities_in_unit_interval(self, model):
        data_a = make_graph(num_nodes=20, num_edges=60)
        data_b = make_graph(num_nodes=20, num_edges=60)
        with torch.no_grad():
            logits, _ = model(data_a, data_b)
        probs = torch.sigmoid(logits)
        assert (probs >= 0).all() and (probs <= 1).all()

    def test_batched_input(self, model):
        batch_a = make_batch(num_graphs=2, num_nodes=20, num_edges=60)
        batch_b = make_batch(num_graphs=2, num_nodes=15, num_edges=45)
        with torch.no_grad():
            logits, _ = model(batch_a, batch_b)
        assert logits.shape[0] == 40

    def test_asymmetric_chain_sizes(self, model):
        data_a = make_graph(num_nodes=50, num_edges=150)
        data_b = make_graph(num_nodes=10, num_edges=30)
        with torch.no_grad():
            logits, _ = model(data_a, data_b)
        assert logits.shape == (50, 1)

    def test_gradient_flows(self, model):
        from models.ecabsd_v3_model import ECABSDModelV3
        m = ECABSDModelV3(
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
        assert len(grads) > 0
        assert any(g.abs().sum().item() > 0 for g in grads)


class TestGCNEncoder:

    @pytest.fixture
    def encoder(self):
        from models.ecabsd_v3_model import GCNEncoderV3
        return GCNEncoderV3(
            input_dim=ESM_DIM,
            hidden_dim=HIDDEN_DIM,
            edge_dim=EDGE_DIM,
            num_layers=NUM_GCN,
            dropout=DROPOUT,
        )

    def test_output_shape(self, encoder):
        data = make_graph(num_nodes=20, num_edges=60)
        with torch.no_grad():
            out = encoder(data.x, data.edge_index, data.edge_attr)
        assert out.shape == (20, HIDDEN_DIM)

    def test_no_nan_output(self, encoder):
        data = make_graph(num_nodes=20, num_edges=60)
        with torch.no_grad():
            out = encoder(data.x, data.edge_index, data.edge_attr)
        assert not torch.isnan(out).any()

    def test_different_graph_sizes(self, encoder):
        for n in [10, 30, 100]:
            data = make_graph(num_nodes=n, num_edges=n * 3)
            with torch.no_grad():
                out = encoder(data.x, data.edge_index, data.edge_attr)
            assert out.shape == (n, HIDDEN_DIM)


class TestCrossAttention:

    @pytest.fixture
    def cross_attn(self):
        from models.cross_attention import CrossAttention
        return CrossAttention(
            embed_dim=HIDDEN_DIM,
            num_heads=NUM_HEADS,
            dropout=DROPOUT,
        )

    def _make_embeddings(self, n_a=20, n_b=15):
        return torch.randn(1, n_a, HIDDEN_DIM), torch.randn(1, n_b, HIDDEN_DIM)

    def test_output_shape(self, cross_attn):
        feat_a, feat_b = self._make_embeddings(20, 15)
        with torch.no_grad():
            out, attn = cross_attn(feat_a, feat_b)
        assert out.shape == (1, 20, HIDDEN_DIM)

    def test_no_nan_output(self, cross_attn):
        feat_a, feat_b = self._make_embeddings(20, 15)
        with torch.no_grad():
            out, _ = cross_attn(feat_a, feat_b)
        assert not torch.isnan(out).any()

    def test_asymmetric_sequence_lengths(self, cross_attn):
        for n_a, n_b in [(5, 100), (100, 5), (50, 50)]:
            feat_a, feat_b = self._make_embeddings(n_a, n_b)
            with torch.no_grad():
                out, _ = cross_attn(feat_a, feat_b)
            assert out.shape == (1, n_a, HIDDEN_DIM)
