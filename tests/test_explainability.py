"""
tests/test_explainability.py
============================
Unit tests for ECABSD Explainability modules (Grad-CAM & Attention Rollout).
"""

import pytest
import numpy as np
import torch
from torch_geometric.data import Data


ESM_DIM    = 33
EDGE_DIM   = 5
HIDDEN_DIM = 256
NUM_HEADS  = 4
NUM_GCN    = 6
DROPOUT    = 0.0


def make_graph(num_nodes: int = 20, num_edges: int = 60) -> Data:
    x          = torch.randn(num_nodes, ESM_DIM)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    edge_attr  = torch.randn(num_edges, EDGE_DIM)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


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


# ---------------------------------------------------------------------------
# Grad-CAM
# ---------------------------------------------------------------------------

class TestGradCAM:

    def test_saliency_shape_matches_chain_a_nodes(self, model):
        """Saliency output must be 1-D with length == chain A nodes."""
        from explainability.gradcam import GradCAM
        data_a = make_graph(num_nodes=30)
        data_b = make_graph(num_nodes=25)
        gradcam = GradCAM(model, target_layer_idx=-1)
        saliency = gradcam.compute(data_a, data_b)
        gradcam.remove_hooks()
        assert saliency.shape == (30,)

    def test_saliency_no_nan(self, model):
        """Saliency must not contain NaN values."""
        from explainability.gradcam import GradCAM
        data_a = make_graph(num_nodes=20)
        data_b = make_graph(num_nodes=20)
        gradcam = GradCAM(model, target_layer_idx=-1)
        saliency = gradcam.compute(data_a, data_b)
        gradcam.remove_hooks()
        assert not np.isnan(saliency).any()

    def test_saliency_normalized_to_unit_interval(self, model):
        """Saliency must be normalized to [0, 1]."""
        from explainability.gradcam import GradCAM
        data_a = make_graph(num_nodes=30)
        data_b = make_graph(num_nodes=25)
        gradcam = GradCAM(model, target_layer_idx=-1)
        saliency = gradcam.compute(data_a, data_b)
        gradcam.remove_hooks()
        assert saliency.min() >= 0.0
        assert saliency.max() <= 1.0 + 1e-6  # small float tolerance

    def test_saliency_without_chain_b(self, model):
        """Grad-CAM must work even without a partner chain (data_b=None)."""
        from explainability.gradcam import GradCAM
        data_a = make_graph(num_nodes=20)
        gradcam = GradCAM(model, target_layer_idx=-1)
        saliency = gradcam.compute(data_a, data_b=None)
        gradcam.remove_hooks()
        assert saliency.shape == (20,)

    def test_target_layer_first_conv(self, model):
        """Target layer index=0 (first layer) should also produce a valid result."""
        from explainability.gradcam import GradCAM
        data_a = make_graph(num_nodes=20)
        data_b = make_graph(num_nodes=20)
        gradcam = GradCAM(model, target_layer_idx=0)
        saliency = gradcam.compute(data_a, data_b)
        gradcam.remove_hooks()
        assert saliency.shape == (20,)
        assert not np.isnan(saliency).any()


# ---------------------------------------------------------------------------
# Attention Rollout
# ---------------------------------------------------------------------------

class TestAttentionRollout:

    def test_scores_shape_matches_chain_a_nodes(self, model):
        """Rollout score array must be 1-D with length == chain A nodes."""
        from explainability.attention_rollout import AttentionRollout
        data_a = make_graph(num_nodes=30)
        data_b = make_graph(num_nodes=25)
        rollout = AttentionRollout(model)
        scores, attn_matrix = rollout.compute(data_a, data_b)
        rollout.remove_hook()
        assert scores.shape == (30,)

    def test_scores_no_nan(self, model):
        """Scores must not contain NaN values."""
        from explainability.attention_rollout import AttentionRollout
        data_a = make_graph(num_nodes=20)
        data_b = make_graph(num_nodes=20)
        rollout = AttentionRollout(model)
        scores, _ = rollout.compute(data_a, data_b)
        rollout.remove_hook()
        assert not np.isnan(scores).any()

    def test_scores_normalized_to_unit_interval(self, model):
        """Scores must be normalized to [0, 1]."""
        from explainability.attention_rollout import AttentionRollout
        data_a = make_graph(num_nodes=30)
        data_b = make_graph(num_nodes=25)
        rollout = AttentionRollout(model)
        scores, _ = rollout.compute(data_a, data_b)
        rollout.remove_hook()
        assert scores.min() >= 0.0
        assert scores.max() <= 1.0 + 1e-6

    def test_attn_matrix_shape(self, model):
        """Attention matrix shape should be (N_a, N_b)."""
        from explainability.attention_rollout import AttentionRollout
        data_a = make_graph(num_nodes=30)
        data_b = make_graph(num_nodes=25)
        rollout = AttentionRollout(model)
        _, attn_matrix = rollout.compute(data_a, data_b)
        rollout.remove_hook()
        # N_a axis must be 30; N_b can vary depending on pooling but must be >0
        assert attn_matrix.ndim == 2
        assert attn_matrix.shape[0] == 30
        assert attn_matrix.shape[1] > 0
