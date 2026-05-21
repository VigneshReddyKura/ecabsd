import torch
import pytest
from torch_geometric.data import Data, Batch
from models.ecabsd_model import ECABSDModel, GCNEncoder, CrossAttention
from train import CombinedLoss

def create_synthetic_data(num_nodes=50, num_edges=100, x_dim=33, edge_dim=5):
    # Synthetic node features
    x = torch.randn(num_nodes, x_dim)
    # Synthetic edges (random pairs)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    # Synthetic edge attributes
    edge_attr = torch.randn(num_edges, edge_dim)
    # Synthetic binary labels
    y = torch.randint(0, 2, (num_nodes,)).float()
    
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

def test_ecabsd_model_forward():
    model = ECABSDModel(
        input_dim=33,
        hidden_dim=256,
        num_heads=4,
        edge_dim=5,
        num_gcn_layers=2 # Keep it small for fast testing
    )
    
    data_a = create_synthetic_data(num_nodes=40)
    data_b = create_synthetic_data(num_nodes=45)
    
    # Needs to be batched to simulate collate_fn behavior
    batch_a = Batch.from_data_list([data_a])
    batch_b = Batch.from_data_list([data_b])
    
    logits, attn = model(batch_a, batch_b)
    
    # Output should have shape [total_nodes, 1]
    assert logits.shape == (40, 1)
    assert not torch.isnan(logits).any()
    
    # Check predict method
    probs, labels, attn_out = model.predict(batch_a, batch_b)
    assert probs.shape == (40, 1)
    assert labels.shape == (40, 1)

def test_combined_loss():
    loss_fn = CombinedLoss()
    
    logits = torch.randn(100)
    targets = torch.randint(0, 2, (100,)).float()
    
    loss = loss_fn(logits, targets)
    
    assert loss.dim() == 0
    assert not torch.isnan(loss)
    assert loss.item() > 0

def test_gcn_encoder():
    encoder = GCNEncoder(
        input_dim=33,
        hidden_dim=256,
        edge_dim=5,
        num_heads=4,
        num_layers=2
    )
    
    data = create_synthetic_data(num_nodes=30)
    out = encoder(data.x, data.edge_index, data.edge_attr)
    
    # Hidden representation should have shape [num_nodes, hidden_dim]
    assert out.shape == (30, 256)
