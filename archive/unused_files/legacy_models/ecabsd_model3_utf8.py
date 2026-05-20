import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import unbatch
from torch_geometric.nn import GATConv

class GraphEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_heads=4, num_layers=3, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList()
        self.proj = nn.Linear(in_dim, hidden_dim)
        for _ in range(num_layers):
            self.layers.append(GATConv(hidden_dim, hidden_dim // num_heads, heads=num_heads, dropout=dropout))
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, edge_index):
        x = self.proj(x)
        for layer in self.layers:
            x_res = x
            x = layer(x, edge_index)
            x = F.elu(x)
            x = self.dropout(x)
            x = x + x_res
        return x

class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key_value):
        attn_out, attn_weights = self.mha(query, key_value, key_value)
        out = self.norm(query + self.dropout(attn_out))
        return out, attn_weights

class ECABSDModel(nn.Module):
    """
    Upgraded ECABSD Model integrating ESM2 Embeddings, PyG GAT, and Cross Attention.
    """
    def __init__(
        self,
        esm_dim: int = 1280, # Defaults to ESM2 650M
        hidden_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 3,
        dropout: float = 0.1,
        cross_attention: bool = True
    ):
        super().__init__()
        
        self.encoder = GraphEncoder(
            in_dim=esm_dim, 
            hidden_dim=hidden_dim, 
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout
        )
        
        self.use_cross_attention = cross_attention
        if self.use_cross_attention:
            self.cross_attn_a_to_b = CrossAttention(hidden_dim, num_heads, dropout)
            self.cross_attn_b_to_a = CrossAttention(hidden_dim, num_heads, dropout)
            
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2 if self.use_cross_attention else hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, data_a, data_b=None):
        # We assume data_a.x contains the ESM2 embeddings
        h_a = self.encoder(data_a.x, data_a.edge_index)
        
        if data_b is not None:
            h_b = self.encoder(data_b.x, data_b.edge_index)
        else:
            h_b = h_a
            data_b = data_a

        # Handle batching
        batch_a = data_a.batch if hasattr(data_a, 'batch') and data_a.batch is not None else torch.zeros(data_a.num_nodes, dtype=torch.long, device=data_a.x.device)
        batch_b = data_b.batch if hasattr(data_b, 'batch') and data_b.batch is not None else torch.zeros(data_b.num_nodes, dtype=torch.long, device=data_b.x.device)

        h_a_list = unbatch(h_a, batch_a)
        h_b_list = unbatch(h_b, batch_b)

        cross_out_list = []
        attn_list = []

        if self.use_cross_attention:
            for h_a_single, h_b_single in zip(h_a_list, h_b_list):
                h_a_seq = h_a_single.unsqueeze(0)
                h_b_seq = h_b_single.unsqueeze(0)
                
                h_a_cross, attn_ab = self.cross_attn_a_to_b(h_a_seq, h_b_seq)
                
                # Combine original structure representation with cross-chain context
                out_a_single = torch.cat([h_a_single, h_a_cross.squeeze(0)], dim=-1)
                
                cross_out_list.append(out_a_single)
                attn_list.append(attn_ab.squeeze(0))
                
            h_fused = torch.cat(cross_out_list, dim=0)
        else:
            h_fused = h_a
            
        logits = self.classifier(h_fused)
        
        return logits, attn_list

    def predict(self, data_a, data_b=None, threshold: float = 0.5):
        self.eval()
        with torch.no_grad():
            logits, attn = self.forward(data_a, data_b)
            probs = torch.sigmoid(logits)
            labels = (probs.squeeze(-1) >= threshold).long()
        return probs, labels, attn
