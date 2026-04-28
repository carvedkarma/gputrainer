import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from .base import BaseModel

class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, dropout: float = 0.1,
                 alpha: float = 0.2, concat: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat
        
        self.W = nn.Linear(in_features, out_features, bias=False)
        self.a = nn.Parameter(torch.zeros(2 * out_features, 1))
        nn.init.xavier_uniform_(self.a)
        
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.dropout_layer = nn.Dropout(dropout)
        
    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        Wh = self.W(h)
        N = Wh.size(1)
        
        a_input = self._prepare_attentional_mechanism_input(Wh)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(-1))
        
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=-1)
        attention = self.dropout_layer(attention)
        
        h_prime = torch.matmul(attention, Wh)
        
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime
            
    def _prepare_attentional_mechanism_input(self, Wh: torch.Tensor) -> torch.Tensor:
        N = Wh.size(1)
        
        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=1)
        Wh_repeated_alternating = Wh.repeat(1, N, 1)
        
        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=-1)
        
        return all_combinations_matrix.view(-1, N, N, 2 * self.out_features)


class CrossAssetGNN(BaseModel):
    def __init__(
        self,
        input_dim: int,
        num_assets: int = 4,
        hidden_dim: int = 128,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.2,
        output_dim: int = 3
    ):
        super().__init__("cross_asset_gnn", input_dim, output_dim)
        
        self.num_assets = num_assets
        self.hidden_dim = hidden_dim
        self.total_input_dim = input_dim
        
        self.temporal_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.temporal_lstm = nn.LSTM(
            hidden_dim, hidden_dim // 2,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )
        self.temporal_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
        features_per_asset = max(input_dim // num_assets, 1)
        self.node_encoder = nn.Sequential(
            nn.Linear(features_per_asset, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.attention_layers = nn.ModuleList()
        for _ in range(num_layers):
            heads = nn.ModuleList([
                GraphAttentionLayer(hidden_dim, hidden_dim // num_heads, dropout)
                for _ in range(num_heads)
            ])
            self.attention_layers.append(heads)
            
        self.edge_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        self.temporal_aggregator = nn.LSTM(
            hidden_dim, hidden_dim // 2,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * num_assets, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def compute_adjacency(self, h: torch.Tensor) -> torch.Tensor:
        batch_size, num_nodes, hidden = h.shape
        
        h_repeat_1 = h.unsqueeze(2).repeat(1, 1, num_nodes, 1)
        h_repeat_2 = h.unsqueeze(1).repeat(1, num_nodes, 1, 1)
        
        edge_features = torch.cat([h_repeat_1, h_repeat_2], dim=-1)
        edge_features = edge_features.view(batch_size, num_nodes * num_nodes, -1)
        
        adj = self.edge_predictor(edge_features)
        adj = adj.view(batch_size, num_nodes, num_nodes)
        
        return adj
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            h = self.temporal_encoder(x)
            lstm_out, _ = self.temporal_lstm(h)
            logits = self.temporal_classifier(lstm_out[:, -1, :])
            return logits
        
        batch_size, seq_len, num_assets, features = x.shape
        
        x = x.view(batch_size * seq_len, num_assets, features)
        h = self.node_encoder(x)
        
        adj = self.compute_adjacency(h)
        
        for heads in self.attention_layers:
            head_outputs = [head(h, adj) for head in heads]
            h = torch.cat(head_outputs, dim=-1)
            
        h = h.view(batch_size, seq_len, num_assets, -1)
        
        asset_features = []
        for i in range(num_assets):
            asset_h = h[:, :, i, :]
            lstm_out, _ = self.temporal_aggregator(asset_h)
            asset_features.append(lstm_out[:, -1, :])
            
        combined = torch.cat(asset_features, dim=-1)
        
        logits = self.classifier(combined)
        return logits
    
    def get_asset_relations(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            batch_size = x.shape[0]
            return torch.eye(self.num_assets, device=x.device).unsqueeze(0).repeat(batch_size, 1, 1)
        
        batch_size, seq_len, num_assets, features = x.shape
        x = x.view(batch_size * seq_len, num_assets, features)
        h = self.node_encoder(x)
        adj = self.compute_adjacency(h)
        adj = adj.view(batch_size, seq_len, num_assets, num_assets)
        return adj.mean(dim=1)


class TemporalGNN(BaseModel):
    def __init__(
        self,
        input_dim: int,
        num_nodes: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        num_heads: int = 4,
        temporal_window: int = 10,
        dropout: float = 0.2,
        output_dim: int = 3
    ):
        super().__init__("temporal_gnn", input_dim, output_dim)
        
        self.num_nodes = num_nodes
        self.temporal_window = temporal_window
        self.total_input_dim = input_dim
        
        self.temporal_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.temporal_attention_fallback = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.temporal_output = nn.Linear(hidden_dim, output_dim)
        
        features_per_node = max(input_dim // num_nodes, 1)
        self.spatial_encoder = nn.Linear(features_per_node, hidden_dim)
        
        self.spatial_attention = nn.ModuleList([
            GraphAttentionLayer(hidden_dim, hidden_dim, dropout)
            for _ in range(num_layers)
        ])
        
        self.temporal_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.output = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x: torch.Tensor, adj: Optional[torch.Tensor] = None) -> torch.Tensor:
        if x.dim() == 3:
            h = self.temporal_encoder(x)
            attn_out, _ = self.temporal_attention_fallback(h, h, h)
            logits = self.temporal_output(attn_out[:, -1, :])
            return logits
        
        batch_size, seq_len, num_nodes, features = x.shape
        
        if adj is None:
            adj = torch.ones(batch_size, num_nodes, num_nodes, device=x.device)
        
        x = x.view(-1, features)
        h = self.spatial_encoder(x)
        h = h.view(batch_size * seq_len, num_nodes, -1)
        
        adj_expanded = adj.unsqueeze(1).repeat(1, seq_len, 1, 1)
        adj_expanded = adj_expanded.view(batch_size * seq_len, num_nodes, num_nodes)
        
        for layer in self.spatial_attention:
            h = layer(h, adj_expanded)
            
        h = h.view(batch_size, seq_len, num_nodes, -1)
        
        node_outputs = []
        for n in range(num_nodes):
            node_h = h[:, :, n, :]
            attn_out, _ = self.temporal_attention(node_h, node_h, node_h)
            node_outputs.append(attn_out[:, -1, :])
            
        spatial_final = h[:, -1, 0, :]
        temporal_final = node_outputs[0]
        
        fused = self.fusion(torch.cat([spatial_final, temporal_final], dim=-1))
        
        logits = self.output(fused)
        return logits
